import os
import torch
import torch.nn as nn
from transformers import DeformableDetrModel, DeformableDetrConfig

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

LOCAL_DEFORMABLE_DETR_PATH = os.environ.get("LOCAL_DEFORMABLE_DETR_PATH", "")

def _cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

def _swap_slot0_per_batch(hidden_states, best_idx):
    B, Q, D = hidden_states.shape
    out = hidden_states.clone()
    ar = torch.arange(B, device=hidden_states.device)
    tmp = out[ar, 0, :].clone()
    out[ar, 0, :] = out[ar, best_idx, :]
    out[ar, best_idx, :] = tmp
    return out

def _best_iou_query(provisional_boxes, ref_box_xyxy):
    pb_xyxy = _cxcywh_to_xyxy(provisional_boxes)           # (B,Q,4)
    ref = ref_box_xyxy.unsqueeze(1).expand_as(pb_xyxy)     # (B,Q,4)
    inter_lt = torch.max(pb_xyxy[..., :2], ref[..., :2])
    inter_rb = torch.min(pb_xyxy[..., 2:], ref[..., 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_p = (pb_xyxy[..., 2] - pb_xyxy[..., 0]).clamp(0) * (pb_xyxy[..., 3] - pb_xyxy[..., 1]).clamp(0)
    area_r = (ref[..., 2] - ref[..., 0]).clamp(0) * (ref[..., 3] - ref[..., 1]).clamp(0)
    union = (area_p + area_r - inter).clamp(min=1e-6)
    iou = inter / union                                    # (B,Q)
    best_q = iou.argmax(dim=1)
    return best_q

class MinimalTrackFormer(nn.Module):
    """
    Deformable-DETR + temporal attention + simple trajectory head.
    """
    def __init__(self, pretrained_path=None, config=None, num_queries=100, hidden_dim=256, horizon=10):
        super().__init__()

        if not (LOCAL_DEFORMABLE_DETR_PATH and os.path.exists(LOCAL_DEFORMABLE_DETR_PATH)):
            raise FileNotFoundError(f"LOCAL_DEFORMABLE_DETR_PATH not found: {LOCAL_DEFORMABLE_DETR_PATH}")

        cfg = DeformableDetrConfig.from_pretrained(
            LOCAL_DEFORMABLE_DETR_PATH, local_files_only=True
        )
        cfg.output_hidden_states = True
        cfg.num_labels = 2
        cfg.num_queries = int(num_queries)
        cfg.use_pretrained_backbone = False

        self.detector = DeformableDetrModel.from_pretrained(
            LOCAL_DEFORMABLE_DETR_PATH,
            config=cfg, ignore_mismatched_sizes=True, local_files_only=True
        )
        hidden_dim = cfg.d_model

        self.track_queries = nn.Parameter(torch.randn(num_queries, hidden_dim))

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )

        self.class_embed = nn.Linear(hidden_dim, cfg.num_labels)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4), nn.Sigmoid()
        )

        self.horizon = int(horizon)
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * self.horizon)
        )

    def forward(
        self,
        pixel_values,                  # (B,T,C,H,W)
        prev_track_memory=None,
        return_track_memory=False,
        predict_future=False,
        *,
        gt_boxes_first_k=None,         # (B,K,4) normalized cxcywh
        k_burnin: int = 3,
        priming_prob: float = 1.0,
        disable_temporal: bool = False,
        force_prime: bool = False,     
    ):
        B, T, C, H, W = pixel_values.shape
        all_logits, all_boxes_out, all_boxes_raw, all_memories = [], [], [], []
        memory, last_hidden = prev_track_memory, None

        last_box_xyxy = None

        for t in range(T):
            frame = pixel_values[:, t]  # (B,3,H,W)
            det_out = self.detector(pixel_values=frame, output_hidden_states=True, return_dict=True)
            hidden_states = det_out.last_hidden_state  # (B,Q,D)

            provisional_boxes = self.bbox_embed(hidden_states)   # (B,Q,4) normalized cxcywh

            #GT-priming
            do_prime = (
                ((self.training and (gt_boxes_first_k is not None) and (t < int(k_burnin)) and
                  (priming_prob >= 1.0 or torch.rand(1, device=hidden_states.device).item() < priming_prob))
                 )
                or (force_prime and (gt_boxes_first_k is not None) and (t < int(k_burnin)))
            )

            if do_prime:
                with torch.no_grad():
                    gt_t = gt_boxes_first_k[:, t, :]  # (B,4) cxcywh
                    best_q = _best_iou_query(provisional_boxes, _cxcywh_to_xyxy(gt_t))
                hidden_states = _swap_slot0_per_batch(hidden_states, best_q)
            else:
                if last_box_xyxy is not None:
                    with torch.no_grad():
                        best_q = _best_iou_query(provisional_boxes, last_box_xyxy)
                    hidden_states = _swap_slot0_per_batch(hidden_states, best_q)

            # Temporal attention
            if (not disable_temporal) and (memory is not None):
                attn_output, _ = self.temporal_attention(hidden_states, memory, memory)
                hidden_states = hidden_states + attn_output
                hidden_states = hidden_states + self.ffn(hidden_states)

            # Heads
            pred_logits = self.class_embed(hidden_states)        # (B,Q,2)
            pred_boxes_raw = self.bbox_embed(hidden_states)      # (B,Q,4) for loss
            pred_boxes_out = pred_boxes_raw.clone()              # copy for output/vis

            if do_prime:

                pred_boxes_out[:, 0, :] = gt_boxes_first_k[:, t, :]


            with torch.no_grad():
                last_box_xyxy = _cxcywh_to_xyxy(pred_boxes_out[:, 0, :])  # (B,4)

            all_logits.append(pred_logits)
            all_boxes_raw.append(pred_boxes_raw)
            all_boxes_out.append(pred_boxes_out)

            last_hidden = hidden_states
            memory = hidden_states if self.training else hidden_states.detach()
            all_memories.append(memory)

        pred_logits    = torch.stack(all_logits, dim=1)     # (B,T,Q,2)
        pred_boxes     = torch.stack(all_boxes_out, dim=1)  # (B,T,Q,4)
        pred_boxes_raw = torch.stack(all_boxes_raw, dim=1)  # (B,T,Q,4)

        result = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_boxes_raw": pred_boxes_raw,
            "burnin": int(k_burnin)
        }


        if predict_future:
            logits_last = pred_logits[:, -1]
            scores = logits_last.softmax(-1)[..., 0]
            top_idx = scores.argmax(dim=-1)
            B_idx = torch.arange(B, device=top_idx.device)
            tgt_state = last_hidden[B_idx, top_idx, :]
            last_box  = pred_boxes[:, -1][B_idx, top_idx, :]
            offs = self.traj_head(tgt_state).view(B, self.horizon, 2).cumsum(dim=1)
            cxcy = (last_box[:, :2].unsqueeze(1) + offs).clamp(0, 1)
            wh   = last_box[:, 2:].unsqueeze(1).expand(-1, self.horizon, -1)
            result["future_boxes"] = torch.cat([cxcy, wh], dim=-1)
            result["future_offsets"] = offs


        with torch.no_grad():
            result["scores"] = result["pred_logits"].softmax(-1)[..., 0]  # (B,T,Q)

        if return_track_memory:
            result["track_memory"] = memory
        return result

    def predict_single_object(self, pixel_values, prev_track_memory=None):
        output = self.forward(pixel_values, prev_track_memory, return_track_memory=True, force_prime=False)
        probs_last = output["pred_logits"].softmax(-1)[0, -1]
        boxes_last = output["pred_boxes"][0, -1]
        score, top_idx = probs_last[:, 0].max(dim=0)
        box = boxes_last[top_idx]
        return box.detach().cpu(), score.item(), output["track_memory"]
