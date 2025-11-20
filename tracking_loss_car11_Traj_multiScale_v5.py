import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou

# Helper: cxcywh -> xyxy
def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1: x = x.unsqueeze(0)
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack((x_c-0.5*w, y_c-0.5*h, x_c+0.5*w, y_c+0.5*h), dim=-1)

class SOTCriterion(nn.Module):
    """
    Unified loss for single-object tracking and short-horizon forecasting.
    """
    def __init__(self, target_q=0,
                 w_ce=2.0, w_l1=5.0, w_giou=3.0, w_consistency=0.5,
                 w_traj_ade=0.5, w_traj_fde=0.5,
                 w_anchor=15.0):
        super().__init__()
        self.tq = int(target_q)
        self.w_ce, self.w_l1, self.w_giou, self.w_cons = w_ce, w_l1, w_giou, w_consistency
        self.w_traj_ade, self.w_traj_fde = float(w_traj_ade), float(w_traj_fde)
        self.w_anchor = float(w_anchor)

    def _traj_losses(self, pred_fut, gt_fut, mask=None):
        pred_c = pred_fut[..., :2]
        gt_c   = gt_fut[..., :2]
        if mask is None:
            ade = (pred_c - gt_c).abs().mean()
            fde = (pred_c[:, -1] - gt_c[:, -1]).abs().mean()
        else:
            m = mask.float().unsqueeze(-1)
            valid_sum = m.sum().clamp_min(1.0)
            ade = ((pred_c - gt_c).abs() * m).sum() / valid_sum
            last_m = mask[:, -1].float().clamp_min(1e-6)
            fde_step = (pred_c[:, -1] - gt_c[:, -1]).abs().sum(dim=-1)
            fde = (fde_step * last_m).sum() / last_m.sum().clamp_min(1.0)
        return ade, fde

    def forward(self, outputs, targets):
        logits = outputs["pred_logits"]
        boxes_for_loss = outputs.get("pred_boxes_raw", outputs["pred_boxes"])
        B,T,Q,_ = logits.shape
        dev = logits.device

        tgt_labels = torch.full((B,T,Q), 1, dtype=torch.long, device=dev)
        tgt_labels[:,:,self.tq] = 0
        loss_ce = nn.CrossEntropyLoss()(logits.reshape(-1,2), tgt_labels.reshape(-1))

        pred = boxes_for_loss[:,:,self.tq,:].reshape(B*T,4)
        gt   = torch.cat([t["boxes"].to(dev) for t in targets], dim=0)
        loss_bbox = nn.L1Loss()(pred, gt)
        giou_mat = generalized_box_iou(box_cxcywh_to_xyxy(pred), box_cxcywh_to_xyxy(gt))
        giou_diag = giou_mat.diagonal().mean()
        loss_giou = 1.0 - giou_diag

        loss_cons = pred.new_tensor(0.0)
        if self.w_cons > 0 and "target_states" in outputs:
            hs = outputs["target_states"]
            if hs.size(1) > 1:
                sim = nn.functional.cosine_similarity(hs[:,1:,:], hs[:,:-1,:], dim=-1).mean()
                loss_cons = (1 - sim) * self.w_cons

        loss_anchor = pred.new_tensor(0.0)
        K = int(outputs.get("burnin", 0) or 0)
        if K > 0 and self.w_anchor > 0:
            raw_k = boxes_for_loss[:, :min(K,T), self.tq, :]
            gt_k  = torch.stack([t["boxes"].to(dev)[:min(K,T)] for t in targets], dim=0)
            loss_anchor = self.w_anchor * nn.functional.l1_loss(raw_k, gt_k)

        loss_traj = pred.new_tensor(0.0)
        loss_traj_ade = pred.new_tensor(0.0)
        loss_traj_fde = pred.new_tensor(0.0)
        if ("future_boxes" in outputs) and all("future_boxes" in t for t in targets):
            pred_fut = outputs["future_boxes"]
            gt_fut   = torch.stack([t["future_boxes"].to(dev) for t in targets], dim=0)
            fut_mask = None
            if all(("future_mask" in t) for t in targets):
                fut_mask = torch.stack([t["future_mask"].to(dev) for t in targets], dim=0)
            ade, fde = self._traj_losses(pred_fut, gt_fut, fut_mask)
            loss_traj_ade, loss_traj_fde = ade.detach(), fde.detach()
            loss_traj = self.w_traj_ade * ade + self.w_traj_fde * fde

        total = (self.w_ce*loss_ce + self.w_l1*loss_bbox + self.w_giou*loss_giou
                 + loss_cons + loss_traj + loss_anchor)

        return {
            "loss_total": total,
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_anchor": loss_anchor,
            "loss_consistency": loss_cons,
            "loss_traj": loss_traj,
            "loss_traj_ade": loss_traj_ade,
            "loss_traj_fde": loss_traj_fde,
        }
