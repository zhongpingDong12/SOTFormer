from pathlib import Path
import sys
PROJ = str(Path(__file__).resolve().parent)
if PROJ not in sys.path: sys.path.insert(0, PROJ)


import os, math, json, ast, argparse, random, time, warnings, csv
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, DefaultDict

os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker0")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("LOCAL_DEFORMABLE_DETR_PATH", "")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from torch.amp import autocast, GradScaler
AMP_DTYPE = torch.bfloat16

from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
from collections import defaultdict

from MinimalTrackFormer_car11_Traj_multiScale_v5_patched import MinimalTrackFormer
from tracking_loss_car11_Traj_multiScale_v5 import SOTCriterion


def setup_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def compute_resize(hw: Tuple[int,int], short_side: int, max_side: int):
    """Return resized (H',W') and scale factors (s_h, s_w); only downscale."""
    H0, W0 = hw
    if short_side <= 0 and max_side <= 0:
        return (H0, W0), (1.0, 1.0)
    s1 = 1.0
    if short_side > 0:
        s1 = min(1.0, float(short_side) / max(H0, W0) if min(H0, W0)==0 else float(short_side) / min(H0, W0))
    H1, W1 = int(round(H0 * s1)), int(round(W0 * s1))
    s2 = 1.0
    if max_side > 0 and max(H1, W1) > max_side:
        s2 = float(max_side) / max(H1, W1)
    H2, W2 = int(round(H1 * s2)), int(round(W1 * s2))
    sh, sw = (H2 / max(H0, 1), W2 / max(W0, 1))
    return (H2, W2), (sh, sw)

def _is_norm_bbox_list(bboxes: List[List[float]]) -> bool:
    """Rough check: most values within [0,1.2] considered normalized."""
    vals = []
    for b in bboxes[: min(2000, len(bboxes))]:
        vals.extend(b)
    if not vals:
        return False
    vals = np.asarray(vals, dtype=np.float32)
    ratio = float(((vals >= -1e-6) & (vals <= 1.2)).mean())
    return ratio > 0.9


class LaSOTDataset(Dataset):
    """
    Supports three sources:
      A) *_compat.json: { seq_name: {"img_names":[...], "gt_rect":[[x,y,w,h]...]} }
      B) COCO style: {"images":[{id,file_name,width,height,video_id,frame_id}], "annotations":[{image_id,bbox,track_id,...}]}
      C) LaSOT folder: <data_dir>/<class>/<sequence>/{img/*.jpg, groundtruth.txt}

    Training: sample a random segment of T frames; validation: take first T frames.
    Read images, resize using short_side/max_side and scale GT accordingly (pixels).
    """
    def __init__(self, data_dir: str, json_path: Optional[str],
                 segment_len: int, is_training: bool = True,
                 short_side: int = 800, max_side: int = 800,
                 gt_fmt: str = "auto",
                 coco_group: str = "video_id",
                 augment: bool = False):
        self.data_dir = data_dir
        self.segment_len = segment_len
        self.is_training = is_training
        self.short_side = short_side
        self.max_side = max_side
        self.seq_infos: List[Dict[str, Any]] = []
        self.gt_fmt = gt_fmt
        self.coco_group = coco_group
        self.augment = augment and is_training

        self._basename2paths: DefaultDict[str, List[str]] = defaultdict(list)
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                fl = f.lower()
                if fl.endswith(".jpg") or fl.endswith(".jpeg") or fl.endswith(".png"):
                    self._basename2paths[f].append(os.path.join(root, f))
        if is_main_process():
            total_imgs = sum(len(v) for v in self._basename2paths.values())
            print(f"[COCO] basename index built from {self.data_dir}: {total_imgs} entries; duplicates=0 (kept first)")

        def _resolve_path(name: str) -> Optional[str]:
            if os.path.isabs(name) and os.path.exists(name):
                return name
            cand = os.path.join(self.data_dir, name)
            if os.path.exists(cand):
                return cand
            cand2 = os.path.join(self.data_dir, "images", name)
            if os.path.exists(cand2):
                return cand2
            hits = self._basename2paths.get(os.path.basename(name), [])
            if len(hits) == 1:
                return hits[0]
            elif len(hits) > 1:
                img_hits = [p for p in hits if "/img/" in p.replace("\\", "/")]
                if img_hits:
                    return img_hits[0]
                if is_main_process():
                    print(f"[WARN] Multiple matches for {name}, pick {hits[0]}")
                return hits[0]
            return None

        if json_path and os.path.exists(json_path):
            with open(json_path, "r") as f:
                meta = json.load(f)
            if isinstance(meta, str):
                meta = ast.literal_eval(meta)

            if isinstance(meta, dict) and ("images" in meta) and ("annotations" in meta):
                imgs = {im["id"]: im for im in meta["images"]}
                ann_by_img: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
                for ann in meta["annotations"]:
                    ann_by_img[ann["image_id"]].append(ann)

                if self.gt_fmt == "auto":
                    all_bbox = [ann.get("bbox", [0,0,0,0]) for ann in meta["annotations"][:5000]]
                    if _is_norm_bbox_list(all_bbox):
                        self.gt_fmt = "xywh_norm"
                    else:
                        self.gt_fmt = "xywh_px"
                    if is_main_process():
                        print(f"[COCO] auto-detected gt_fmt = {self.gt_fmt}")

                def _group_key(im: Dict[str, Any]) -> str:
                    if self.coco_group == "dirname":
                        fn = im.get("file_name", "")
                        norm = fn.replace("\\", "/")
                        d = "/".join(norm.split("/")[:-1]).strip("/")
                        return d if d else "root"
                    else:
                        return f"vid{im.get('video_id', -1)}"

                groups: DefaultDict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
                for iid, im in imgs.items():
                    groups[_group_key(im)].append((iid, im))

                for gkey, im_list in groups.items():
                    im_list.sort(key=lambda x: (imgs[x[0]].get("frame_id", x[0])))
                    frames, gts = [], []
                    for iid, im in im_list:
                        fn = im.get("file_name")
                        full = _resolve_path(fn) if fn else None
                        if not full:
                            continue
                        frames.append(full)
                        anns = ann_by_img.get(iid, [])
                        pick = None
                        if len(anns) == 1:
                            pick = anns[0]
                        elif len(anns) > 1:
                            pick = min(anns, key=lambda a: a.get("track_id", 1<<30))
                        if pick is None:
                            gts.append([0,0,1,1])
                        else:
                            gts.append(pick.get("bbox", [0,0,1,1]))
                    if len(frames) > 0:
                        self.seq_infos.append({"name": gkey, "frames": frames, "gt": gts})
            elif isinstance(meta, dict):
                items = list(meta.items())
                for seq_name, info in items:
                    if isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict):
                        info = info[0]
                    if not isinstance(info, dict):
                        continue
                    frames = info.get("img_names") or info.get("frames") or info.get("images")
                    gts = info.get("gt_rect") or info.get("gt") or info.get("ann") or info.get("bboxes")
                    if not frames or not gts:
                        continue
                    resolved = []
                    for p in frames:
                        full = _resolve_path(p)
                        if full:
                            resolved.append(full)
                    if len(resolved) != len(frames):
                        continue
                    self.seq_infos.append({"name": str(seq_name), "frames": resolved, "gt": gts})
            else:
                pass
        else:
            for cat in sorted(os.listdir(self.data_dir)):
                cat_dir = os.path.join(self.data_dir, cat)
                if not os.path.isdir(cat_dir):
                    continue
                for seq in sorted(os.listdir(cat_dir)):
                    seq_dir = os.path.join(cat_dir, seq)
                    img_dir = os.path.join(seq_dir, "img")
                    gt_path = os.path.join(seq_dir, "groundtruth.txt")
                    if not (os.path.isdir(img_dir) and os.path.isfile(gt_path)):
                        continue
                    frames = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)
                                     if f.lower().endswith((".jpg", ".png"))])
                    with open(gt_path, "r") as f:
                        gt = [list(map(float, line.strip().split(","))) for line in f]
                    if len(frames) != len(gt) or len(frames) == 0:
                        continue
                    self.seq_infos.append({"name": f"{cat}/{seq}", "frames": frames, "gt": gt})

        if len(self.seq_infos) == 0 and is_main_process():
            print("[WARN] LaSOTDataset: no valid sequences loaded.")

    def __len__(self):
        return len(self.seq_infos)

    def __getitem__(self, idx: int):
        info = self.seq_infos[idx]
        frames: List[str] = info["frames"]
        gt_raw: List[List[float]] = info["gt"]
        total = len(frames)

        if self.is_training:
            if total < self.segment_len:
                start = 0
                T = self.segment_len
                frames = frames + [frames[-1]] * (T - total)
                gt_raw = gt_raw + [gt_raw[-1]] * (T - total)
            else:
                T = self.segment_len
                start = random.randint(0, max(0, total - T))
                frames = frames[start:start + T]
                gt_raw = gt_raw[start:start + T]
        else:
            if total < self.segment_len:
                start = 0
                T = self.segment_len
                frames = frames + [frames[-1]] * (T - total)
                gt_raw = gt_raw + [gt_raw[-1]] * (T - total)
            else:
                start = 0
                T = self.segment_len
                frames = frames[start:start + T]
                gt_raw = gt_raw[start:start + T]

        do_flip = self.augment and random.random() < 0.5

        imgs_t = []
        boxes_scaled_cxcywh_pix = []

        for i, p in enumerate(frames):
            im = Image.open(p).convert("RGB")
            W0, H0 = im.size
            (Hn, Wn), (sh, sw) = compute_resize((H0, W0), self.short_side, self.max_side)

            if (Hn, Wn) != (H0, W0):
                im = F.resize(im, size=[Hn, Wn], antialias=True)

            if do_flip:
                im = F.hflip(im)

            t = F.to_tensor(im)
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            t = (t - mean) / std
            imgs_t.append(t)

            x, y, w, h = gt_raw[i]
            if self.gt_fmt == "xywh_px":
                x_n = x * sw; y_n = y * sh; w_n = w * sw; h_n = h * sh
            elif self.gt_fmt == "xywh_norm":
                x_n = x * (W0 * sw); y_n = y * (H0 * sh)
                w_n = w * (W0 * sw); h_n = h * (H0 * sh)
            elif self.gt_fmt == "cxcywh_norm":
                cx = x * (W0 * sw); cy = y * (H0 * sh)
                w_n = w * (W0 * sw); h_n = h * (H0 * sh)
                if do_flip:
                    cx = Wn - cx
                boxes_scaled_cxcywh_pix.append([cx, cy, w_n, h_n]); continue
            else:
                x_n = x * (W0 * sw); y_n = y * (H0 * sh)
                w_n = w * (W0 * sw); h_n = h * (H0 * sh)

            cx  = x_n + w_n / 2.0
            cy  = y_n + h_n / 2.0

            if do_flip:
                cx = Wn - cx

            boxes_scaled_cxcywh_pix.append([cx, cy, w_n, h_n])

        imgs = torch.stack(imgs_t, dim=0)
        boxes_pix = torch.tensor(boxes_scaled_cxcywh_pix, dtype=torch.float32)

        target = {"boxes": boxes_pix}
        return imgs, target


def collate_batch(batch):
    imgs_list   = [b[0] for b in batch]
    boxes_px_l  = [b[1]["boxes"] for b in batch]

    sizes_each = [[(x[t].shape[-2], x[t].shape[-1]) for t in range(x.shape[0])]
                  for x in imgs_list]
    max_h = max(max(h for h, _ in sizes) for sizes in sizes_each)
    max_w = max(max(w for _, w in sizes) for sizes in sizes_each)

    padded_imgs = []
    for x in imgs_list:
        T, C, H, W = x.shape
        pad_h, pad_w = max_h - H, max_w - W
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        padded_imgs.append(x)
    imgs = torch.stack(padded_imgs, dim=0)

    def cxcywh_px_to_norm_per_frame(cxcywh_px: torch.Tensor, hw_list: List[tuple]):
        assert cxcywh_px.shape[0] == len(hw_list)
        outs = []
        for i in range(cxcywh_px.shape[0]):
            Ht, Wt = hw_list[i]
            cx, cy, w, h = cxcywh_px[i].unbind(-1)
            Wt = max(int(Wt), 1); Ht = max(int(Ht), 1)
            outs.append(torch.stack([
                (cx / Wt).clamp_(0, 1),
                (cy / Ht).clamp_(0, 1),
                (w  / Wt).clamp_(1e-6, 1.0),
                (h  / Ht).clamp_(1e-6, 1.0),
            ], dim=-1))
        return torch.stack(outs, dim=0)

    boxes_list = []
    for boxes_px, hw_list in zip(boxes_px_l, sizes_each):
        boxes_norm = cxcywh_px_to_norm_per_frame(boxes_px, hw_list)
        boxes_list.append(boxes_norm)

    boxes = torch.stack(boxes_list, dim=0)
    targets = {"boxes": boxes}
    return imgs, targets


def collate_val(batch):
    assert len(batch) == 1, "dl_val requires batch_size=1"
    imgs, target = batch[0]
    T, C, H, W = imgs.shape
    imgs = imgs.unsqueeze(0)
    target = {"boxes": target["boxes"].unsqueeze(0), "orig_hw": torch.tensor([[H, W]], dtype=torch.float32)}
    return imgs, target


def safe_model_forward(model, imgs, boxes=None, **kwargs):
    try:
        return model(imgs, gt_boxes_first_k=boxes, **kwargs)
    except TypeError:
        try:
            return model(imgs, boxes, **kwargs)
        except TypeError:
            warnings.warn("[Compat] model.forward only accepts imgs; not passing boxes/kwargs.", RuntimeWarning)
            return model(imgs)


@torch.no_grad()
def evaluate(model, criterion, dl_val, device, distributed=False, kb_cap: int = 3):
    was_training = model.training
    model.eval()
    total, cnt = 0.0, 0
    do_eval = (not distributed) or is_main_process()

    if do_eval:
        for imgs, batched in dl_val:
            imgs = imgs.to(device, non_blocking=True)
            boxes_px = batched["boxes"].to(device, non_blocking=True)
            H, W = batched["orig_hw"][0].tolist()
            scale = torch.tensor([W, H, W, H], device=device).view(1, 1, 4)
            boxes = boxes_px / scale

            T = imgs.size(1)
            kb = 3
            boxes_firstk = boxes[:, :kb, :]

            with torch.inference_mode(), autocast(device_type='cuda', dtype=AMP_DTYPE):
                outputs = safe_model_forward(
                    model, imgs, boxes_firstk,
                    k_burnin=kb, priming_prob=1.0,
                    force_prime=True
                )
                loss_out = criterion(outputs, [{"boxes": boxes[0]}])
                loss = loss_out["loss_total"] if isinstance(loss_out, dict) else loss_out

            if torch.isfinite(loss):
                total += float(loss.item())
                cnt += 1

    if distributed:
        dist.barrier()

    mean_val = (total / max(1, cnt)) if do_eval else None
    if was_training:
        model.train()
    return mean_val


def extract_pred_boxes(outputs):
    """
    Extract (B,T,4) normalized boxes from outputs:
    - prefer pred_boxes[pick_best_query]
    - use non-background highest score as foreground
    """
    if not isinstance(outputs, dict):
        return None

    pb = outputs.get("pred_boxes", None)
    if pb is None:
        pb = outputs.get("boxes", None)
    if pb is None or (not torch.is_tensor(pb)):
        return None

    if pb.dim() == 3 and pb.size(-1) == 4:
        return pb.clamp(0, 1)

    if pb.dim() == 4 and pb.size(-1) == 4:
        pl = outputs.get("pred_logits", None)
        if torch.is_tensor(pl) and pl.dim() == 4 and list(pl.shape[:3]) == list(pb.shape[:3]):
            probs = torch.softmax(pl, dim=-1)
            C = probs.size(-1)
            if C >= 2:
                fg_scores = probs[..., :C-1].max(dim=-1).values
            else:
                fg_scores = probs.squeeze(-1)
            idx = fg_scores.argmax(dim=-1, keepdim=True)
            idx4 = idx.unsqueeze(-1).expand(-1, -1, -1, 4)
            best = pb.gather(dim=2, index=idx4).squeeze(2)
            return best.clamp(0, 1)

        return pb[..., 0, :].clamp(0, 1)

    return None


@torch.no_grad()
def compute_val_metrics(model, dl_val, device, diag_norm=0.005, p20=20.0,
                        distributed=False, csv_path=None, epoch=None, kb_cap: int = 3,
                        viz_every: int = 0, viz_prime_k: int = 3):
    was_training = model.training
    model.eval()
    do_eval = (not distributed) or is_main_process()

    auc_list, p20_list, nprec_list = [], [], []

    if do_eval:
        for i, (imgs, batched) in enumerate(dl_val):
            imgs = imgs.to(device, non_blocking=True)
            gt_px = batched["boxes"].to(device, non_blocking=True)
            H, W = batched["orig_hw"][0].tolist()
            scale = torch.tensor([W, H, W, H], device=device).view(1, 1, 4)
            gt_norm = gt_px / scale

            T = imgs.size(1)
            kb = 3
            boxes_firstk = gt_norm[:, :kb, :]

            with torch.inference_mode(), autocast(device_type='cuda', dtype=AMP_DTYPE):
                outputs = safe_model_forward(
                    model, imgs, boxes_firstk, k_burnin=kb, priming_prob=1.0, force_prime=True
                )

            pb = None
            if isinstance(outputs, dict):
                pb = outputs.get("pred_boxes", outputs.get("boxes", None))
            picked_norm = None

            if torch.is_tensor(pb) and pb.dim() == 4 and pb.size(-1) == 4:
                b = 0
                t_ref = kb - 1
                def cxcywh_to_xyxy_t(x):
                    cx, cy, w, h = x.unbind(-1)
                    x1 = cx - 0.5 * w; y1 = cy - 0.5 * h
                    x2 = x1 + w;     y2 = y1 + h
                    return torch.stack([x1, y1, x2, y2], dim=-1)

                gt_ref_xyxy = cxcywh_to_xyxy_t(gt_norm[b, t_ref])
                pred_xyxy_allq_ref = cxcywh_to_xyxy_t(pb[b, t_ref])

                def iou_xyxy_t(a, b):
                    ax1, ay1, ax2, ay2 = a.unbind(-1)
                    bx1, by1, bx2, by2 = b.unbind(-1)
                    ix1 = torch.maximum(ax1, bx1); iy1 = torch.maximum(ay1, by1)
                    ix2 = torch.minimum(ax2, bx2); iy2 = torch.minimum(ay2, by2)
                    iw = torch.clamp(ix2 - ix1, min=0); ih = torch.clamp(iy2 - iy1, min=0)
                    inter = iw * ih
                    area_a = torch.clamp(ax2 - ax1, min=0) * torch.clamp(ay2 - ay1, min=0)
                    area_b = torch.clamp(bx2 - bx1, min=0) * torch.clamp(by2 - by1, min=0)
                    union = area_a + area_b - inter
                    return torch.where(union > 0, inter / union, torch.zeros_like(union))

                q_iou = iou_xyxy_t(pred_xyxy_allq_ref, gt_ref_xyxy.expand_as(pred_xyxy_allq_ref))
                q_best = q_iou.argmax().item()
                picked_norm = pb[b, :, q_best, :].unsqueeze(0).clamp(0, 1)

            else:
                picked_norm = extract_pred_boxes(outputs)
                if picked_norm is None:
                    warnings.warn("[Metrics] cannot extract pred_boxes from outputs, skipping batch.")
                    continue
                picked_norm = picked_norm.clamp(0, 1)

            pred_px = picked_norm * scale

            def cxcywh_to_xywh(x):
                cx, cy, w, h = x.unbind(-1)
                x1 = cx - 0.5 * w
                y1 = cy - 0.5 * h
                return torch.stack([x1, y1, w, h], dim=-1)

            pred_xywh = cxcywh_to_xywh(pred_px[0]).cpu().numpy()
            gt_xywh   = cxcywh_to_xywh(gt_px[0]).cpu().numpy()

            def iou_xywh(a, b):
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                ax2, ay2 = ax+aw, ay+ah
                bx2, by2 = bx+bw, by+bh
                ix1, iy1 = max(ax, bx), max(ay, by)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
                inter = iw*ih
                ua = aw*ah + bw*bh - inter
                return inter/ua if ua>0 else 0.0

            Tn = min(len(pred_xywh), len(gt_xywh))
            if Tn == 0:
                continue

            ious = np.array([iou_xywh(pred_xywh[j], gt_xywh[j]) for j in range(Tn)], dtype=np.float32)
            thr = np.linspace(0, 1, 101, dtype=np.float32)
            succ = np.array([(ious >= t).mean() for t in thr], dtype=np.float32)
            auc = np.trapezoid(succ, thr)

            def center_err(a, b):
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                acx, acy = ax+aw/2.0, ay+ah/2.0
                bcx, bcy = bx+bw/2.0, by+bh/2.0
                return math.hypot(acx-bcx, acy-bcy)

            errs = np.array([center_err(pred_xywh[j], gt_xywh[j]) for j in range(Tn)], dtype=np.float32)
            prec20 = float((errs <= p20).mean())

            bbox_diag = np.array([math.hypot(gt_xywh[j][2], gt_xywh[j][3]) for j in range(Tn)], dtype=np.float32)
            thr_norm = diag_norm * bbox_diag
            nprec = float((errs <= thr_norm).mean())

            auc_list.append(auc)
            p20_list.append(prec20)
            nprec_list.append(nprec)

            if viz_every > 0 and (i % viz_every == 0):
                try:
                    import cv2
                    fr0 = imgs[0, 0].detach().cpu()
                    fr0 = (fr0 * torch.tensor([0.229,0.224,0.225]).view(3,1,1) + torch.tensor([0.485,0.456,0.406]).view(3,1,1)).clamp(0,1)
                    fr0 = (fr0.permute(1,2,0).numpy() * 255).astype(np.uint8)[:, :, ::-1]
                    px = pred_xywh[0]; gx = gt_xywh[0]
                    def draw(x, color, img):
                        x1,y1,w,h = [int(v) for v in x]
                        x2,y2 = x1+w, y1+h
                        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                    draw(px, (0,255,0), fr0)
                    draw(gx, (0,0,255), fr0)
                    out = os.path.join("viz_samples", f"val_{epoch or 0:03d}_{i:05d}.jpg")
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    cv2.imwrite(out, fr0)
                except Exception as e:
                    if is_main_process():
                        print("[VIZ] failed:", e)

    if distributed:
        dist.barrier()
    if was_training:
        model.train()

    if (not do_eval) or len(auc_list) == 0:
        return None, None, None

    mean_auc   = float(sum(auc_list) / len(auc_list)) * 100.0
    mean_p20   = float(sum(p20_list) / len(p20_list)) * 100.0
    mean_nprec = float(sum(nprec_list) / len(nprec_list)) * 100.0

    if csv_path is not None and epoch is not None and is_main_process():
        need_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            cw = csv.writer(f)
            if need_header:
                cw.writerow(["epoch", "SuccessAUC(%)", "Precision@20px(%)", "NormPrec(%)"])
            cw.writerow([epoch, f"{mean_auc:.2f}", f"{mean_p20:.2f}", f"{mean_nprec:.2f}"])

    return mean_auc, mean_p20, mean_nprec


@torch.no_grad()
def _diag_best_of_q_upper_bound(model, dl_val, device, kb_cap: int = 3, max_batches: int = 20):
    was_training = model.training
    model.eval()

    auc_list = []
    cnt = 0

    for i, (imgs, batched) in enumerate(dl_val):
        if i >= max_batches:
            break

        imgs = imgs.to(device, non_blocking=True)
        gt_px = batched["boxes"].to(device, non_blocking=True)
        H, W = batched["orig_hw"][0].tolist()
        scale = torch.tensor([W, H, W, H], device=device).view(1, 1, 4)
        gt_norm = gt_px / scale

        T = imgs.size(1)
        kb = 3
        boxes_firstk = gt_norm[:, :kb, :]

        with torch.inference_mode(), autocast(device_type='cuda', dtype=AMP_DTYPE):
            outputs = safe_model_forward(
                model, imgs, boxes_firstk, k_burnin=kb, priming_prob=1.0, force_prime=True
            )

        pb = None
        if isinstance(outputs, dict):
            pb = outputs.get("pred_boxes", outputs.get("boxes", None))
        if not (torch.is_tensor(pb) and pb.dim() == 4 and pb.size(-1) == 4):
            continue

        pred_px_allq = pb.clamp(0, 1) * scale
        gt_px_b = gt_px

        def cxcywh_to_xyxy(x):
            cx, cy, w, h = x.unbind(-1)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = x1 + w
            y2 = y1 + h
            return torch.stack([x1, y1, x2, y2], dim=-1)

        pred_xyxy_allq = cxcywh_to_xyxy(pred_px_allq[0])
        gt_xyxy = cxcywh_to_xyxy(gt_px_b[0])

        def iou_xyxy_t(a, b):
            ax1, ay1, ax2, ay2 = a.unbind(-1)
            bx1, by1, bx2, by2 = b.unbind(-1)
            ix1 = torch.maximum(ax1, bx1)
            iy1 = torch.maximum(ay1, by1)
            ix2 = torch.minimum(ax2, bx2)
            iy2 = torch.minimum(ay2, by2)
            iw = torch.clamp(ix2 - ix1, min=0)
            ih = torch.clamp(iy2 - iy1, min=0)
            inter = iw * ih
            area_a = torch.clamp(ax2 - ax1, min=0) * torch.clamp(ay2 - ay1, min=0)
            area_b = torch.clamp(bx2 - bx1, min=0) * torch.clamp(by2 - by1, min=0)
            union = area_a + area_b - inter
            return torch.where(union > 0, inter / union, torch.zeros_like(union))

        Tn, Q = pred_xyxy_allq.shape[:2]
        gt_rep = gt_xyxy[:Tn].unsqueeze(1).expand(Tn, Q, 4)
        ious_allq = iou_xyxy_t(pred_xyxy_allq[:Tn], gt_rep)
        best_iou = ious_allq.max(dim=1).values

        thr = torch.linspace(0, 1, 101, device=best_iou.device)
        succ = torch.stack([(best_iou >= t).float().mean() for t in thr], dim=0)
        auc = torch.trapezoid(succ, thr).item()
        auc_list.append(auc)
        cnt += 1

    if was_training:
        model.train()

    if cnt == 0:
        return None
    return float(sum(auc_list) / cnt) * 100.0


@dataclass
class VariantCfg:
    name: str
    use_future: bool
    k_burnin: int

def build_variant_cfg(name: str) -> Dict[str, Any]:
    if name == "frame_only":
        return {"name": "frame_only", "use_future": False, "k_burnin": 0}
    elif name == "temporal":
        return {"name": "temporal", "use_future": True, "k_burnin": 0}
    elif name == "temporal_primed":
        return {"name": "temporal", "use_future": True, "k_burnin": 3}
    elif name == "full":
        return {"name": "full", "use_future": True, "k_burnin": 3}
    else:
        return {"name": name, "use_future": True, "k_burnin": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/apps/data/share/LaSOT14_Dataset/images')
    parser.add_argument('--train_json', default='/apps/data/share/LaSOT14_Dataset/annotations/train/train.json')
    parser.add_argument('--val_json', default='/apps/data/share/LaSOT14_Dataset/annotations/val/val.json')
    parser.add_argument('--output_dir', default='../ckpts/run_full_coco')

    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--segment_len', type=int, default=12)
    parser.add_argument('--tail_len', type=int, default=10, help="tail length (OBS_TAIL)")
    parser.add_argument('--lr_heads', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--k_burnin', type=int, default=3)
    parser.add_argument('--prime_p_start', type=float, default=1.0)
    parser.add_argument('--prime_p_end', type=float, default=0.3)
    parser.add_argument('--prime_decay_epochs', type=int, default=40)

    parser.add_argument('--short_side', type=int, default=640)
    parser.add_argument('--max_side', type=int, default=960)
    parser.add_argument('--num_queries', type=int, default=300, help="recommend 300 to match pretraining")

    parser.add_argument('--gt_fmt', type=str, default='auto',
                        choices=['auto','xywh_px','xywh_norm','cxcywh_norm'])

    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--plateau_patience', type=int, default=3)
    parser.add_argument('--plateau_factor', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=1e-7)
    parser.add_argument('--clip_grad_norm', type=float, default=0.3)
    parser.add_argument('--find_unused', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--diag_norm', type=float, default=0.005)
    parser.add_argument('--p20', type=float, default=20.0)
    parser.add_argument('--variant', type=str, default='temporal_primed',
                        choices=['frame_only','temporal','temporal_primed','full'])

    parser.add_argument('--viz_every', type=int, default=0)
    parser.add_argument('--viz_prime_k', type=int, default=3)

    parser.add_argument('--coco_group', type=str, default='video_id',
                        choices=['video_id', 'dirname'],
                        help="COCO grouping key: video_id (default) or dirname of file_name")

    parser.add_argument('--diag_best_of_q', action='store_true',
                        help='Run a quick diagnostic upper bound by picking the best query per frame on val set.')
    parser.add_argument('--diag_max_batches', type=int, default=20,
                        help='Max number of val batches to use for the diagnostic.')

    parser.add_argument('--augment', action='store_true', help='Enable data augmentation during training')

    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world > 1

    if distributed:
        from datetime import timedelta
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world,
            timeout=timedelta(minutes=30)
        )
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    ds = LaSOTDataset(args.data_dir, args.train_json, args.segment_len,
                      is_training=True, short_side=args.short_side, max_side=args.max_side,
                      gt_fmt=args.gt_fmt, coco_group=args.coco_group,
                      augment=args.augment)
    if is_main_process():
        print(f"[Sanity] train seqs={len(ds)}, steps/epoch={len(ds)}")
    if len(ds) == 0:
        if is_main_process():
            print("[FATAL] Train dataset is empty. Check your JSON or data_dir.")
        if distributed:
            dist.destroy_process_group()
        return
    per_gpu_bs = max(1, args.batch_size // max(1, world))
    sampler = DistributedSampler(ds, shuffle=True, drop_last=True) if distributed else None
    dl = DataLoader(
        ds, batch_size=per_gpu_bs, sampler=sampler, shuffle=(sampler is None),
        num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_batch
    )

    dl_val = None
    if args.val_json and os.path.exists(args.val_json):
        ds_val = LaSOTDataset(args.data_dir, args.val_json, args.segment_len,
                              is_training=False, short_side=args.short_side, max_side=args.max_side,
                              gt_fmt=args.gt_fmt, coco_group=args.coco_group)
        if is_main_process():
            print(f"[Sanity] val   seqs={len(ds_val)}")
        dl_val = DataLoader(
            ds_val, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=True, drop_last=False, collate_fn=collate_val
        )

    model = MinimalTrackFormer(num_queries=args.num_queries, hidden_dim=256, horizon=max(1, args.tail_len//2)).to(device)
    
    criterion = SOTCriterion(
        w_ce=2.0,
        w_l1=5.0,
        w_giou=3.0,
        w_consistency=0.5,
        w_traj_ade=0.5,
        w_traj_fde=0.5,
        w_anchor=15.0
    )

    var = build_variant_cfg(args.variant)

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=args.find_unused, broadcast_buffers=False
        )

    base_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    backbone_params, head_params = [], []
    for n, p in base_model.named_parameters():
        if not p.requires_grad: 
            continue
        (backbone_params if n.startswith("detector.") else head_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": args.lr_heads},
         {"params": backbone_params, "lr": args.lr_backbone}],
        weight_decay=args.weight_decay
    )
    
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    warmup_epochs = 3
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=len(dl) * warmup_epochs
    )
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.plateau_factor, 
        patience=args.plateau_patience, cooldown=0, min_lr=args.min_lr
    )
    
    scaler = GradScaler(enabled=True, growth_interval=2000, backoff_factor=0.25)

    if args.diag_best_of_q and dl_val is not None:
        upper = _diag_best_of_q_upper_bound(
            model if not isinstance(model, nn.parallel.DistributedDataParallel) else model.module,
            dl_val, device, kb_cap=3, max_batches=args.diag_max_batches
        )
        if is_main_process():
            if upper is not None:
                print(f"[Diag] Best-of-Q AUC upper bound ~ {upper:.2f}%")
            else:
                print("[Diag] Best-of-Q skipped (no multi-query outputs)")

    best_val_loss = float("inf")
    best_auc = -1.0
    best_auc_path = os.path.join(args.output_dir, "checkpoint_best_auc.pth")
    best_loss_path = os.path.join(args.output_dir, "checkpoint_best_loss.pth")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        if distributed:
            sampler.set_epoch(epoch)
        base_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        model.train()
        running = 0.0
        count = 0
        t0 = time.time()

        if epoch <= args.prime_decay_epochs:
            priming_prob = args.prime_p_start - (args.prime_p_start - args.prime_p_end) * ((epoch-1) / max(1, args.prime_decay_epochs))
        else:
            priming_prob = args.prime_p_end
        k_burnin = 3

        for step, (imgs, batched) in enumerate(dl):
            imgs = imgs.to(device, non_blocking=True)
            boxes = batched["boxes"].to(device, non_blocking=True)
            boxes[..., 2:4] = boxes[..., 2:4].clamp_(min=1e-6, max=1.0)

            T = imgs.size(1)
            
            kb = 3
            boxes_firstk = boxes[:, :kb, :]

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=AMP_DTYPE):
                outputs = safe_model_forward(
                    model, imgs, boxes_firstk,
                    k_burnin=kb, priming_prob=float(priming_prob)
                )
                targets_list = [{"boxes": boxes[b]} for b in range(imgs.size(0))]
                loss_dict = criterion(outputs, targets_list)
                loss = loss_dict["loss_total"] if isinstance(loss_dict, dict) else loss_dict

            if not torch.isfinite(loss):
                if is_main_process():
                    print(f"[WARN] non-finite loss at epoch {epoch} step {step}, skip.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if args.clip_grad_norm and args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach().item())
            count += 1

            if epoch <= warmup_epochs:
                warmup_scheduler.step()

            if is_main_process() and (step % 10 == 0):
                print(f"Epoch {epoch}/{args.epochs} Step {step} Loss {running / max(1, count):.4f}")

        epoch_train_loss = running / max(1, count)

        val_metric = None
        auc_mean = p20_mean = nprec_mean = None
        if dl_val is not None:
            base_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
            val_metric = evaluate(base_model, criterion, dl_val, device, distributed=distributed, kb_cap=3)
            auc_mean, p20_mean, nprec_mean = compute_val_metrics(
                base_model, dl_val, device,
                diag_norm=args.diag_norm, p20=args.p20,
                distributed=distributed,
                csv_path=os.path.join(args.output_dir, "val_metrics_log.csv"),
                epoch=epoch, kb_cap=3,
                viz_every=args.viz_every, viz_prime_k=args.viz_prime_k
            )

        if is_main_process():
            ck = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pth")
            sd = model.module.state_dict() if distributed else model.state_dict()
            torch.save(sd, ck)
            took = time.time() - t0
            if val_metric is not None:
                print(f"Saved: {ck}  | train_loss={epoch_train_loss:.4f} val_loss={val_metric:.4f} | took {took:.1f}s")
                if auc_mean is not None:
                    print(f"[VAL Metrics] AUC={auc_mean:.2f}%  P@20px={p20_mean:.2f}%  NPrec={nprec_mean:.2f}%")
            else:
                print(f"Saved: {ck}  | epoch_loss={epoch_train_loss:.4f} | took {took:.1f}s")

        drive_loss = val_metric if (val_metric is not None) else epoch_train_loss
        if epoch > warmup_epochs:
            main_scheduler.step(drive_loss)

        if is_main_process():
            cur_lrs = [pg['lr'] for pg in optimizer.param_groups]
            print(f"[LR] {cur_lrs}")

        improved_loss = (drive_loss < (best_val_loss - 1e-6))
        if improved_loss and is_main_process():
            best_val_loss = drive_loss
            sd = model.module.state_dict() if distributed else model.state_dict()
            torch.save(sd, os.path.join(args.output_dir, "checkpoint_best_loss.pth"))
            print(f"[Best-Loss] val_loss={best_val_loss:.4f} -> {os.path.join(args.output_dir, 'checkpoint_best_loss.pth')}")

        improved_auc = (auc_mean is not None) and (auc_mean > best_auc + 1e-6)
        if improved_auc and is_main_process():
            best_auc = auc_mean
            sd = model.module.state_dict() if distributed else model.state_dict()
            torch.save(sd, os.path.join(args.output_dir, "checkpoint_best_auc.pth"))
            print(f"[Best-AUC] AUC={best_auc:.2f}% -> {os.path.join(args.output_dir, 'checkpoint_best_auc.pth')}")
            epochs_no_improve = 0
        else:
            if is_main_process():
                epochs_no_improve += 1
                print(f"[EarlyStop] (AUC) no improvement for {epochs_no_improve}/{args.early_stop_patience} epochs")

        if epochs_no_improve >= args.early_stop_patience:
            if is_main_process():
                print("[EarlyStop] AUC patience reached. Stopping training.")
            break

    if is_main_process():
        final = os.path.join(args.output_dir, "checkpoint_final.pth")
        sd = model.module.state_dict() if distributed else model.state_dict()
        torch.save(sd, final)
        print("Saved:", final)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
