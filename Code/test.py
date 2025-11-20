import os, json, ast, argparse
from pathlib import Path
from PIL import Image

import torch
import torch.distributed as dist
import torchvision.transforms.functional as F

from models.MinimalTrackFormer_car11_Traj_multiScale_v5_patched import MinimalTrackFormer

def load_sequences_from_json_or_scan(data_dir, json_path=None):
    seqs = []
    if json_path and os.path.exists(json_path):
        with open(json_path,'r') as f:
            meta = json.load(f)
        if isinstance(meta, str):
            meta = ast.literal_eval(meta)
        if isinstance(meta, dict):
            items = meta.items()
        else:
            items = []
        for seq_name, info in items:
            if isinstance(info, list) and len(info)>0 and isinstance(info[0], dict):
                info = info[0]
            frames = info.get("img_names") or info.get("frames") or info.get("images")
            gt     = info.get("gt_rect")  or info.get("gt")
            if frames and not os.path.isabs(frames[0]):
                frames = [os.path.join(data_dir, p) for p in frames]
            seqs.append({"name": seq_name, "frames": frames, "gt": gt})
    else:
        for cat in os.listdir(data_dir):
            cp = os.path.join(data_dir, cat)
            if not os.path.isdir(cp): continue
            for seq in os.listdir(cp):
                sp = os.path.join(cp, seq)
                img_dir = os.path.join(sp, "img")
                gt_path = os.path.join(sp, "groundtruth.txt")
                if not os.path.isdir(img_dir): continue
                frames = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                 if f.lower().endswith(('.jpg','.png'))])
                gt = None
                if os.path.isfile(gt_path):
                    with open(gt_path,'r') as f:
                        gt = [list(map(float, l.strip().split(','))) for l in f if l.strip()]
                seqs.append({"name": seq, "frames": frames, "gt": gt})
    return seqs

def to_tensor_norm(img_p):
    im = Image.open(img_p).convert("RGB")
    t = F.to_tensor(im)
    mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:,None,None]
    return (t - mean)/std, im.size  # (C,H,W), (W,H)

def xywh_to_cxcywh_norm(box_xywh, img_wh):
    x, y, w, h = box_xywh
    W, H = img_wh
    cx = (x + w/2.0) / max(W, 1)
    cy = (y + h/2.0) / max(H, 1)
    wn = w / max(W, 1)
    hn = h / max(H, 1)
    return [cx, cy, wn, hn]

def extract_xywh_per_frame(outputs, T, img_wh=None):
    cand = None
    for k in ["pred_boxes", "boxes", "outputs_boxes"]:
        if k in outputs: cand = outputs[k]; break
    if cand is None:
        for v in outputs.values():
            if torch.is_tensor(v) and v.dim()>=2 and v.shape[-1] in (4,6):
                cand = v; break
    if cand is None: raise RuntimeError("Cannot find predicted boxes in outputs")

    b = cand
    if b.dim()==2 and b.shape[-1]>=4:
        boxes = b[..., :4].detach().cpu()
    elif b.dim()==3 and b.shape[-1]>=4:
        boxes = b[0, :T, :4].detach().cpu() if b.shape[0]!=T else b[..., :4].detach().cpu()
    elif b.dim()==4 and b.shape[-1]>=4:
        # [B,T,Q,4] : choose by scores if provided
        Q = b.shape[-2]
        if "scores" in outputs and torch.is_tensor(outputs["scores"]):
            sc = outputs["scores"][0, :T, :Q]
            idx = torch.argmax(sc, dim=-1)
        else:
            idx = torch.zeros(T, dtype=torch.long)
        sel = b[0, :T, ...]
        boxes = sel[torch.arange(sel.shape[0]), idx, :4].detach().cpu()
    else:
        raise RuntimeError(f"Unexpected box shape: {list(b.shape)}")

    if img_wh is not None:
        W, H = img_wh
        vmax = float(boxes.max().item()) if boxes.numel()>0 else 0.0
        if 0.0 <= vmax <= 2.0:  # looks normalized cxcywh
            cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
            x = (cx - w/2.0) * W
            y = (cy - h/2.0) * H
            boxes = torch.stack([x, y, w*W, h*H], dim=-1)
    return boxes

def save_xywh_txt(path, boxes):
    with open(path, "w") as f:
        for b in boxes:
            x,y,w,h = [float(v) for v in b]
            f.write(f"{x:.2f},{y:.2f},{w:.2f},{h:.2f}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='/apps/data/share/LaSOT14_Dataset/images')
    ap.add_argument('--test_json', default='/apps/data/share/LaSOT14_Dataset/annotations/test/test.json')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output_dir', default='./results/full')
    ap.add_argument('--variant', type=str, default='temporal', choices=['temporal','frame_only'])
    ap.add_argument('--segment_len', type=int, default=30)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world > 1
    if distributed:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world)
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    model = MinimalTrackFormer(num_queries=100, hidden_dim=256, horizon=0).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    seqs = load_sequences_from_json_or_scan(args.data_dir, args.test_json)
    seqs = seqs[rank::world] if distributed else seqs
    print(f"[rank {rank}] sequences: {len(seqs)}")

    with torch.no_grad():
        for s in seqs:
            name, frames, gts = s["name"], s["frames"], s.get("gt", None)
            if not frames: continue

            all_boxes = []
            for start in range(0, len(frames), args.segment_len):
                frs = frames[start:start+args.segment_len]
                imgs = []
                size_wh = None
                for p in frs:
                    t, (W,H) = to_tensor_norm(p)
                    size_wh = (W,H)
                    imgs.append(t)
                imgs = torch.stack(imgs, dim=0).unsqueeze(0).to(device)  # (1,T,3,H,W)

                if start == 0 and gts and len(gts) > 0:
                    # gts[0] : [x,y,w,h] (pixels)
                    gt0_norm = torch.tensor([xywh_to_cxcywh_norm(gts[0], size_wh)], dtype=torch.float32, device=device)  # (1,4)
                    gt_first = gt0_norm.unsqueeze(0)  # (1,1,4)
                    out = model(
                        imgs,
                        gt_boxes_first_k=gt_first,
                        k_burnin=1, priming_prob=1.0,
                        disable_temporal=(args.variant=='frame_only'),
                        force_prime=True
                    )
                else:
                    out = model(
                        imgs,
                        gt_boxes_first_k=None,
                        k_burnin=0, priming_prob=0.0,
                        disable_temporal=(args.variant=='frame_only')
                    )

                boxes = extract_xywh_per_frame(out, T=imgs.shape[1], img_wh=size_wh)
                all_boxes.extend(boxes.tolist())

            out_fp = os.path.join(args.output_dir, f"{name}.txt")
            save_xywh_txt(out_fp, all_boxes)
            print(f"[rank {rank}] saved {name} -> {out_fp}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()