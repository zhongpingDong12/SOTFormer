import os, json, argparse, math, csv
import numpy as np

def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1+aw, ay1+ah
    bx2, by2 = bx1+bw, by1+bh
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = aw*ah + bw*bh - inter
    return inter/ua if ua>0 else 0.0

def center_err(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy = ax+aw/2.0, ay+ah/2.0
    bcx, bcy = bx+bw/2.0, by+bh/2.0
    return math.hypot(acx-bcx, acy-bcy)

def load_results_one(res_file):
    return [list(map(float, l.strip().split(','))) for l in open(res_file,'r') if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_json', required=True, help=".../annotations/val/val_compat.json")
    ap.add_argument('--results_dir', required=True, help="dir containing <seq>.txt")
    ap.add_argument('--out_csv', default='./val_metrics.csv')
    ap.add_argument('--diag_norm', type=float, default=0.005, help="Normalized precision threshold as ratio of image diagonal (default 0.5%)")
    ap.add_argument('--fps_hint', type=float, default=None, help="Optional FPS to print")
    args = ap.parse_args()

    meta = json.load(open(args.val_json,'r'))
    if isinstance(meta, list):
        d={}
        for i, it in enumerate(meta):
            name = it.get('name') or it.get('sequence') or f'vid_{i+1}'
            d[name]=it
        meta=d

    seq_names = list(meta.keys())
    if len(seq_names)==0:
        print("[FATAL] no sequences in val_json.")
        return

    ious_all, prec20_all, nprec_all = [], [], []
    rows = [("seq","SuccessAUC","Precision@20px","NormPrec@{:.2f}%".format(args.diag_norm*100))]

    for seq in seq_names:
        info = meta[seq]
        imgs = info.get('img_names') or info.get('images') or info.get('frames')
        gts  = info.get('gt_rect')  or info.get('gt')     or info.get('ann') or info.get('bboxes')
        if not imgs or not gts:
            continue

        res_file = os.path.join(args.results_dir, f"{seq}.txt")
        if not os.path.isfile(res_file):
            print(f"[WARN] missing result for {seq}, skip")
            continue
        pred = load_results_one(res_file)

        T = min(len(pred), len(gts))
        if T == 0: 
            continue

        ious = np.array([iou_xywh(pred[i], gts[i]) for i in range(T)], dtype=np.float32)
        thr = np.linspace(0,1,101)
        succ = [(ious>=t).mean() for t in thr]
        auc = np.trapz(succ, thr)

        diag = np.array([math.hypot(g[2], g[3]) for g in gts[:T]])
        errs = np.array([center_err(pred[i], gts[i]) for i in range(T)])
        prec20 = (errs <= 20).mean()

        thr_norm = args.diag_norm * diag
        nprec = (errs <= thr_norm).mean()

        ious_all.append(auc)
        prec20_all.append(prec20)
        nprec_all.append(nprec)
        rows.append((seq, f"{auc*100:.2f}", f"{prec20*100:.2f}", f"{nprec*100:.2f}"))

    if len(ious_all)==0:
        print("[FATAL] no valid sequences evaluated (no result files match).")
        return

    mean_auc   = np.mean(ious_all)*100
    mean_p20   = np.mean(prec20_all)*100
    mean_nprec = np.mean(nprec_all)*100

    print("==== Validation (VAL) ====")
    print(f"Success AUC: {mean_auc:.2f}")
    print(f"Precision@20px: {mean_p20:.2f}")
    print(f"Normalized Precision: {mean_nprec:.2f}")
    if args.fps_hint is not None:
        print(f"FPS (hint): {args.fps_hint:.1f}")

    with open(args.out_csv, 'w', newline='') as f:
        cw = csv.writer(f)
        cw.writerows(rows)
        cw.writerow(["MEAN", f"{mean_auc:.2f}", f"{mean_p20:.2f}", f"{mean_nprec:.2f}"])
    print(f"[OK] Wrote per-seq metrics -> {args.out_csv}")

if __name__ == "__main__":
    main()
