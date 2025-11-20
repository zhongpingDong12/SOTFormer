# SOTFormer

Official code and video demonstrations for  
**SOTFormer: A Constant-Memory Transformer with Ground-Truth-Primed Initialization for Online Single Object Tracking (CVPR 2026).**

<p align="center">
  <img src="sotformer_architecture.jpg" width="95%">
</p>


## 🌟 Overview

SOTFormer is a **constant-memory** online tracker built on Deformable-DETR with two key innovations:

1. **Ground-Truth-Primed (GT-Primed) Initialization**  
   – Removes cold-start drift by swapping the highest-IoU query into slot-0 for the first *K=3* frames.

2. **Constant-Memory Temporal Transformer**  
   – Updates memory with a *detached* refinement state, guaranteeing O(1) GPU memory over long sequences.

3. **Unified Multi-Task Head**  
   – Jointly learns detection, identity consistency, and short-horizon trajectory prediction.


# 🚀 Installation

## 1. Clone the repository
```bash
git clone https://github.com/zhongpingDong12/SOTFormer.git
cd SOTFormer
```

##  2. Create environment

```bash
conda create -n sotformer python=3.10 -y
conda activate sotformer
```


## 3. Install PyTorch (modify CUDA version if needed)

Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```


## 4. Install dependencies

```bash
pip install -r requirements.txt
```


## 5. (Optional) Offline HuggingFace Models

If running on servers without internet:

```bash
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

Place the Deformable-DETR weights in:

```
/apps/users/icps/deformable_detr_local/
```

(as used in your code).


# 📁 Dataset Preparation

SOTFormer supports LaSOT, Mini-LaSOT (your custom 20% split), and LaSOT-Car subsets.

Folder structure:

```
datasets/
    LaSOT/
        videos/
        annotations/
```

You may adjust `--data_root` in training scripts.



# 🚀 Training

The training follows your paper and code exactly:

* GT-Primed swap for first **K** frames
* Constant-memory update
* Multi-task loss (CE + L1 + GIoU + ADE + FDE + anchor loss)

Run:

```bash
python train_sotformer.py \
    --data_root /path/to/LaSOT \
    --batch_size 1 \
    --epochs 50 \
    --horizon 10 \
    --burn_in 3 \
    --lr 2e-4 \
    --save_dir checkpoints/
```

During training, the model uses:

* `pred_boxes_raw` for gradients
* `pred_boxes` (overwritten for burn-in frames) for visualization

This matches your implementation exactly.



# 🎯 Inference

### Single video tracking

```bash
python inference.py \
    --video_path input.mp4 \
    --checkpoint checkpoints/sotformer_best.pth \
    --output output.mp4
```

### H-step trajectory forecasting

```bash
python predict_future.py \
    --image path/to/image.jpg \
    --checkpoint checkpoints/sotformer_best.pth \
    --horizon 10
```



# 🖼️ Visualization

### Tracking visualization

```bash
python visualize_tracking.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/sotformer_best.pth \
    --save_path vis/
```

Optional overlays:

```
--draw_gt
--draw_trajectory
```



# 📊 Qualitative Results

<p align="center">
  <img src="figures/scenerios2.1.jpg" width="95%">
</p>

<p align="center">
  <img src="figures/scenerios2.2.jpg" width="95%">
</p>

<p align="center">
  <img src="figures/scenerios2.3.jpg" width="95%">
</p>

These match the eight Mini-LaSOT attributes used in the paper:

* **(a)** Fast Motion (FM)
* **(b)** Occlusion (OCC)
* **(c)** Scale Change (SC)
* **(d)** Illumination Change (IC)
* **(e)** Nighttime (NT)
* **(f)** Background Clutter (BC)
* **(g)** Deformation (DF)
* **(h)** Underwater Environment (UE)

---

# 🎥 **Video Demonstrations**

Will be uploaded here:

👉 **[https://github.com/zhongpingDong12/SOTFormer/videos](https://github.com/zhongpingDong12/SOTFormer/videos)**

Includes:

* Occlusion & re-entry
* Fast motion
* Deformation
* Constant-memory long-sequence demo

---

# 🧩 **Model Checkpoints**

Available after acceptance:

```
SOTFormer-MiniLaSOT  (coming soon)
SOTFormer-LaSOT-Car  (coming soon)
```

---

# 📜 **Citation**

```bibtex
@article{dong2026sotformer,
  title={SOTFormer: A Constant-Memory Transformer with Ground-Truth-Primed Initialization for Online Single Object Tracking},
  author={Dong, Zhongping and Yu, Pengyang and Li, Shuangjian and Chen, Liming and Kechadi, Mohand-Tahar},
  journal={CVPR},
  year={2026}
}
```

---

# 📄 **License**

MIT License — see `LICENSE` file.


