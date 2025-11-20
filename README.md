# SOTFormer

Official code and video demonstrations for  
**SOTFormer: A Constant-Memory Transformer with Ground-Truth-Primed Initialization for Online Single Object Tracking (CVPR 2026).**

<p align="center">
  <img src="sotformer_architecture.jpg" width="95%">
</p>

---

## 🌟 Overview

SOTFormer is a **constant-memory** online tracker built on Deformable-DETR with two key innovations:

1. **Ground-Truth-Primed (GT-Primed) Initialization**  
   – Removes cold-start drift by swapping the highest-IoU query into slot-0 for the first *K=3* frames.

2. **Constant-Memory Temporal Transformer**  
   – Updates memory with a *detached* refinement state, guaranteeing O(1) GPU memory over long sequences.

3. **Unified Multi-Task Head**  
   – Jointly learns detection, identity consistency, and short-horizon trajectory prediction.

---

# 🚀 Installation

## 1. Clone the repository
```bash
git clone https://github.com/zhongpingDong12/SOTFormer.git
cd SOTFormer
