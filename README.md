# **SMOC-Net**  
**Leveraging Camera Pose for Self-Supervised Monocular Object Pose Estimation**  

A self-supervised framework for 6D object pose estimation that eliminates differentiable renderers via **relative-pose constraints**, achieving SOTA performance with faster training.

## **✨ Key Features**
- 🚀 **Faster Training** than differentiable-renderer methods
- 🔄 **Relative-Pose Constraint** bridges synthetic-to-real domain gap
- 🎯 **Teacher-Student Architecture** with geometric refinement

## **🚀 Quick Start**
### **1. Clone Repository**
```bash
git clone https://github.com/tan-tao11/SMOC-Net.git
cd SMOC-Net
```

### **2. Install Dependencies**
```bash
conda create -n smoc-net python=3.9 -y
conda activate smoc-net
pip install -r requirements.txt
```

### **3. Prepare Datasets**
Dataset Structure:
```bash
datasets/BOP_DATASETS/
  ├── lm/    # LineMOD
  └── lmo/   # LineMOD-Occluded
```
🔗 Download demo data:
[OneDrive Link](https://1drv.ms/u/c/054882095addfd6a/EW_W2NediVxLk7Yi2T43ST8BUIrTEqDZJhgx37sOqzjMqg?e=Fe0S5Z)

### **4. Download Pre-trained Weights**
```bash
mkdir -p weights/gdrn/lm
```
🔗 Download weights:
[OneDrive Link](https://1drv.ms/u/c/054882095addfd6a/EXkfGthAF2hFsEgJhNHIa5cBN7XR-ELVALWfefOjmv4V1Q?e=4rNqoX)

### **5. Run Training**
```bash
python train.py --config configs/train_ape.yaml
```

## 📖 Citation
```bash
@inproceedings{tan2023smoc,
  title     = {SMOC-Net: Leveraging Camera Pose for Self-Supervised Monocular Object Pose Estimation},
  author    = {Tao Tan and Qiulei Dong},
  booktitle = {CVPR},
  pages     = {21307--21316},
  year      = {2023}
}
```

## 🎯 Acknowledgements
Built upon:

- [Self6dpp](https://github.com/THU-DA-6D-Pose-Group/Self6dpp)
- [GDR-Net](https://github.com/THU-DA-6D-Pose-Group/GDR-Net)

We thank the authors for their excellent contributions!
