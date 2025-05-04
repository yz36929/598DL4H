# 3D CNN for Alzheimer’s Classification

_Reproducing Hao et al. (2019), “CNN Design for Alzheimer’s Disease”_  
**Paper**: https://arxiv.org/pdf/1911.03740  
**Original code**: https://github.com/NYUMedML/CNN_design_for_AD

---

## 📚 References

- **Target paper**: arXiv:1911.03740  
- **Original implementation**: NYUMedML/CNN_design_for_AD  

---

## 🗂️ Data

We originally explored raw ADNI→CLINICA preprocessing but ultimately used pre-processed volumes:
```bash
data/
├── nifti/ # pre-processed .nii / .nii.gz files
└── labels.tsv # subject_id, session, diagnosis, age
```
---

## 🔧 Installation

```bash
conda create -n adni-cnn python=3.9 nibabel pytorch torchvision cudatoolkit=11.8 scikit-learn pyyaml tensorboard
conda activate adni-cnn
pip install pyhealth
```

---

## 🚀 Usage
1. Prepare data
Place your .nii / .nii.gz files under data/nifti/ and create data/labels.tsv.

2. Train the baseline model

```bash
python -m src.train
```
- Normalizes and augments (flips, rotations, translations)

- Trains 3-block 3D CNN (InstanceNorm)

- Logs metrics & graphs to TensorBoard

- Saves best weights to checkpoints/

3. Evaluate
At the end of training, the best model is automatically evaluated on the hold-out test split.

4. PyHealth variant

```bash
python -m src.train_pyhealth
```
Wraps our model for PyHealth’s Trainer API.

---

## 🧪 Ablation & Extensions
Toggle flags or edit code to explore:

- Normalization: InstanceNorm vs. BatchNorm

- Spatial downsampling: impact of early pooling

- Width vs. Depth: scaling filters vs. layers

- Age-incorporation: set include_age: true/false

---

## 📈 Results
After reproducing the baseline, we ran ablations:

1. Normalization: InstanceNorm ≈ 2 % higher balanced accuracy than BatchNorm

2. Downsampling: Early aggressive pooling degrades performance by ≈ 5 %

3. Width vs. Depth: Doubling width yields ≈ 3 % gain; doubling depth shows diminishing returns

4. Age metadata: modest ≈ 1.5 % performance boost

---

## 👤 Acknowledgements
Hao et al. (2019) for the original architecture & augmentations

ADNI consortium for the data

PyHealth for training utilities

