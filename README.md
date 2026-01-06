# blood-cell-image-classification-with-random-forest
# BloodMNIST Classification: Classical ML and CNN Baselines

This project explores blood cell image classification using the **BloodMNIST**
dataset from the MedMNIST v2 collection. The goal is to build clear,
educational, and reproducible baselines using both **classical machine
learning** and **deep learning** approaches.

The notebook walks through the entire workflow:
- loading and preparing the dataset,
- building a classical ML baseline (Random Forest),
- building a deep learning baseline (CNN),
- visualizing training behavior,
- analyzing predictions and misclassifications,
- comparing model performance.

This project is designed to be beginner‑friendly, well‑documented, and suitable
for portfolio use on GitHub and Kaggle.

---

## 🎯 Project Aim

The main objective of this project is to:

- build **interpretable baselines** for blood cell image classification,
- compare classical ML vs deep learning approaches,
- understand how spatial information affects model performance,
- create a clean, educational notebook suitable for public portfolios.

---

## 📚 Dataset: BloodMNIST (MedMNIST v2)

This project uses the **BloodMNIST** dataset from the MedMNIST collection, a
large-scale set of lightweight biomedical image datasets designed for
educational and research purposes.

### 🔗 Dataset Link  
You can download BloodMNIST from the official MedMNIST website:  
https://medmnist.com/

On Kaggle, the dataset is often provided as a single file such as:  
`bloodmnist_224.npz`

### 🧪 What the dataset contains
- **8 classes** of blood cell types  
- **17,092 images** in total  
- Each image is an RGB image (28×28 originally; resized in this project)  
- Pre-split into:
  - **Training set**
  - **Validation set**
  - **Test set**

The `.npz` file typically contains arrays such as:
- `train_images`, `train_labels`
- `val_images`, `val_labels`
- `test_images`, `test_labels`  
(or similarly named keys depending on the source).

---

## 📥 How to use the dataset in this project

This notebook expects a **single `.npz` file**, similar to how it is loaded on Kaggle:

```python
data = np.load('bloodmnist_224.npz')
```

### Steps:

1. Download the BloodMNIST `.npz` file (e.g. `bloodmnist_224.npz`) from MedMNIST or Kaggle.

2. Place it in the **root directory** of this repository:

```
BloodMNIST-Project/
├── bloodmnist_224.npz
├── notebook.ipynb
└── README.md
```

3. The notebook will load it automatically using the path above.

If you rename the file or place it in a different folder, update the path in the notebook accordingly.

---

## 🧪 Methods

### 1. Classical Machine Learning (Random Forest)

We first flatten each image into a feature vector and train a
**Random Forest classifier**. This serves as a classical ML baseline.

**Why this baseline?**  
Random Forests are fast, robust, and provide a good non‑deep‑learning reference.

**Result:**  
- Validation accuracy: **≈86.8%**

---

### 2. Convolutional Neural Network (CNN)

We then train a simple CNN on resized images to capture spatial
patterns that classical ML cannot.

**Why CNN?**  
CNNs learn edges, textures, and morphological features directly from pixels.

**Result:**  
- Validation accuracy: **≈95.4%**

This significantly outperforms the Random Forest baseline.

---

## 📊 Visualizations

The notebook includes several visualizations to make the analysis intuitive:

### ✔ Training curves  
Shows how accuracy and loss evolve over epochs.

### ✔ Confusion matrices  
For both Random Forest and CNN, highlighting class‑wise performance.

### ✔ Sample predictions  
A grid of images with **True vs Predicted** labels  
(green = correct, red = incorrect).

### ✔ Misclassified samples  
A focused look at where the CNN struggles.

These visualizations help interpret model behavior beyond raw accuracy.

---

## 📈 Results Summary

| Model            | Accuracy | Notes |
|------------------|----------|-------|
| Random Forest    | ~86.8%   | Strong classical baseline |
| CNN (simple)     | ~95.4%   | Learns spatial features, much higher performance |

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BloodMNIST-Project.git
   ```

2. Download the BloodMNIST `.npz` file (e.g. `bloodmnist_224.npz`) from MedMNIST or Kaggle.

3. Place the file in the project root:
   ```text
   BloodMNIST-Project/
   ├── bloodmnist_224.npz
   ├── notebook.ipynb
   └── README.md
   ```

4. Open the notebook:
   ```bash
   notebook.ipynb
   ```

5. Run all cells.

---

## 📄 Dataset Citation

If you use this dataset, please cite:

Yang et al., *“MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification”*, Scientific Data, 2023.

---

## 🧭 Future Improvements

Possible extensions include:

- data augmentation,
- deeper CNN architectures,
- transfer learning (ResNet, EfficientNet),
- hyperparameter tuning,
- model interpretability (Grad-CAM).

---

## 🙌 Acknowledgements

This project uses the BloodMNIST dataset from the MedMNIST collection.  
Special thanks to the dataset authors for making biomedical imaging accessible
for research and education.
