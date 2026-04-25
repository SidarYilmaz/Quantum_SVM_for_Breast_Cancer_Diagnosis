# Design of PSO-Based Hybrid Quantum Support Vector Machines for Breast Cancer Diagnosis

This repository presents a research project integrating **Quantum Machine Learning (QML)** with **Meta-heuristic Optimization** to improve diagnostic accuracy on the Wisconsin Breast Cancer dataset. The proposed **PSO-QSVM** architecture achieves **95.61% test accuracy**, outperforming both Classical SVM and standard QSVM baselines.

---

## 💡 Motivation

Breast cancer diagnosis benefits from algorithms that can capture subtle, high-dimensional patterns in medical data. Classical SVMs hit a performance ceiling on this dataset around ~94% accuracy. We asked two questions:

1. **Can quantum kernels capture patterns that classical kernels miss?**
2. **Can we automate the tedious hyperparameter tuning process?**

The answer was a measurable **+1.75% accuracy improvement** (93.86% → 95.61%) — a difference that matters in clinical decision support, where every misclassification has a human cost.

---

## 🚀 Project Overview & Methodology

The core idea is to leverage the high-dimensional feature spaces of quantum mechanics to identify complex patterns in medical data that classical algorithms might miss.

- **Quantum Data Encoding:** Classical medical features are transformed into quantum states using **Amplitude Encoding**, allowing for efficient representation in Hilbert space.
- **Hybrid Classification (QSVM):** Unlike classical SVM, our model utilizes **Quantum Kernels** calculated on quantum circuits to measure data similarity with higher precision.
- **Automated Optimization (PSO):** To eliminate the inefficiency of manual tuning, the critical **C-parameter** is optimized using **Particle Swarm Optimization (PSO)**, ensuring the model reaches its peak performance automatically.

---

## 📊 Comparative Performance Analysis

The proposed **PSO-QSVM** model was benchmarked against classical SVM and standard QSVM models. The results demonstrate a clear advantage:

| Model Architecture | Test Accuracy |
| --- | --- |
| Classical SVM | 93.86% |
| QSVM | 94.74% |
| **PSO-QSVM (Proposed)** | **95.61%** |

---

## 📋 Technical Specifications

- **Dataset:** Wisconsin Breast Cancer (Diagnostic) Dataset — 569 samples, 30 features
- **Preprocessing:** Min-Max Scaling and L² Normalization to satisfy quantum state requirements
- **Stack:** Python, Qiskit, PennyLane, scikit-learn, NumPy, Pandas
- **Optimization:** Particle Swarm Optimization for SVM C-parameter tuning

---

## ⚙️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/SidarYilmaz/Quantum_SVM_for_Breast_Cancer_Diagnosis.git
cd Quantum_SVM_for_Breast_Cancer_Diagnosis

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and run the notebooks
jupyter notebook notebooks/
```

---

## 📁 Repository Structure
.  
├── notebooks/              # Jupyter notebooks (main experiments)  
├── src/                    # Reusable Python modules  
├── experiments/results/    # Output metrics, plots, logs  
├── requirements.txt        # Python dependencies  
└── README.md  

---

## 👥 Research Team

This project was developed at **Karadeniz Technical University**:

- **Prof. Dr. Tuğrul ÇAVDAR** — Supervisor
- **İrem Buse ÖZKÖSE** — Researcher
- **Sidar YILMAZ** — Researcher

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{yilmaz2025psoqsvm,
  title  = {Design of PSO-Based Hybrid Quantum Support Vector Machines for Breast Cancer Diagnosis},
  author = {Yılmaz, Sidar and Özköse, İrem Buse and Çavdar, Tuğrul},
  year   = {2025},
  institution = {Karadeniz Technical University},
  url    = {https://github.com/SidarYilmaz/Quantum_SVM_for_Breast_Cancer_Diagnosis}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
