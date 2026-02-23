# Design of PSO-Based Hybrid Quantum Support Vector Machines for Breast Cancer Diagnosis

This repository presents a cutting-edge approach to medical data classification by integrating **Quantum Machine Learning (QML)** with **Meta-heuristic Optimization**. The project focuses on enhancing the diagnostic accuracy of breast cancer using a hybrid **PSO-QSVM** architecture.

## 🚀 Project Overview & Methodology

The core of this research lies in leveraging the high-dimensional feature spaces of quantum mechanics to identify complex patterns in medical data that classical algorithms might miss.

* **Quantum Data Encoding:** Classical medical features are transformed into quantum states using **Amplitude Encoding**, allowing for efficient representation in Hilbert space.
* **Hybrid Classification (QSVM):** Unlike classical SVM, our model utilizes **Quantum Kernels** calculated on quantum circuits to measure data similarity with higher precision.
* **Automated Optimization (PSO):** To eliminate the inefficiency of manual tuning, the critical **C-parameter** is optimized using **Particle Swarm Optimization (PSO)**, ensuring the model reaches its peak performance automatically.

## 📊 Comparative Performance Analysis

The proposed **PSO-QSVM** model was benchmarked against classical SVM and standard QSVM models. The results demonstrate a clear advantage:

| Model Architecture | Test Accuracy |
| :--- | :--- |
| **Classical SVM** | 93.86% |
| **QSVM** | 94.74% |
| **PSO-QSVM (Proposed)** | **95.61%** |

<img width="500" height="395" alt="image" src="https://github.com/user-attachments/assets/bbc8958f-3876-4681-b369-219655e3b1a9" />


## 📋 Technical Specifications

* **Dataset:** Wisconsin Breast Cancer (Diagnostic) Dataset (569 samples, 30 features).
* **Preprocessing:** Data was normalized using **Min-Max Scaling** and **L² Normalization** to satisfy quantum state requirements.
* **Stack:** Developed in Python using **Qiskit**, **PennyLane**, and **Scikit-learn**.

## 👥 Research Team
This project was developed at Karadeniz Technical University by:
* **Prof. Dr. Tuğrul ÇAVDAR**
* **İrem Buse ÖZKÖSE**
* **Sidar YILMAZ**
