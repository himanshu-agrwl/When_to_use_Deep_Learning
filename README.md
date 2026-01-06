# When_to_use_Deep_Learning
This repository explores model selection across data types by comparing Logistic Regression, ANN, and CNN. It demonstrates why classical ML and ANNs work well for tabular data, while CNNs are necessary for image-based tasks, using breast cancer data and MNIST as case studies.

Overview
This repository presents a comparative study on model selection in machine learning, focusing on when classical models and ANNs are sufficient and when CNNs become essential.

Rather than assuming deep learning is always superior, this project starts with strong baseline models, evaluates their performance, and then justifies the transition to more complex architectures based on data characteristics.

🎯 Objectives

Establish Logistic Regression as a baseline for tabular data

Evaluate whether ANN provides meaningful improvement over classical ML

Demonstrate why CNNs are necessary for image-based tasks

Perform quantitative and qualitative evaluation, not just accuracy comparison

🗂️ Project Structure
ann-to-cnn-model-selection/
│
├── tabular-data/
│   ├── logistic_regression.ipynb
│   ├── ann_tabular.ipynb
│   └── results.md
│
├── image-data/
│   ├── cnn_mnist.ipynb
│   └── results.md
│
├── README.md

📊 Part 1: Tabular Data — Logistic Regression & ANN
Dataset

Breast Cancer Wisconsin Dataset

30 numerical features

Binary classification task

🔹 Baseline Model: Logistic Regression

Logistic Regression is used as the baseline model because:

It performs strongly on linearly separable tabular data

It provides interpretability

It sets a realistic performance benchmark

🔹 ANN Model

A fully connected Artificial Neural Network (ANN) was trained on the same dataset to evaluate whether a deeper model adds value.

✅ Key Observation

ANN and Logistic Regression achieved similar accuracy and confusion matrices

Error patterns were nearly identical

🧠 Insight

On structured tabular data with well-engineered features, increasing model complexity does not necessarily improve performance.

This validates the importance of starting with baselines before applying deep learning.

🖼️ Part 2: Image Data — CNN from Scratch
Dataset

MNIST handwritten digits

28×28 grayscale images

10-class classification problem

🔹 Why CNN?

Unlike tabular data, image data has:

Spatial structure

Local pixel dependencies

Translation invariance requirements

ANNs and classical ML models fail to exploit these properties effectively.

🔹 CNN Architecture

Convolutional layers for feature extraction

MaxPooling for spatial reduction

Dense layers for classification

Softmax output for multi-class prediction

The CNN was built from scratch, without transfer learning.

📈 Evaluation & Analysis
Metrics Used

Accuracy

Confusion Matrix

Prediction Visualization

Error (misclassification) Analysis

🔹 Confusion Matrix Heatmap

Highlights systematic confusions between visually similar digits

Shows errors are due to data ambiguity, not random failure

🔹 Prediction Visualizations

Sample correct predictions for interpretability

Visualization of incorrect predictions for error analysis

Automatic comparison with ground truth labels (no manual inspection)

🧠 Key Learnings

Deep learning is not always necessary

Baselines matter and should not be skipped

ANN ≠ CNN — architecture must match data structure

CNNs excel when spatial information is critical

Model evaluation should include qualitative analysis, not just metrics

🚀 Technologies Used

Python

NumPy

Scikit-learn

TensorFlow / Keras

Matplotlib & Seaborn

Google Colab

📌 Final Takeaway

Model selection should be driven by data characteristics, not by model complexity.
This project demonstrates a principled transition from Logistic Regression → ANN → CNN based on empirical evidence.

🔗 Future Work

Apply CNNs to more complex image datasets (CIFAR-10)

Experiment with data augmentation

Compare CNN vs ANN on flattened image inputs

Explore transfer learning
