# 💧 Water Quality Prediction Using Machine Learning

## 📘 Overview

This project analyzes and predicts the potability of water based on physicochemical properties using machine learning. It demonstrates my ability to clean, visualize, and model real-world environmental data, aligning with my interest in applying AI to public health and bioinformatics challenges.

## 🎯 Objectives

- Explore and visualize water quality features
- Build classification models to predict water potability
- Evaluate model performance using multiple metrics
- Identify key features influencing water safety

## 📂 Dataset

- **Source**: Kaggle - Water Potability Dataset
- **Features**: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity
- **Target**: `Potability` (0 = Not Safe, 1 = Safe)
- **Preprocessing**:
  - Missing values filled with column means
  - Standardization using `StandardScaler`

## 🛠️ Tools & Technologies

- **Languages**: Python
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Models**: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Decision Tree

## 📊 Exploratory Data Analysis

- **Histograms**: Distribution of each feature
- **Boxplots**: Outlier detection
- **Correlation Matrix**: Feature relationships
- **Pairplots**: Visualizing potability clusters
- **Scatter Plots**: Feature interactions (e.g., Hardness vs. Conductivity)

## 🤖 Machine Learning Models

### 🔹 Classification Models

- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**

Each model was evaluated using:
- Accuracy
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance

### 🏆 Best Model
- **Selected Based On**: Highest accuracy and balanced performance
- **Feature Importance**: Visualized for interpretability
## 📈 Results

- **Best Accuracy**: Achieved with [insert best model name]
- **Key Features**: [e.g., Organic Carbon, Conductivity, Sulfate]
- **Prediction Output**: CSV file with predicted potability for unseen data

## 👨‍🔬 About Me

I'm **Abdur-Rasheed Abiodun Adeoye**, a data scientist with a strong interest in applying AI to biological and environmental challenges. This is part of my portfolio for in **Bioinformatics**.

## 📎 How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/water-quality-ml.git
cd water-quality-ml

# Run the Python script
python Water_Quality_DS_Project.py



