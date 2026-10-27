# Datheon Machine Learning & Data Science Internship

## Task 1: Multi-Class Benchmark Classification Pipeline

This project is part of the Datheon Machine Learning & Data Science Internship.

### Project Overview

This task implements an end-to-end supervised multi-class classification pipeline using the Iris dataset.

The project covers:

- Data loading and exploration using Pandas
- Exploratory data analysis using Seaborn
- Feature scaling using StandardScaler
- 80/20 training and testing split
- Logistic Regression classification
- Decision Tree classification
- Accuracy, precision, recall, and F1-score evaluation
- Classification reports
- Confusion matrix visualization
- Decision Tree decision boundary visualization
- Model performance comparison

### Dataset

The project uses the Iris dataset containing 150 observations and 5 columns:

- sepal_length
- sepal_width
- petal_length
- petal_width
- species

The dataset contains three species:

- Setosa
- Versicolor
- Virginica

### Models

Two supervised classification models were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier

### Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| Decision Tree | 1.00 | 1.00 | 1.00 | 1.00 |

Both models correctly classified all 30 observations in the test set.

### Visualizations

The notebook includes:

- Iris feature pairplot
- Logistic Regression confusion matrix heatmap
- Decision Tree confusion matrix heatmap
- Decision Tree decision boundary visualization

### Project File

`Machine Learning & Data Science Internship.ipynb`

### Technologies Used

- Python
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib
- Jupyter Notebook
