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




# Datheon Machine Learning & Data Science Internship

## TASK 2: Real Estate Housing Price Prediction (Ridge & Lasso Regression)

This project is part of the Datheon Machine Learning & Data Science Internship.

### Project Overview

This task implements a supervised regression pipeline to predict real estate housing prices using the California Housing dataset.

The project covers:

- Data loading and exploration using Pandas
- Exploratory data analysis and visualization
- Data preprocessing and feature preparation
- Multicollinearity assessment using Variance Inflation Factor (VIF)
- Ordinary Least Squares (OLS) regression
- Ridge Regression using L2 regularization
- Lasso Regression using L1 regularization
- Hyperparameter tuning
- RMSE, MAE, and R² evaluation
- Model performance comparison

### Dataset

The project uses the California Housing dataset.

The dataset contains housing-related features, including:

- median_income
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- latitude
- longitude

The target variable is the median house value.

### Models

Three regression models were trained and evaluated:

1. Ordinary Least Squares (OLS) Regression
2. Ridge Regression
3. Lasso Regression

### Results

One reported notebook run produced the following results:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| OLS Regression | 69,047.93 | 49,376.47 | 0.6362 |
| Ridge Regression (α = 1) | 69,055.96 | 49,382.62 | 0.6361 |
| Lasso Regression (α = 1) | 69,048.99 | 49,377.33 | 0.6362 |

A separate notebook run reported:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| OLS Regression | 69,126.81 | 49,645.51 | 0.6353 |
| Ridge Regression | 69,134.74 | 49,652.16 | 0.6353 |
| Tuned Lasso Regression | 69,742.36 | 49,957.73 | 0.6288 |

The metric tables represent separate notebook runs.

### Visualizations

The notebook includes exploratory visualizations and analysis of housing features, along with model evaluation and multicollinearity analysis.

### Project File

`Machine Learning & Data Science Internship.ipynb`

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Seaborn
- Matplotlib
- Jupyter Notebook


