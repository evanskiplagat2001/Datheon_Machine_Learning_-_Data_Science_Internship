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






# Task 3: Predictive Customer Churn Classifier (Random Forest & XGBoost)

## Project Overview

This project develops machine learning models to predict customer churn in the telecommunications industry. It compares Random Forest and XGBoost classifiers to identify customers who are likely to leave a service.

The project also applies decision threshold optimization to improve churn detection and support customer retention strategies.

## Objectives

- Explore customer churn patterns and relationships with customer characteristics.
- Clean and preprocess the dataset for machine learning.
- Train Random Forest and XGBoost classification models.
- Evaluate model performance using multiple classification metrics.
- Optimize the XGBoost decision threshold using cross-validation.
- Interpret model results from a business perspective.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

**Dataset file:** `Telco-Customer-Churn.csv`

The dataset contains customer demographic information, account details, subscribed services, payment methods, and churn status.

The target variable is `Churn`, which indicates whether a customer has left the service.

Data cleaning included:
- Converting `TotalCharges` to numeric format.
- Handling missing values.
- Checking and removing duplicate records.
- Excluding `customerID` from model features.

## Project Workflow

1. Import required Python libraries.
2. Load and inspect the dataset.
3. Perform data cleaning and preprocessing.
4. Conduct exploratory data analysis (EDA).
5. Visualize customer churn patterns.
6. Prepare predictor variables and target variable.
7. Split data into training and testing sets using stratified sampling.
8. Encode categorical variables using One-Hot Encoding.
9. Train the Random Forest classifier.
10. Train the XGBoost classifier.
11. Evaluate and compare both models.
12. Analyze XGBoost feature importance.
13. Plot ROC curves.
14. Optimize the XGBoost classification threshold using cross-validation.
15. Evaluate optimized XGBoost at a threshold of 0.55.
16. Conduct final model comparison.
17. Analyze the business impact of churn predictions.

## Machine Learning Models

### Random Forest

Random Forest was used as a baseline ensemble classification model. Class balancing was applied to account for the imbalance in churn classes.

### XGBoost

XGBoost was implemented as a gradient boosting classifier. The `scale_pos_weight` parameter was used to address class imbalance.

The decision threshold was optimized using cross-validation on the training data, with the selected threshold being 0.55.

## Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- ROC-AUC
- PR-AUC
- Negative Predictive Value (NPV)

## Final Results

### Random Forest

| Metric | Score |
|---|---:|
| Accuracy | 78.46% |
| Precision | 62.46% |
| Recall (Sensitivity) | 47.59% |
| Specificity | 89.64% |
| F1-Score | 54.02% |
| ROC-AUC | 81.23% |
| PR-AUC | 58.81% |
| NPV | 82.54% |

### Optimized XGBoost (Threshold = 0.55)

| Metric | Score |
|---|---:|
| Accuracy | 74.98% |
| Precision | 52.05% |
| Recall (Sensitivity) | 74.60% |
| Specificity | 75.12% |
| F1-Score | 61.32% |
| ROC-AUC | 83.00% |
| PR-AUC | 64.51% |
| NPV | 89.09% |

## Model Comparison

Random Forest achieved higher accuracy, precision, and specificity.

Optimized XGBoost achieved higher recall, F1-Score, ROC-AUC, PR-AUC, and NPV.

Since the main objective is to identify customers who are likely to churn, optimized XGBoost was selected as the preferred model for this project.

## Business Impact Analysis

The optimized XGBoost model was evaluated on 1,407 test customers.

| Business Metric | Result |
|---|---:|
| Correctly identified churners | 279 |
| Unnecessary retention targets | 257 |
| Missed churners | 95 |
| Correctly identified retained customers | 776 |
| Churners identified | 74.60% |
| False positive rate | 24.88% |
| Customers flagged for retention | 38.10% |

The model can help businesses identify customers at risk of leaving and prioritize retention campaigns. However, some customers flagged by the model may not actually churn, which creates a trade-off between customer retention efforts and campaign costs.

## Technologies and Libraries

The project was developed using Python and Jupyter Notebook.

Libraries used include:

- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost

## Installation

Install the required Python libraries using:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost
```

## How to Run the Project

1. Clone or download the project repository.
2. Place `Telco-Customer-Churn.csv` in the same directory as the Jupyter Notebook.
3. Open the notebook using Jupyter Notebook or JupyterLab.
4. Select **Kernel → Restart Kernel and Run All Cells**.
5. Review the generated visualizations, evaluation metrics, and model comparisons.

## Conclusion

This project demonstrates a complete machine learning workflow for customer churn prediction, including data cleaning, exploratory analysis, feature preprocessing, model training, evaluation, and threshold optimization.

Optimized XGBoost achieved a recall of 74.60%, an F1-Score of 61.32%, and a ROC-AUC of 83.00%.

The results indicate that XGBoost is more effective at identifying potential churners than the Random Forest baseline. The model could support data-driven customer retention strategies by helping businesses prioritize customers who are at risk of leaving.


