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







DATHEON MACHINE LEARNING & DATA SCIENCE INTERNSHIP
TASK 2: REAL ESTATE HOUSING PRICE PREDICTION (RIDGE & LASSO REGRESSION)
This project is part of the Datheon Machine Learning & Data Science Internship.
Project Overview
This task implements an end-to-end supervised regression pipeline to predict housing prices using the California Housing dataset.

The project covers:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature preparation
- Multicollinearity assessment using Variance Inflation Factor (VIF)
- Baseline Ordinary Least Squares (OLS) regression
- Regularization using Ridge and Lasso regression
- Hyperparameter tuning
- Model evaluation using RMSE, MAE, and R²
- Comparison of regression model performance
Dataset
The project uses the California Housing dataset.

The dataset contains housing-related features such as:
- Median income
- Housing median age
- Average rooms
- Average bedrooms
- Population
- Average occupancy
- Latitude
- Longitude
The target variable is the median house value.
Project Workflow
The notebook follows these main steps:
1. Load the California Housing dataset.
2. Inspect the dataset structure and summary statistics.
3. Perform exploratory data analysis and visualize feature distributions.
4. Prepare the data for regression modeling.
5. Assess multicollinearity using VIF.
6. Train an OLS regression model as the baseline.
7. Train Ridge and Lasso regression models.
8. Tune regularization parameters where applicable.
9. Evaluate and compare models using RMSE, MAE, and R².
Models Implemented
1. Ordinary Least Squares (OLS) Regression
OLS is used as the baseline model to estimate the relationship between housing features and median house value.
2. Ridge Regression
Ridge regression applies L2 regularization to penalize large coefficients and help manage multicollinearity.
3. Lasso Regression
Lasso regression applies L1 regularization. It can shrink coefficients toward zero and may perform feature selection.
Evaluation Metrics
The models are evaluated using the following metrics:
- RMSE (Root Mean Squared Error): Measures the square root of the average squared prediction error.
- MAE (Mean Absolute Error): Measures the average absolute difference between predicted and actual values.
- R² (Coefficient of Determination): Measures the proportion of target variance explained by the model.
Model Results
The notebook contains results from more than one run. The following values are reported in one run:
Model	RMSE	MAE	R²
OLS Regression	69,047.93	49,376.47	0.6362
Ridge Regression (α = 1)	69,055.96	49,382.62	0.6361
Lasso Regression (α = 1)	69,048.99	49,377.33	0.6362


A separate run reports:
Model	RMSE	MAE	R²
OLS Regression	69,126.81	49,645.51	0.6353
Ridge Regression	69,134.74	49,652.16	0.6353
Tuned Lasso Regression	69,742.36	49,957.73	0.6288


Note: These metric tables correspond to separate notebook runs and are presented separately rather than combined.
Key Observations
- OLS, Ridge, and Lasso produced broadly similar performance in the reported results.
- Ridge regression showed a small difference from OLS in the first reported run.
- The tuned Lasso result in the separate run had a lower R² and higher error metrics than the OLS and Ridge results in that run.
- Regularization provides a way to control coefficient magnitude and address multicollinearity.
Project Structure
Task-2-Housing-Price-Prediction/
├── README.md
├── notebook/
│   └── housing_price_prediction.ipynb
├── data/
│   └── california_housing.csv
└── requirements.txt
The directory and file names above are an illustrative structure; adjust them to match the actual repository.
Requirements
The notebook may use the following Python libraries:
- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- statsmodels
How to Run
1. Clone or download the project repository.
2. Install the required packages.
3. Open the Jupyter Notebook.
4. Run the notebook cells in order.
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
Conclusion
This task demonstrates a regression workflow for housing price prediction using OLS, Ridge, and Lasso regression. It includes exploratory analysis, VIF-based multicollinearity assessment, regularization, hyperparameter tuning, and evaluation with RMSE, MAE, and R².

