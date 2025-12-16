# Anxiety Prediction Project

This project aims to predict anxiety levels in individuals based on behavioral, psychological, lifestyle, and demographic factors using machine learning techniques.

## Project Overview

Social anxiety, or social phobia, affects millions globally and arises from a complex mix of personal and environmental factors. This synthetic dataset simulates real-world patterns and includes high-anxiety cases to support predictive modeling and research in intervention strategies.

The main goal of this project is to predict an individual’s anxiety score (1-10) using a combination of exploratory data analysis, statistical testing, feature engineering, and machine learning models. 

## Dataset

The dataset includes the following types of features:

* Demographics: Age, Gender, Occupation
* Lifestyle: Sleep hours, Physical activity, Diet quality, Alcohol use, Caffeine intake, Smoking habits
* Physical & Mental Health Indicators: Heart rate, Breathing rate, Stress level, Sweating level, Dizziness
* Mental Health History: Use of medication, Therapy frequency, Family history of anxiety, Recent major life events
* Target variable: Anxiety Level (1-10)

**Dataset Summary**:

* No missing values or duplicates
* Small proportion of outliers, the data is retained
* Target variable is right-skewed, ordinal numerical variable
* Certain numerical features (stress levels, therapy sessions, sleep hours, caffeine intake) show strong correlations with anxiety


## Project Workflow

The project is divided into several stages: EDA, statistical testing, feature engineering, model building and hyperparameter tuning. This is a regression task, because of the nature of the target variable, and for this project I have chosen linear regression model, random forest and XGBoost. The metrics used to test the model performance were **root mean squared error (RMSE)** to penalize larger errors more, and **coefficient of determination** $R^2$ to determine the variability of the data explained by the model.

### 1. Exploratory Data Analysis (EDA)

* Visualized distributions of numerical and categorical features
* Identified key predictors: Stress Level, Therapy Sessions, Sleep Hours, Caffeine Intake
* Observed higher anxiety levels among certain occupations, including nurses, lawyers, scientists, doctors, students, engineers, freelancers, and athletes
* Categorical factors influencing anxiety: Smoking, Medication, Major Life Events, Occupation, Family History of Anxiety
* Younger people in their 30s report higher anxiety levels than people in their 40s and 50s.
* Therapy sessions and Sleep Hours show moderate linear correlation, while for all other features correlations are negligible
* Confirmed minimal multicollinearity among predictors

### 2. Statistical Testing

* Confirmed EDA findings using t-tests, Mann-Whitney U-tests, ANOVA, and Kruskal-Wallis tests, followed with post-hoc tests like Tukey's HSD and Dunn's test. Compared the results of parametric to non-parametric tests to determine which ones to rely on for this data.
* Confirmed that the top contributors to anxiety are Stress Level, Sleep Hours, Family History of Anxiety, followed by Caffeine Intake and the number of Monthly Therapy Sessions
* Minor contributions to anxiety levels from smoking, dizziness, medication, or recent life events
* Transformation considered for skewed predictors (log/Box-Cox) like Therapy Sessions, while for other features this is not needed

### 3. Feature Engineering

* Split data into training and test sets, and applied scaling only to training data to prevent leakage.
* Standardized numerical features to improve model performance, and log-transformed Therapy Sessions due to skew.
* Created Stress Level per Sleep Hour feature — strongest predictor of anxiety.
* Categorical features one-hot encoded (low cardinality, less than 15 categories each).
* Computed mutual information to identify top predictors: Stress Level per Sleep Hour, followed by Stress Level and Sleep Hours, which confirms the findings from the previous stages. Caffeine Intake, Diet Quality, Alcohol Consumption have a smaller effect.

### 4. Linear Regression Modeling

* Used statsmodels for initial model building and diagnostics, as it gives more detailed feedback about the model performance.
* Built linear models in scikit-learn with and without interaction terms, to potentially capture nonlinear relationships and compare their contributions compared to the model without interaction.
* Interaction model improved predictive performance significantly over baseline, with potentially small overfitting.
*  Determined that the interaction model is the best, and makes predictions within approximately 1 point, on average, for the Anxiety Level. 
* RMSE on the training set was 1.02, while on the test set it was 1.08.
* Checked model assumptions: linearity, residual normality, homoscedasticity, multicollinearity. No assumptions are violated.


### 5. Random Forest & XGBoost (In Progress)

1. **Random Forest**:
* Initial model results indicate overfitting issues, with training RMSE of 0.38, and test RMSE of 1.02.
* Most important features using built-in feature importance: Stress Level (47%), Sleep Hours (22%), Therapy Sessions (9%), Caffeine Intake (4%); others contribute minimally
* Model naturally captures nonlinear relationships, but needs hyperparameter tuning to improve generalization

2. **XGBoost**:
* Initial model results indicate overfitting issues, with training RMSE of 0.94, and test RMSE of 1.02, indicating smaller overfitting issue than for Random Forest model.
* Captures complex relationships in data, and has a performance on the test set similar to Random Forest.
* Should be tuned to improve performance.

  
### 6. Hyperparameter tuning (In Progress)

Hyperparameter tuning will be performed in the future for Random Forest and XGBoost to improve model generalization. Therefore, there are no insights yet.


## Key Takeaways

* Top predictors of anxiety are Stress levels, sleep hours, and family history of anxiety.
* Moderate predictors are Therapy sessions per month and daily caffeine intake.
* Certain occupations (e.g., nurses, lawyers, scientists, students, engineers, freelancers, athletes) show higher anxiety levels.
* Top predictors of anxiety (Stress levels, Sleep hours, Family History of Anxiety) were validated during all stages, using statistical testing and mutual information scores to confirm their contribution.
* Interaction terms improve linear regression performance, but tree-based models (Random Forest, XGBoost) are better at capturing nonlinear relationships and are expected to outperform linear regression model with adequate hyperparameter tuning.
* Creating derived features (e.g., Stress Level per Sleep Hour) and scaling/encoding can potentially improve model performance and interpretability, with adequate hyperparameter tuning.
* Additional feature engineering can be performed, such as creating ratios, grouping data into bins, and evaluating by combining permutation feature importance with SHAP values to better understand both global and local contributions of predictors.
