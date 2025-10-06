# Heart Disease Prediction using Machine Learning
## Project Overview
This project aims to analyze, predict, and visualize heart disease risks using machine learning techniques. The workflow includes data preprocessing, feature selection, dimensionality reduction (PCA), model training, evaluation, and deployment. Both supervised and unsupervised learning models are implemented to provide insights and predictions regarding heart disease.

## Objectives
- Perform data preprocessing and cleaning (handle missing values, encode categorical variables, scale features).  
- Apply **PCA** for dimensionality reduction.  
- Perform feature selection using statistical and ML-based techniques.  
- Train supervised models:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine (SVM)  
- Apply unsupervised models:  
  - K-Means Clustering  
  - Hierarchical Clustering  
- Optimize model performance using **GridSearchCV** and **RandomizedSearchCV**.
  
## Project Workflow

1. **Data Preprocessing**  
   - Load the dataset and handle missing values.  
   - Encode categorical variables and scale numerical features.  
   - Perform exploratory data analysis (EDA) with visualizations.  

2. **Dimensionality Reduction (PCA)**  
   - Apply PCA to reduce feature dimensionality while retaining variance.  
   - Visualize explained variance and principal components.  

3. **Feature Selection**  
   - Rank features using Random Forest or XGBoost importance scores.  
   - Apply Recursive Feature Elimination (RFE) and Chi-Square Test.  
   - Select the most relevant features for modeling.  

4. **Supervised Learning**  
   - Train models: Logistic Regression, Decision Tree, Random Forest, and SVM.  
   - Evaluate performance using accuracy, precision, recall, F1-score, ROC curves, and AUC.  

5. **Unsupervised Learning**  
   - Apply K-Means and Hierarchical Clustering.  
   - Visualize clusters and compare with actual disease labels.  

6. **Hyperparameter Tuning**  
   - Optimize model parameters using GridSearchCV and RandomizedSearchCV.  
   - Compare tuned models against baseline performance.  

7. **Model Export & Deployment**  
   - Save trained models and pipelines using `joblib`
  

## Dataset

- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)  
- **Features:** 14 attributes including age, sex, chest pain type, resting blood pressure, cholesterol, and more.  
- **Target:** Presence (1) or absence (0) of heart disease.  
- **Status:** Dataset has been fully preprocessed and used in the completed analysis.


