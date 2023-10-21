# Machine Learning Model for Environment Water Prediction

<h3 align="left"><a href="https://huggingface.co/spaces/yosefferiyanto/Water_Prediction_Milestone_2_Hacktiv8">Machine learning application to predict environment water.</a></h3>

## Owner

- [Yosef Feriyanto](https://www.linkedin.com/in/yosef-feriyanto-522754175/)

## Project's Background

**Background:**

 This model will be very helpful in making decisions regarding the suitability of a water source for use, so that it can prevent the spread of disease caused by contaminated water from the data obtained.

**Problem Statement:**

Develop and analyze a classification model which aims to create a model that can predict whether a water environment is safe or not, based on the content of elements, compounds, microbes and viruses in water.

## Datasets

**Dataset Overview:**

- Dataset Source: This dataset is obtained from the Kaggle public dataset
- The data used is a classification of whether a water environment is safe or not based on the content of elements, compounds, microbes and viruses in it. All elements and compound content units in this dataset use ppm or parts per million, bacteria using CFU/mL or colony forming unit, and viruses using PFU/mL or plaque forming unit.

## Objectives

- Data Preparation and Exploration: To clean and explore the dataset to understand the variables that influence environemntal water.
- Predictive Modeling: To develop a classification model that can predict wether the environmental water is safe or not with an recall score of at least 80%.
- Scoring: Recall to minimize false negative (predicted safe but actually doesn't) with score at least 90% & Precision-Recall-AUC score at least 70%.
- Evaluation: To evaluate the effectiveness of the model and the proposed retention strategies within a set timeframe (e.g., 3 months).

## Demo

App Demo Link : https://huggingface.co/spaces/yosefferiyanto/Water_Prediction_Milestone_2_Hacktiv8

## Technology Stacks

#### Data Manipulation and Analysis:
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

#### Data Preprocessing:
- **Scikit-learn (sklearn)**: Various preprocessing techniques
- **Imbalanced-learn (imblearn)**: Pipeline construction

#### Machine Learning Algorithms:
- **Scikit-learn (sklearn)**
  - Ada Boosting: Classification
  - Random Forest: Classification
  - k-Nearest Neighbors (k-NN): Classification
  - Decision Trees: Classification
  - Support Vector Machines (SVM): Classification
- **Hugging Face**: Model deployment

#### Model Evaluation Techniques:
- **Scikit-learn (sklearn)**
  - Stratified K-Fold: Cross-validation
  - Randomized Search CV: Hyperparameter tuning
  - Metrics: 
    - F1 Score
    - Recall
    - Precision
    - Accuracy
    - ROC-AUC
    - PR-AUC
  - Confusion Matrix
  - Precision-Recall Curve
  - ROC Curve

#### Data Visualization:
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Plotting and visualization

#### Miscellaneous:
- **Python's Standard Library**: Warning control
- **Python's Standard Library**: Time
