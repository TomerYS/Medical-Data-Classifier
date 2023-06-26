# TAUIntro2DS_SolutionModel
This repository contains the solution model for the Kaggle competition: TAU Intro2DS - Final Assignment - Spring 2023.

[Kaggle Competition Link](https://www.kaggle.com/competitions/intro2ds-final-assignment-spring-2023/overview)

# score: 0.96772

## Competition Description:

### Our Story:
Within the realm of medical marvels, a dataset of immense potential has surfaced, harboring the secrets to personalized healthcare. This extraordinary collection captures the personal information and results of 20 medical tests conducted on a diverse group of patients from the illustrious nation of Israel.

### The Mission:
As aspiring data scientists, your mission is to navigate this dataset and embark on a crucial classification task: to discern the patients' risk of developing a specific medical condition known as Harmony Syndrome. This syndrome, shrouded in mystery, manifests as an imbalance in the body's intricate systems, affecting individuals in different ways.

### The Goal:
Your goal is to classify the patients into two categories: 'At Risk' and 'Healthy'. By analyzing the patterns hidden within the personal information and test results, you will unlock the keys to early detection and intervention, offering hope for individuals susceptible to Harmony Syndrome.
You will need to create a predictive model capable of identifying those at a higher risk of developing the syndrome, enabling targeted preventive measures and timely medical interventions.

### Why?
By framing the task as identifying the risk of Harmony Syndrome, you will engage with the medical world and its challenges. This task highlights the importance of early detection and intervention, and the potential impact classification efforts can have on improving healthcare outcomes.
The task also emphasizes the value of personalized medicine and the need to uncover patterns and correlations within the dataset to drive advancements in medical research and patient care.

## Instructions:

### Objective:
Your primary goal is to maximize the AUC metric of your trained model.

### Data:
- **Training Data**: You will be provided with a training dataset to train your model.
- **Test Data**: The test dataset will have the last column (label) removed and replaced with zeros. Your task is to submit predictions for these records.

### Submission Format:
Prepare a submission file with two columns. The first column should contain the ID of each record, and the second column should contain the corresponding prediction (real value).

### Scoring:
During the competition, you will receive feedback on the performance of your model based on the provided test data. Your AUC score will be calculated on 50% of the test data. The remaining 50% of the test data will only be revealed once the submission date is over, and the score on this portion will be used for grading.

### Submission Limit:
You are allowed to make a maximum of 5 submissions per day. Use your submissions wisely to refine and improve your model.

### Individual Work:
This project should be completed individually. Collaboration or sharing of code is not allowed.

### Questions:
If you have any questions or need clarifications, please utilize the Moodle platform to ask your queries.

### Moodle:
Upload your most recent code file (.ipynb notebook or .py file) to the Moodle under the "HW 4" section.

Best of luck with your submission, and remember to make the most of your limited submissions!

# The Model:

## XGBoost-based Healthcare Model

The provided Python code is a model built using the XGBoost algorithm, a popular gradient boosting machine learning library. This script uses healthcare data (such as personal and health measurements) of individuals for training and testing.

## Data Ingestion

The data comes from two separate datasets: `personal_info_train.csv` (and test counterpart) and `measurements_results_train.csv` (and test counterpart), which are merged into single datasets for both training and testing based on the 'patient_id'.

The personal information dataset is expected to contain demographic and other personal details about the patients. The measurements dataset holds the results of various tests and measurements taken from the patient.

## Data Preprocessing

This script includes extensive data preprocessing steps:

1. It checks and removes any duplicate records based on 'patient_id'.
2. The 'gender' categorical feature is converted to a binary numeric representation using a simple map (e.g., Male: 0, Female: 1).
3. Outliers in the 'height' and 'weight' fields are handled and the Body Mass Index (BMI) is calculated. The missing BMIs are imputed using median Imputer.
4. Missing values in certain columns are handled by creating a new '_flag' column to indicate missingness and imputing missing values with median values.
5. The 'country' and 'region' columns are dropped. Other categorical columns with missing values are filled with 'NaN' or 'Missing'.
6. Date fields 'created_at' and 'birth_date' are converted to the datetime format, and new features ('created_year', 'created_month', 'created_day', and 'age') are extracted from these. After feature extraction, the original date fields are dropped.
7. The 'label' field (the target variable) and 'patient_id' field are separated from the training data.
8. Categorical features are encoded using LabelEncoder.

The preprocessed datasets are saved to 'final_procc.csv' for further use.

## Model Pipeline

The script constructs a pipeline consisting of a column transformer (for one-hot encoding of categorical features) and an XGBoost classifier. The training data is split into a training and a validation set.

## Model Training and Evaluation

The model is trained using the training set and then used to predict probabilities for the training set, validation set, and test set. It calculates the ROC AUC score for both the training set and validation set. Additionally, it logs several other metrics like accuracy, precision, recall, F1-score, log loss, MCC, balanced accuracy, and confusion matrix for the validation set predictions.

## Feature Importance

A bar plot is created to visualize the feature importances derived from the model, helping us understand which features have the most influence on the model's predictions.

## Model Interpretation

A confusion matrix and ROC curve are plotted to understand the performance of the model visually.

## Submission

Finally, the predicted probabilities for the test set are saved in a CSV file named 'mysubmission-XGBoost3.5_without_SMOTE(12).csv' and the model is saved in a pickle file named 'XGBoost_model3.5_without_SMOTE(12).pickle' for future use.

Also, the XGBoost Decision Tree is plotted to understand the decision-making process of the model.

The script uses logging to keep track of various steps and metrics throughout the process.
