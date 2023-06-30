# TAUIntro2DS_SolutionModel
This repository contains the solution model for the Kaggle competition: TAU Intro2DS - Final Assignment - Spring 2023.
As for now the main script is not organized and will be splited into different files.

[Kaggle Competition Link](https://www.kaggle.com/competitions/intro2ds-final-assignment-spring-2023/overview)

# score: 0.96772
Best model so far: XGBoost_model(1).pickle

## Project Structure
The code has been organized into multiple files for better modularity and readability. Here's a brief description of each file:

- `data_loading.py`: Contains functions to load the datasets.
- `preprocessing.py`: Includes data preprocessing steps such as handling missing values, feature engineering, and encoding categorical variables.
- `model.py`: Defines the model training and evaluation pipeline.
- `visualize.py`: Provides functions for visualizing model performance, feature importance, and data distributions.
- `submission.py`: Handles predictions and saves the submission file.
- `main.py`: The main script that orchestrates the entire process.

## Usage
To run the solution model, follow these steps:

1. Ensure that you have the necessary dependencies installed. You can find them listed in the `requirements.txt` file.
2. Place the competition datasets (`personal_info_train.csv`, `personal_info_test.csv`, `measurements_results_train.csv`, `measurements_results_test.csv`) in the same directory as the code files.
3. Run the `main.py` script.

## Steps Performed in the Solution Model

1. **Data Loading**: The datasets are loaded using the `load_datasets()` function from `data_loading.py`.
2. **Data Preprocessing**: The loaded datasets are preprocessed using the `preprocess_data()` function from `preprocessing.py`. This step includes handling missing values, feature engineering, and encoding categorical variables.
3. **Model Training and Evaluation**: The preprocessed data is used to train and evaluate the model using the `train_and_evaluate()` function from `model.py`. The model pipeline includes a column transformer for one-hot encoding categorical features and an XGBoost classifier.
4. **Model Visualization**: The model's performance and interpretability are visualized using the `visualize()` function from `visualize.py`. The function plots the confusion matrix, ROC curve, feature importance, and other relevant visualizations.
5. **Prediction and Submission**: The model is used to predict the test set probabilities, and the predictions are saved to a submission file using the `predict_and_save_results()` function from `submission.py`. The submission file is named `mysubmission-XGBoost(1).csv`.
6. **Model Persistence**: The trained model is saved in a pickle file named `XGBoost_model(1).pickle` for future use.

Feel free to explore the individual files to understand the implementation details and customize the code as per your requirements.

For any questions or clarifications, please refer to the competition's Kaggle page or reach out via the Moodle platform.
# The Model

The solution model is based on the XGBoost algorithm, a popular gradient boosting machine learning library. This script uses healthcare data, including personal and health measurements, to train and test the model.

## Data Ingestion

The data is loaded using the `load_datasets()` function from `data_loading.py`. It comes from two separate datasets: `personal_info_train.csv` (and its test counterpart) and `measurements_results_train.csv` (and its test counterpart). These datasets are merged based on the 'patient_id' column to create unified datasets for both training and testing. The personal information dataset contains demographic and personal details of the patients, while the measurements dataset holds the results of various tests and measurements.

## Data Preprocessing

Extensive data preprocessing is performed using the `preprocess_data()` function from `preprocessing.py` to prepare the datasets for modeling. The following steps are carried out:

1. Duplicate records based on 'patient_id' are checked and removed.
2. The 'gender' categorical feature is converted into a binary numeric representation using a mapping (e.g., Male: 0, Female: 1).
3. Outliers in the 'height' and 'weight' fields are handled, and the Body Mass Index (BMI) is calculated. Missing BMIs are imputed using median imputation.
4. Missing values in specific columns are handled by creating a new '_flag' column to indicate missingness and imputing the missing values with medians.
5. The 'country' and 'region' columns are dropped. Other categorical columns with missing values are filled with 'NaN' or 'Missing'.
6. Date fields 'created_at' and 'birth_date' are converted to the datetime format, and new features ('created_year', 'created_month', 'created_day', and 'age') are extracted. The original date fields are then dropped.
7. The 'label' field (the target variable) and 'patient_id' field are separated from the training data.
8. Categorical features are encoded using LabelEncoder.

The preprocessed datasets are saved to 'final_procc.csv' for further use.

## Model Pipeline

The solution model employs a pipeline defined in `model.py` that consists of a column transformer for one-hot encoding categorical features and an XGBoost classifier. The training data is split into a training set and a validation set.

## Model Training and Evaluation

The model is trained and evaluated using the `train_and_evaluate()` function from `model.py`. The function returns the trained model pipeline (`model_pipeline`) along with the training and validation ROC AUC scores, validation features, labels, and predictions.

## Model Visualization

The `visualize()` function from `visualize.py` is used to generate visualizations of the model's performance. This function plots the confusion matrix, ROC curve, feature importance, and other relevant visualizations based on the trained model pipeline (`model_pipeline`), validation features, labels, and predictions.

## Prediction and Submission

The `predict_and_save_results()` function from `submission.py` is utilized to make predictions on the test data using the trained model pipeline (`model_pipeline`). The function saves the predicted probabilities, training and validation ROC AUC scores, and the test data along with the corresponding IDs to a submission file (`mysubmission-XGBoost(1).csv`).

The trained model is also saved in a pickle file named `XGBoost_model(1).pickle` for future use.

The script uses logging to track various steps and metrics throughout the process.
