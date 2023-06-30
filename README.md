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

1. **Data merging**: The `personal_info_train` and `measurements_train` datasets are merged based on the 'patient_id' column to create the `train` dataset. Similarly, the `personal_info_test` and `measurements_test` datasets are merged to create the `test` dataset.
2. **Removing duplicates**: Duplicate records in the `train` dataset are identified and removed based on the 'patient_id' column using the `drop_duplicates()` function.
3. **Dropping unnecessary columns**: The 'country' and 'region' columns are dropped from both the `train` and `test` datasets using the `drop()` function.
4. **Mapping gender to numeric values**: The 'gender' column in both the `train` and `test` datasets is mapped to numeric values using a mapping dictionary.
5. **Fixing height and weight outliers**: Outliers in the 'height' and 'weight' columns are addressed by applying specific transformations. Heights below 10 are multiplied by 100, and weights above 200 are divided by 1000.
6. **Calculating BMI and filling missing values**: BMI (Body Mass Index) is calculated for records with valid height and weight values. Missing BMI values are imputed using the median BMI value calculated from the available data.
7. **Handling missing values and adding flags**: Certain columns ('test_2', 'test_6', 'test_8', 'test_10', 'test_12', 'test_15') are checked for missing values. New columns with '_flag' suffixes are added to indicate whether a value is missing or not. Missing values are imputed with the median value of each respective column.
8. **Calculating average steps and filling missing values**: The columns 'steps_day_1', 'steps_day_3', 'steps_day_4', and 'steps_day_5' are used to calculate the average number of steps for each record. Missing values in the 'steps_day_2' column are filled with the calculated average.
9. **Filling missing categorical columns**: Missing values in the categorical columns ('HMO', 'city', 'employment') are filled with the string 'NaN' using the `fillna()` function.
10. **Converting dates and extracting features**: Date columns ('created_at' and 'birth_date') are converted to the datetime format using the `pd.to_datetime()` function. From the 'created_at' column, the year and fractional month values are extracted and stored in the 'created_year' column. From the 'birth_date' column, the age in years and fractional years is calculated relative to the current date. The original date columns are dropped.
11. **Separating the target variable**: The 'label' column is separated from the `train` dataset and stored as the `target` variable. The 'patient_id' column is also dropped from the `train` dataset.
12. **Storing patient IDs for submission**: The 'patient_id' column in the `test` dataset is stored separately as the `test_ids` variable. The 'patient_id' column is dropped from the `test` dataset.
13. **Encoding categorical columns**: Categorical columns ('employment', 'HMO', 'city') in the `train` dataset are encoded using `LabelEncoder` from the `sklearn.preprocessing` module. The encoded values are stored back into the respective columns. The same label encoders are applied to the `test` dataset, with exception handling in case any encoding issues occur.
14. **Printing processing completion message**: A message is printed to indicate that the data preprocessing steps have been completed.
15. **Returning preprocessed data**: The preprocessed `train` and `test` datasets, the `target` variable, the `test_ids` variable, and the list of categorical columns (`cat_cols`) are returned as the output of the `preprocess_data()` function.

Please note that these explanations provide a high-level overview of the preprocessing steps. For more detailed information, please refer to the code comments and the specific preprocessing functions used in the code.
The preprocessed datasets are saved to 'final_procc.csv' for further use.

## Model Pipeline

The solution model employs a pipeline defined in `model.py` that consists of a column transformer for one-hot encoding categorical features and an XGBoost classifier. The training data is split into a training set and a validation set.

## Model Training and Evaluation

The model is trained and evaluated using the `train_and_evaluate()` function from `model.py`. The function returns the trained model pipeline (`model_pipeline`) along with the training and validation ROC AUC scores, validation features, labels, and predictions.
The `train_and_evaluate` function in the code performs the following steps:

1. **Column Transformation and Preprocessing**: It creates a column transformer, `preprocessor`, using `make_column_transformer` from `sklearn.compose`. The column transformer applies one-hot encoding to the categorical columns specified in `categorical_columns` and leaves the remaining columns unchanged.
2. **Model Pipeline Creation**: It creates a pipeline, `model_pipeline`, using `make_pipeline` from `sklearn.pipeline`. The pipeline consists of the `preprocessor` and an `XGBClassifier` from `xgboost`. The XGBoost classifier is configured with specific hyperparameters.
3. **Data Splitting**: It splits the `training_data` into training and validation sets using `train_test_split` from `sklearn.model_selection`. The validation set size is set to 20% of the training data.
4. **Model Fitting**: It fits the `model_pipeline` to the training data and labels using the `fit` method. The fitted model is then ready for making predictions.
5. **Prediction and Evaluation**: It predicts the probabilities for the training and validation sets using the `predict_proba` method. These probabilities are used to calculate the ROC AUC (Area Under the Receiver Operating Characteristic Curve) scores for both the training and validation sets using `roc_auc_score` from `sklearn.metrics`. Additionally, it predicts the labels for the validation set using the `predict` method.
6. **Results Return**: It returns the `model_pipeline`, train and validation AUC scores, the validation features, labels, and predictions.
7. **Printing Messages**: It prints messages to indicate the progress of fitting the model and the successful fitting of the model.

The function encapsulates the training and evaluation process of an XGBoost-based model. It preprocesses the data, creates a pipeline with the XGBoost classifier, fits the model, and returns the necessary information for further analysis and evaluation.

## Model Visualization

The `visualize()` function from `visualize.py` is used to generate visualizations of the model's performance. This function plots the confusion matrix, ROC curve, feature importance, and other relevant visualizations based on the trained model pipeline (`model_pipeline`), validation features, labels, and predictions.
The `visualize` function in the code performs the following visualization steps:

1. **Confusion Matrix Visualization**: It plots the confusion matrix using `ConfusionMatrixDisplay` from `sklearn.metrics`. The confusion matrix is based on the true labels (`validation_labels`) and predicted labels (`validation_predictions`). The plot provides insights into the performance of the model in terms of true positive, true negative, false positive, and false negative predictions.
2. **Decision Tree Visualization**: It plots the first decision tree of the model using `plot_tree` from `xgboost`. The decision tree visualizes the hierarchical structure of the model's decision-making process.
3. **Correlation Matrix Visualization**: It creates a correlation matrix using `corr_matrix` calculated from `data_df.corr()`. The correlation matrix represents the pairwise correlation between different features in the dataset. It is plotted using `sns.heatmap` from `seaborn` to visualize the strength and direction of the correlations.
4. **Feature Importance Visualization**: It creates a horizontal bar plot to visualize the feature importance of the model. The importance of each feature is obtained from the `feature_importances_` attribute of the `xgbclassifier` in the `model_pipeline`. The features and their importance values are sorted and plotted using `plt.barh`. The plot helps identify the most influential features in the model.
5. **Scaling Features**: It performs feature scaling using `StandardScaler` from `sklearn.preprocessing`. The numerical features of `data_df` are scaled using `scaler.fit_transform` and stored in `scaled_features`. Scaling ensures that features are on a similar scale, which can be beneficial for certain machine learning algorithms.
6. **Skewness and Kurtosis Calculation**: It calculates and prints the skewness and kurtosis values for each numerical feature in `data_df`. Skewness measures the asymmetry of the data distribution, while kurtosis measures the tails' thickness. These statistics provide insights into the distribution characteristics of the numerical features.

The function combines different visualization techniques to gain insights into the model's performance, the importance of features, correlations among features, and the distribution of numerical features.

## Prediction and Submission

The `predict_and_save_results()` function from `submission.py` is utilized to make predictions on the test data using the trained model pipeline (`model_pipeline`). The function saves the predicted probabilities, training and validation ROC AUC scores, and the test data along with the corresponding IDs to a submission file (`mysubmission-XGBoost(1).csv`).
The `predict_and_save_results` function in the code performs the following steps:

1. **Making Predictions**: It uses the `model_pipeline` to predict probabilities for the `test_data` using `predict_proba`. The predicted probabilities are stored in `test_predictions_proba`.
2. **Creating Submission DataFrame**: It creates a submission dataframe (`submission_df`) containing the `test_ids` and the predicted probabilities (`test_predictions_proba`).
3. **Saving Submission**: It saves the submission dataframe to a CSV file in the "Submission Files" directory. The file name includes a timestamp to differentiate submissions.
4. **Saving Model**: It saves the trained `model_pipeline` to a pickle file in the "Models" directory. The file name also includes a timestamp.
5. **Calculating Metrics**: It calculates various evaluation metrics based on the `validation_labels` (true labels) and `validation_predictions` (predicted labels). The metrics include accuracy, precision, recall, F1-score, log loss, Matthews correlation coefficient (MCC), balanced accuracy, and the elements of the confusion matrix (true positive, false positive, false negative, true negative).
6. **Logging Metrics**: It creates a metrics dictionary containing the calculated metrics, along with other model parameters and information such as the timestamp, model name, learning rate, number of estimators, maximum depth, minimum child weight, gamma, subsample, colsample_bytree, reg_lambda, reg_alpha, and scale_pos_weight.
7. **Saving Metrics**: It saves the metrics dictionary to a CSV file named "model_metrics.csv". If the file already exists, it appends the new metrics to the existing file. The file contains information about multiple model runs.

Overall, the function predicts the probabilities for the test data, saves the submission, saves the trained model, calculates evaluation metrics, and logs the metrics for further analysis and comparison.

The trained model is also saved in a pickle file named `XGBoost_model(1).pickle` for future use.
