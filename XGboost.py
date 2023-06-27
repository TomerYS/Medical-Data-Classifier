import pandas as pd
import datetime
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import logging
import pickle
import os
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, log_loss, confusion_matrix, 
                             matthews_corrcoef, balanced_accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)


# Setup logging
logging.basicConfig(filename='XGBoost_Log.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('____________________________________________________________________________________________________________________')
logging.info('XGBoost_model')
logging.info('Description of changes in the model:.....')

personal_info_train = pd.read_csv('personal_info_train.csv')
measurements_train = pd.read_csv('measurements_results_train.csv')
personal_info_test = pd.read_csv('personal_info_test.csv')
measurements_test = pd.read_csv('measurements_results_test.csv')

# Merge datasets on patient_id
train = pd.merge(personal_info_train, measurements_train, on='patient_id')
test = pd.merge(personal_info_test, measurements_test, on='patient_id')

# Identify duplicates in training dataset
duplicateRowsDF = train[train.duplicated(['patient_id'])]

# Check if duplicates exist
if duplicateRowsDF.empty:
    print('No duplicates found.')
else:
    print("Duplicate Rows based on 'patient_id' are:", duplicateRowsDF, sep='\n')

# Remove duplicates
train.drop_duplicates(subset='patient_id', keep='first', inplace=True)

# Convert gender to numeric values using a map
gender_map = {'M': 0, 'F': 1}
train['gender'] = train['gender'].map(gender_map)
test['gender'] = test['gender'].map(gender_map)

# height and weight columns have some values that are too high or too low to be realistic
for df in [train, test]:
    df.loc[df['height'] < 10, 'height'] = df.loc[df['height'] < 10, 'height'] * 100

for df in [train, test]:
    df.loc[df['weight'] > 200, 'weight'] = df.loc[df['weight'] > 200, 'weight'] / 1000

# Calculate BMI where possible
for df in [train, test]:
    mask = df['bmi'] > 80
    df.loc[mask,'bmi'] = df.loc[mask,'weight'] / (df.loc[mask,'height']/100)**2

for df in [train, test]:
    mask = df['bmi'] < 10
    df.loc[mask,'bmi'] = df.loc[mask,'weight'] / (df.loc[mask,'height']/100)**2

# If the BMI is still missing after this calculation, we can fill with the median
imputer = SimpleImputer(strategy='median')
train[['bmi']] = imputer.fit_transform(train[['bmi']])
test[['bmi']] = imputer.transform(test[['bmi']])

# Columns to check for missing values. For each column, create a new '_flag' column and fill missing values with mean
columns_to_check = ['steps_day_2', 'test_2', 'test_6', 'test_8', 'test_10', 'test_12', 'test_15']
for col in columns_to_check:
    train[col + '_flag'] = np.where(train[col].isna(), 0, 1)
    test[col + '_flag'] = np.where(test[col].isna(), 0, 1)

train[columns_to_check] = imputer.fit_transform(train[columns_to_check])
test[columns_to_check] = imputer.transform(test[columns_to_check])

# Drop the 'country' column from train and test dataframes
train.drop(['country'], axis=1, inplace=True)
test.drop(['country'], axis=1, inplace=True)
train.drop(['region'], axis=1, inplace=True)
test.drop(['region'], axis=1, inplace=True)

# Create a SimpleImputer object with strategy as 'most_frequent'. Replace missing values with "NaN"
frequent_imputer = SimpleImputer(strategy='most_frequent')
train['HMO'].fillna('NaN', inplace=True)
test['HMO'].fillna('NaN', inplace=True)
train['city'].fillna('NaN', inplace=True)
test['city'].fillna('NaN', inplace=True)
train['employment'].fillna('NaN', inplace=True)
test['employment'].fillna('NaN', inplace=True)

# Convert dates to datetime format and extract year, month, and day
train['created_at'] = pd.to_datetime(train['created_at'])
train['created_year'] = train['created_at'].dt.year
train['created_month'] = train['created_at'].dt.month
train['created_day'] = train['created_at'].dt.day
train.drop('created_at', axis=1, inplace=True)
train['birth_date'] = pd.to_datetime(train['birth_date'])
test['created_at'] = pd.to_datetime(test['created_at'])
test['created_year'] = test['created_at'].dt.year
test['created_month'] = test['created_at'].dt.month
test['created_day'] = test['created_at'].dt.day
test.drop('created_at', axis=1, inplace=True)
test['birth_date'] = pd.to_datetime(test['birth_date'])

# Create a new feature for age (in years and fraction of year) from the birth_date and drop birth_date
now = pd.to_datetime('today')
train['age'] = (now.year - train['birth_date'].dt.year) + ((now.month - train['birth_date'].dt.month) / 12.0)
train.drop('birth_date', axis=1, inplace=True)
test['age'] = (now.year - test['birth_date'].dt.year) + ((now.month - test['birth_date'].dt.month) / 12.0)
test.drop('birth_date', axis=1, inplace=True)

# Separate target variable 'label' from the training data
target = train['label']
train.drop(['label', 'patient_id'], axis=1, inplace=True)

# Store patient_ids of the test set for final submission, and drop from test set
test_ids = test['patient_id']
test.drop(['patient_id'], axis=1, inplace=True)

# Encode categorical columns using LabelEncoder
cat_cols = ['employment', 'HMO', 'city']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le
    test[col] = le.transform(test[col])

final_pross = pd.concat([train, test], axis=0)
final_pross.to_csv('final_procc.csv', index=False)

# Create a column transformer for preprocessing steps
preprocessor = make_column_transformer(
    (OneHotEncoder(), cat_cols),
    remainder='passthrough'
)

# Create a pipeline with our preprocessor and XGBoost classifier
pipeline = make_pipeline(
    preprocessor,
    XGBClassifier(n_jobs=2, learning_rate=0.008, n_estimators=1500, max_depth=14, min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=1.0)
)
print('Pipeline created successfully')

# Split the training data into training and validation set
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2)
print('Data split successfully')

#for loop to run 10 times
for i in range(10):
    print('Fitting model...')
    pipeline.fit(X_train, y_train)
    print('Model fitted successfully')

    # Predict probabilities for the training and validation set
    print('Making predictions...')
    train_preds = pipeline.predict_proba(X_train)[:, 1]
    val_preds = pipeline.predict_proba(X_val)[:, 1]
    print('Predictions made successfully')

    # Calculate the AUC for the training and validation set
    print('Calculating AUC...')
    train_score = roc_auc_score(y_train, train_preds)
    val_score = roc_auc_score(y_val, val_preds)
    print('AUC calculated successfully')

    # Print AUC scores
    print(f'Train AUC xgboost: {train_score}')
    print(f'Validation AUC: {val_score}')

    # Predict classes for the validation set
    print('Predicting classes...')
    y_val_pred = pipeline.predict(X_val)
    print('Predictions made successfully')

    # Log additional metrics without SMOTE
    logging.info('Metrics for XGBoost:')
    logging.info(f'Train AUC: {train_score}')
    logging.info(f'Validation AUC: {val_score}')
    logging.info(f'Accuracy: {accuracy_score(y_val, y_val_pred)}')
    logging.info(f'Precision: {precision_score(y_val, y_val_pred)}')
    logging.info(f'Recall: {recall_score(y_val, y_val_pred)}')
    logging.info(f'F1-Score: {f1_score(y_val, y_val_pred)}')
    logging.info(f'Log-Loss: {log_loss(y_val, val_preds)}')
    logging.info(f'MCC: {matthews_corrcoef(y_val, y_val_pred)}')
    logging.info(f'Balanced Accuracy: {balanced_accuracy_score(y_val, y_val_pred)}')
    logging.info(f'Confusion Matrix: \n {confusion_matrix(y_val, y_val_pred)}')
    print('Metrics logged successfully')

    feature_names = ['gender', 'HMO', 'height', 'bmi', 'heart_rate', 'steps_day_1', 'steps_day_2', 'steps_day_3', 
                    'steps_day_4', 'steps_day_5', 'city', 'employment', 'weight', 'test_0', 'test_1', 'test_2', 
                    'test_3', 'test_4', 'test_5', 'test_6', 'test_7', 'test_8', 'test_9', 'test_10', 'test_11', 
                    'test_12', 'test_13', 'test_14', 'test_15', 'test_16', 'test_17', 'test_18', 'test_19', 
                    'steps_day_2_flag', 'test_2_flag', 'test_6_flag', 'test_8_flag', 'test_10_flag', 'test_12_flag', 
                    'test_15_flag', 'created_year', 'created_month', 'created_day', 'age']

    # Predict probabilities for the actual test data
    print('Making predictions on test data...')
    test_preds = pipeline.predict_proba(test)[:, 1]
    print('Predictions made successfully')

    importance = pipeline['xgbclassifier'].feature_importances_

    # Creating a dictionary to map feature names with their importance
    feature_importance = dict(zip(feature_names, importance))

    # Sorting the dictionary based on feature importance
    sorted_feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1]))


    # Create a submission dataframe with patient_id and predicted probabilities
    submission = pd.DataFrame({
        'patient_id': test_ids,
        'prediction': test_preds
    })

    # Get the current date and time
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    # Specify the directories
    submission_dir = "Submission Files"
    model_dir = "Models"

    # Use the timestamp string in the filenames
    submission_filename = os.path.join(submission_dir, f'mysubmission-XGBoost-{timestamp_str}.csv')
    model_filename = os.path.join(model_dir, f"XGBoost_model-{timestamp_str}.pickle")

    # Check if directories exist, if not, create them
    os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save the submission dataframe to a csv file without row index
    submission.to_csv(submission_filename, index=False)
    print(f'Submission saved to {submission_filename}')

    # Save model to file
    pickle.dump(pipeline, open(model_filename, "wb"))
    print(f'Model saved to {model_filename}')
    logging.info(f'Model saved to {model_filename}')


    # Get the model parameters
    model = pipeline.named_steps['xgbclassifier']

    # Define the metrics and parameters you want to log
    metrics = {
        'Date_Time': [timestamp_str],
        'Model_Name': ['XGBoost'],
        'Train_AUC': [train_score],
        'Validation_AUC': [val_score],
        'Accuracy': [accuracy_score(y_val, y_val_pred)],
        'Precision': [precision_score(y_val, y_val_pred)],
        'Recall': [recall_score(y_val, y_val_pred)],
        'F1_Score': [f1_score(y_val, y_val_pred)],
        'Log_Loss': [log_loss(y_val, val_preds)],
        'MCC': [matthews_corrcoef(y_val, y_val_pred)],
        'Balanced_Accuracy': [balanced_accuracy_score(y_val, y_val_pred)],
        'Model_Filename': [model_filename],
        'Learning_Rate': [model.learning_rate],
        'N_Estimators': [model.n_estimators],
        'Max_Depth': [model.max_depth],
        'Min_Child_Weight': [model.min_child_weight],
        'Gamma': [model.gamma],
        'Subsample': [model.subsample],
        'Colsample_Bytree': [model.colsample_bytree],
        'Reg_Lambda': [model.reg_lambda],
        'Reg_Alpha': [model.reg_alpha]
    }

    # Convert the dictionary to a DataFrame
    df_metrics = pd.DataFrame(metrics)

    # Define the path of the metrics file
    metrics_file = 'model_metrics.csv'

    # Check if the file exists
    if os.path.isfile(metrics_file):
        # If it exists, load it and append the new data
        df_existing = pd.read_csv(metrics_file)
        df_all = pd.concat([df_existing, df_metrics])
    else:
        # If it does not exist, just use the new data
        df_all = df_metrics

    # Save the DataFrame to a CSV file
    df_all.to_csv(metrics_file, index=False)
    print(f'Metrics saved to {metrics_file}')
    logging.info(f'Metrics saved to {metrics_file}')




# Creating barh plot
plt.figure(figsize=(30, 30))
plt.barh(range(len(sorted_feature_importance)), list(sorted_feature_importance.values()), color='skyblue')
plt.yticks(range(len(sorted_feature_importance)), list(sorted_feature_importance.keys()))
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.show()
print("Feature importance plotted successfully")


# calculate the confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
print("Confusion matrix plotted successfully")

# Use the model to get the target scores:
y_val_scores = pipeline.predict_proba(X_val)[:, 1]
roc_display = RocCurveDisplay.from_estimator(pipeline, X_val, y_val)
roc_display.plot()
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print("ROC plotted successfully")
print("finished displaying graphs")
# Plot the first decision tree
xgb.plot_tree(pipeline['xgbclassifier'], num_trees=2, fmap=model_filename)
plt.title("XGBoost Decision Tree")
plt.show()