import os
import pickle
import pandas as pd
import datetime
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, log_loss, confusion_matrix, 
                             matthews_corrcoef, balanced_accuracy_score,
                             confusion_matrix)

def predict_and_save_results(model_pipeline, validation_features, validation_labels, validation_predictions, train_auc_score, validation_auc_score, test_data, test_ids):

    test_predictions_proba = model_pipeline.predict_proba(test_data)[:, 1]
    print('Predictions made successfully')

    # Create a submission dataframe with patient_id and predicted probabilities
    submission_df = pd.DataFrame({
        'patient_id': test_ids,
        'prediction': test_predictions_proba
    })

    # Get the current date and time
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    # Specify the directories
    # Use the timestamp string in the filenames
    # Check if directories exist, if not, create them
    submission_dir = "Submission Files"
    model_dir = "Models"
    submission_file_path = os.path.join(submission_dir, f'mysubmission-XGBoost-{timestamp_str}.csv')
    model_file_path = os.path.join(model_dir, f"XGBoost_model-{timestamp_str}.pickle")
    os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save the submission dataframe to a csv file without row index
    submission_df.to_csv(submission_file_path, index=False)
    print(f'Submission saved to {submission_file_path}')

    # Save model to file
    pickle.dump(model_pipeline, open(model_file_path, "wb"))
    print(f'Model saved to {model_file_path}')

    # Get the model parameters
    model = model_pipeline.named_steps['xgbclassifier']
    cm = confusion_matrix(validation_labels, validation_predictions)

    # Define the metrics and parameters you want to log
    metrics = {
        'Date_Time': [timestamp_str],
        'Model_Name': ['XGBoost'],
        'Train_AUC': [train_auc_score],
        'Validation_AUC': [validation_auc_score],
        'Accuracy': [accuracy_score(validation_labels, validation_predictions)],
        'Precision': [precision_score(validation_labels, validation_predictions)],
        'Recall': [recall_score(validation_labels, validation_predictions)],
        'F1_Score': [f1_score(validation_labels, validation_predictions)],
        'Log_Loss': [log_loss(validation_labels, validation_predictions)],
        'MCC': [matthews_corrcoef(validation_labels, validation_predictions)],
        'Balanced_Accuracy': [balanced_accuracy_score(validation_labels, validation_predictions)],
        'Confusion_Matrix_TP': [cm[1, 1]],  # True positive
        'Confusion_Matrix_FP': [cm[0, 1]],  # False positive
        'Confusion_Matrix_FN': [cm[1, 0]],  # False negative
        'Confusion_Matrix_TN': [cm[0, 0]],  # True negative
        'Model_Filename': [model_file_path],
        'Learning_Rate': [model.learning_rate],
        'N_Estimators': [model.n_estimators],
        'Max_Depth': [model.max_depth],
        'Min_Child_Weight': [model.min_child_weight],
        'Gamma': [model.gamma],
        'Subsample': [model.subsample],
        'Colsample_Bytree': [model.colsample_bytree],
        'Reg_Lambda': [model.reg_lambda],
        'Reg_Alpha': [model.reg_alpha],
        'scale_pos_weight': [model.scale_pos_weight]
    }

    # Convert the dictionary to a DataFrame
    # Define the path of the metrics file
    metrics_df = pd.DataFrame(metrics)
    metrics_file_path = 'model_metrics.csv'
    if os.path.isfile(metrics_file_path):
        existing_metrics_df = pd.read_csv(metrics_file_path)
        combined_metrics_df = pd.concat([existing_metrics_df, metrics_df])
    else:
        combined_metrics_df = metrics_df

    combined_metrics_df.to_csv(metrics_file_path, index=False)
    print(f'Metrics saved to {metrics_file_path}')