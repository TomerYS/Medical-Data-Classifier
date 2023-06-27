import os
import pickle
import pandas as pd
import datetime

def predict_and_save(pipeline, X_val, y_val, y_val_pred, train_score, val_score, test, test_ids):


    test_preds = pipeline.predict_proba(test)[:, 1]

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
