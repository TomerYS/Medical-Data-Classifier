from data_loading import load_datasets
from preprocessing import preprocess_data
from model import train_and_evaluate
from visualize import visualize
from submission import predict_and_save

def main():
    # Load datasets
    personal_info_train, measurements_train, personal_info_test, measurements_test = load_datasets()

    # Preprocess datasets
    train, test, target, test_ids, label_encoders = preprocess_data(personal_info_train, measurements_train, personal_info_test, measurements_test)

    # Train and evaluate the model
    pipeline, train_score, val_score, X_val, y_val, y_val_pred = train_and_evaluate(train, target)

    # Visualize results
    visualize(pipeline, X_val, y_val, y_val_pred)

    # Predict and save the results
    predict_and_save(pipeline, X_val, y_val, y_val_pred, train_score, val_score, test)

if __name__ == '__main__':
    main()
