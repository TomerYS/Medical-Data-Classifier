from data_loading import load_datasets
from preprocessing import preprocess_data
from model import train_and_evaluate
from visualize import visualize
from submission import predict_and_save_results

def main():
    # Load datasets
    personal_info_train, measurements_train, personal_info_test, measurements_test = load_datasets()

    # Preprocess datasets
    training_data, test_data, target_labels, test_ids, categorical_columns = preprocess_data(personal_info_train, measurements_train, personal_info_test, measurements_test)

    # Train and evaluate the model
    model_pipeline, train_auc_score, validation_auc_score, validation_features, validation_labels, validation_predictions = train_and_evaluate(training_data, target_labels, categorical_columns)

    # Predict and save the results
    predict_and_save_results(model_pipeline, validation_features, validation_labels, validation_predictions, train_auc_score, validation_auc_score, test_data, test_ids)

    # Visualize results
    visualize(model_pipeline, validation_features, validation_labels, validation_predictions, training_data.columns, training_data)

if __name__ == '__main__':
    main()


