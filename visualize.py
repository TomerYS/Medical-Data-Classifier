import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

def visualize(pipeline, X_val, y_val, y_val_pred, feature_names):

    train_preds = pipeline.predict_proba(X_train)[:, 1]
    val_preds = pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = pipeline.predict(X_val)
    feature_names = ['gender', 'HMO', 'height', 'bmi', 'heart_rate', 'steps_day_1', 'steps_day_2', 'steps_day_3', 
                    'steps_day_4', 'steps_day_5', 'city', 'employment', 'weight', 'test_0', 'test_1', 'test_2', 
                    'test_3', 'test_4', 'test_5', 'test_6', 'test_7', 'test_8', 'test_9', 'test_10', 'test_11', 
                    'test_12', 'test_13', 'test_14', 'test_15', 'test_16', 'test_17', 'test_18', 'test_19', 
                    'steps_day_2_flag', 'test_2_flag', 'test_6_flag', 'test_8_flag', 'test_10_flag', 'test_12_flag', 
                    'test_15_flag', 'created_year', 'created_month', 'created_day', 'age']
    importance = pipeline['xgbclassifier'].feature_importances_
    feature_importance = dict(zip(feature_names, importance))
    sorted_feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1]))
    plt.figure(figsize=(30, 30))
    plt.barh(range(len(sorted_feature_importance)), list(sorted_feature_importance.values()), color='skyblue')
    plt.yticks(range(len(sorted_feature_importance)), list(sorted_feature_importance.keys()))
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.show()
    print("Feature importance plotted successfully")
    cm = confusion_matrix(y_val, y_val_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    print("Confusion matrix plotted successfully")
    y_val_scores = pipeline.predict_proba(X_val)[:, 1]
    roc_display = RocCurveDisplay.from_estimator(pipeline, X_val, y_val)
    roc_display.plot()
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    print("ROC plotted successfully")
    print("finished displaying graphs")
    xgb.plot_tree(pipeline['xgbclassifier'], num_trees=2, fmap=model_filename)
    plt.title("XGBoost Decision Tree")
    plt.show()

