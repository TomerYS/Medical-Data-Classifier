from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd

def train_and_evaluate(train, target):
    # Create a column transformer for preprocessing steps
    preprocessor = make_column_transformer(
        (OneHotEncoder(), cat_cols),
        remainder='passthrough'
    )

    # Create a pipeline with our preprocessor and XGBoost classifier
    pipeline = make_pipeline(
        preprocessor,
        XGBClassifier(n_jobs=-1, learning_rate=0.008, n_estimators=1500, max_depth=14, min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=1.0)
    )

    # Split the training data into training and validation set
    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2)

    pipeline.fit(X_train, y_train)

    # Predict probabilities for the training and validation set
    train_preds = pipeline.predict_proba(X_train)[:, 1]
    val_preds = pipeline.predict_proba(X_val)[:, 1]

    # Calculate the AUC for the training and validation set
    train_score = roc_auc_score(y_train, train_preds)
    val_score = roc_auc_score(y_val, val_preds)
    y_val_pred = pipeline.predict(X_val)


    return pipeline, train_score, val_score, X_val, y_val, y_val_pred
