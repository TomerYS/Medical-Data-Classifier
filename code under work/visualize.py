import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

def visualize(pipeline, X_val, y_val, y_val_pred, feature_names):
    # Your visualization code here...
