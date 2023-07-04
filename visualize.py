import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import plot_tree, plot_importance
from scipy.stats import skew, kurtosis
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def visualize(model_pipeline, validation_features, validation_labels, validation_predictions, feature_names, data_df):
    # Get the model parameters
    model = model_pipeline.named_steps['xgbclassifier']
    cm = confusion_matrix(validation_labels, validation_predictions)

    # Confusion Matrix Visualization
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Histograms for each numeric feature
    data_df.hist(figsize=(10, 10), bins=50, xlabelsize=8, ylabelsize=8, color='skyblue', edgecolor='black')
    plt.show()

    # Plot the first decision tree
    plt.close()  # Close any existing figures
    plot_tree(model, rankdir='LR')
    plt.title("First Decision Tree")
    plt.show()


    # Correlation matrix
    corr_matrix = data_df.corr()
    plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k', frameon=False, tight_layout=True, constrained_layout=True)
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, linecolor='black')
    plt.show()

    # Creating barh plot
    importance = model_pipeline['xgbclassifier'].feature_importances_
    feature_importance = dict(zip(feature_names, importance))
    sorted_feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1]))
    plt.figure(figsize=(14, 11), dpi=80, facecolor='w', edgecolor='k')
    plt.barh(range(len(sorted_feature_importance)), list(sorted_feature_importance.values()), color='skyblue')
    plt.yticks(range(len(sorted_feature_importance)), list(sorted_feature_importance.keys()), fontsize=8)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.show()
    print("Feature importance plotted successfully")
