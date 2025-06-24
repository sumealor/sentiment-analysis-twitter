
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay,f1_score
    

def plot_confusion_matrix(model, X, y, path="../results/figures/confusion_matrix.png"):
    ConfusionMatrixDisplay.from_estimator(model, X, y, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.close()


def plot_f1_scores(y_true, y_pred, labels, path="../results/figures/f1_scores.png"):
    f1_vals = f1_score(y_true, y_pred, average=None, labels=labels)
    f1_df = pd.DataFrame({"Sentiment": labels, "F1 Score": f1_vals})
    sns.barplot(data=f1_df, x="Sentiment", y="F1 Score", palette="Set2")
    plt.title("F1 Score per Sentiment Class")
    plt.ylim(0, 1)
    plt.savefig(path)
    plt.close()


def plot_class_distribution(y_train, path="../results/figures/class_distribution.png"):
    sns.countplot(x=y_train, palette="pastel")
    plt.title("Training Set Class Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig(path)
    plt.close()

 