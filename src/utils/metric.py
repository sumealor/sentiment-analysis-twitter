
import json
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

def save_basic_metrics(y_true, y_pred, path="../results/reports/metrics.json"):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted")
    }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_classification_report(y_true, y_pred, path="../results/reports/classification_report.txt"):
    report = classification_report(y_true, y_pred)
    with open(path, "w") as f:
        f.write(report)
