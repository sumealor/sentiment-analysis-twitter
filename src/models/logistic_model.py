
# Model imports
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import numpy as np
import random

np.random.seed(42)
random.seed(42)

def train_model(X_train,y_train,preprocessor):
    
    """
    Splits the dataset,builds a pipeline with preprocessing and logistic regression model,
    trains the model, and prints the evaluation results/metrics.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target labels.
        preprocessor (ColumnTransformer): Preprocessing pipeline.

    Returns:
        sklearn.pipeline.Pipeline: Trained pipeline model.
    """

    # Build full pipeline
    model_pipeline = Pipeline([
        ("preprocessor",preprocessor),
        ("clf",LogisticRegression(
            penalty="l2",
            solver="saga",
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    # Train the model
    model_pipeline.fit(X_train,y_train)

    return model_pipeline
    