
# Preprocessing imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import numpy as np
import random

np.random.seed(42)
random.seed(42)

def build_pipeline(num_cols,text_col,cat_cols):
    
    """
    Builds a ColumnTransformer for preprocessing numeric, text and categorical data

    Args:
        num_cols(list): List of numerical column names
        text_col(str): Name of text column
        cat_cols(list): List of categorical column names
    
    Returns:
        ColumnTransformer(preprocessor): The complete preprocessing pipeline    
    """
    
    # Pipeline for numeric features
    num_pipe = Pipeline([
        ("imputer",SimpleImputer(strategy="mean")),
        ("scaler",StandardScaler())
     ])
    
    # Pipeline for text feature
    text_pipe = Pipeline([
    ("tfidf",TfidfVectorizer(max_features=1000)),
    ("svd",TruncatedSVD(n_components=100,random_state=42))
    ])

    # Pipeline for categorical feature
    cat_pipe = Pipeline([
        ("imputer",SimpleImputer(strategy="constant",fill_value="missing")),
        ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
    ])
    
    # Combined into a single preprocessor
    preprocessor = ColumnTransformer([
        ("text",text_pipe,text_col),
        ("num",num_pipe,num_cols),
        ("cat",cat_pipe,cat_cols)
    ])
    
    return preprocessor