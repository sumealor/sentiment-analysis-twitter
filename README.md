# ✈️ Airline Tweet Sentiment Analysis

This project performs **multiclass sentiment classification** on tweets related to airline experiences. It applies natural language processing, feature engineering, and logistic regression to classify tweets as `positive`, `neutral`, or `negative`.

---

## 📂 Project Structure

```
.
├── notebooks/                # Jupyter notebooks
│   └── sen_sis.ipynb         # Main experimentation file
├── src/                      # Python scripts (modular)
│   ├── features/             # Data loading and preprocessing
│   ├── models/               # Model training logic
│   └── utils/                # (Optional) Helper functions
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned/prepared datasets
├── results/
│   ├── figures/              # Plots and evaluation visuals
│   └── reports/              # Output metrics and logs
└── README.md
```

---

## 📊 Dataset

- Source: [Kaggle Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- Total samples: 14,600 tweets
- Labels: `positive`, `neutral`, `negative`
- Features include:
  - Text content
  - Airline
  - Time of tweet
  - User info
  - Location

---

## 🧪 Pipeline Overview

- **Text Preprocessing**: 
  - Cleaned
  - TF-IDF vectorized (max 1000 features)
  - Reduced with TruncatedSVD (100 components)
- **Numerical Features**:
  - Imputed with mean
  - Scaled with StandardScaler
- **Categorical Features**:
  - Imputed with constant
  - Encoded with OneHotEncoder
- **Classifier**:
  - Logistic Regression with `saga` solver and `class_weight='balanced'`
- **Feature Selection**:
  - `SelectFromModel` (L1 penalty) was tested to reduce dimensionality

---

## ⚙️ Hyperparameter Choices

- Solver: `saga`
- Penalty: `l2`
- Class weight: `balanced`
- Max iterations: `1000`
- GridSearch skipped due to compute cost (see note below)

---

## ⚠️ Note on Hyperparameter Tuning

A full `GridSearchCV` over logistic regression hyperparameters was considered but skipped due to excessive runtime (10+ hours with TF-IDF + SVD pipeline). Instead, baseline manual settings were chosen to maintain reasonable training time.

---

## 📈 Results

| Metric         | Value |
|----------------|-------|
| Accuracy       | 0.71  |
| Macro F1 Score | 0.66  |
| Weighted F1    | 0.72  |

See visualizations in [`results/figures/`](../../../../Downloads/results/figures).

---

## 🧠 Key Insights

- `Neutral` and `Positive` classes were underrepresented
- `class_weight="balanced"` improved recall on those classes
- TF-IDF dominated feature set → required dimensionality reduction
- `SelectFromModel` reduced overfitting but increased training time
- Model favored speed over perfect recall due to pipeline constraints

---

## 🐢 Challenges Faced

- TF-IDF + SVD created a large, sparse feature space
- SAGA solver convergence was slow (~25 mins per training loop)
- GridSearch was infeasible due to runtime (10+ hours)
- Feature selection trade-off between accuracy and compute cost

---

## 🔮 Future Work

- Try faster linear models like `LinearSVC` or `SGDClassifier`
- Move `SelectFromModel` inside the pipeline before grid search
- Test other dimensionality reduction (e.g., PCA or UMAP)
- Add error analysis for misclassified neutral tweets
- Visualize top weighted features from logistic regression

---

## 📊 Recommended Metrics & Figures (to include)

Save these in:
```
results/
└── figures/
```

Create these:

- ✅ `confusion_matrix.png`: From `sklearn.metrics.confusion_matrix` + heatmap
- ✅ `roc_curve.png`: ROC AUC curve if using `predict_proba`
- ✅ `class_distribution.png`: Barplot of class counts in the dataset
- ✅ `model_coefficients.png`: Barplot of top logistic regression weights (optional)
- ✅ `f1_scores.png`: Per-class F1 comparison (bar chart)

---

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/airline-sentiment-analysis.git
   cd airline-sentiment-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook
   ```

4. Open `notebooks/sen_sis.ipynb` and run all cells.

---

## 📁 Acknowledgments

- [Kaggle Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- Python, pandas, scikit-learn, matplotlib, seaborn
