# Supervised Learning — Practical Tutorial

This tutorial covers supervised learning end-to-end: problem framing, data handling, cross-validation, hyperparameter tuning, and class-imbalance strategies. It includes runnable examples in scikit-learn and PySpark ML.

## 1. Overview

Supervised learning trains a model on labeled input→output pairs. Tasks are typically classification (discrete labels) or regression (continuous targets). The typical workflow:
- Problem definition and metrics
- Data collection & exploration
- Preprocessing & feature engineering
- Train/validation/test split (or cross-validation)
- Model selection & training
- Hyperparameter tuning
- Evaluation on holdout test set
- Deployment and monitoring

## 2. Important concepts

- Overfitting vs underfitting — use validation and regularization.
- Bias–variance tradeoff.
- Data leakage — ensure no future information in features.
- Proper metric selection (e.g., recall for rare-disease screening).

## 3. Data splits and cross-validation

- Holdout: train / validation / test (e.g., 70/15/15).
- K-fold CV: rotate train/validation across k folds for robust estimates (e.g., k=5).
- Stratified CV for classification with class imbalance.
- Time-series CV (rolling / expanding) for temporal data.

Example (scikit-learn):

from sklearn.model_selection import train_test_split, StratifiedKFold

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]

## 4. Preprocessing & pipelines

Use pipelines to avoid leakage and make experiments reproducible. Example: scaling, categorical encoding, model.

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('ohe', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols),('cat', cat_transformer, categorical_cols)])

clf = Pipeline(steps=[('preprocessor', preprocessor),('model', RandomForestClassifier(random_state=42))])

## 5. Model selection & evaluation

Choose models appropriate to data size and problem. Always compare to simple baselines. Use cross-validation to estimate generalization.

Metrics:
- Classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC (for imbalanced classes)
- Regression: MAE, RMSE, R²

Example evaluation (scikit-learn):
from sklearn.metrics import classification_report, roc_auc_score
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1]
print(classification_report(y_test, preds))
print('ROC AUC:', roc_auc_score(y_test, probs))

## 6. Hyperparameter tuning

Common approaches:
- GridSearchCV: exhaustive over a grid (good for small search spaces).
- RandomizedSearchCV: random sampling (better for larger spaces).
- Bayesian optimization (Optuna, scikit-optimize): efficient for expensive evaluations.

GridSearchCV example:

from sklearn.model_selection import GridSearchCV
param_grid = {'model__n_estimators': [100, 300], 'model__max_depth': [None, 10, 30]}
gs = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)

Optuna example (simple):

import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 3, 50)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(study.best_params, study.best_value)

Notes:
- Use nested CV if reporting unbiased performance after hyperparameter tuning.
- Use early stopping for iterative models (GBMs, neural nets) and monitor validation metrics.

## 7. Class imbalance strategies

When one class is rare, standard training can bias toward the majority. Strategies:

1. Resampling
- Oversample minority class (random oversampling, SMOTE)
- Undersample majority class

2. Class weights
- Many algorithms accept class_weight='balanced' or explicit weights.

3. Thresholding & calibrated probabilities
- Adjust decision threshold to balance precision/recall.
- Calibrate probabilities (CalibratedClassifierCV).

4. Use appropriate metrics
- Precision-recall curves, PR-AUC, F1, and recall for detection tasks.

scikit-learn example (SMOTE + pipeline):
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

imb_pipeline = ImbPipeline(steps=[('preprocessor', preprocessor),('smote', SMOTE(random_state=42)),('model', RandomForestClassifier(random_state=42))])
imb_pipeline.fit(X_train, y_train)

Example using class weights:
clf = RandomForestClassifier(class_weight='balanced', random_state=42)

## 8. PySpark ML example (classification with cross-validation)

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName('supervised-tutorial').getOrCreate()
df = spark.read.csv('data.csv', header=True, inferSchema=True)  # assume 'label' column

feature_cols = [c for c in df.columns if c != 'label']
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
data = assembler.transform(df).select('features', 'label')

train, test = data.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(featuresCol='features', labelCol='label')
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).addGrid(lr.elasticNetParam, [0.0, 0.5]).build()

evaluator = BinaryClassificationEvaluator(labelCol='label')
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
cvModel = cv.fit(train)
print('Best Params:', cvModel.bestModel.explainParams())

# Class imbalance with weights
# compute class weights and add a weight column to the dataframe
counts = df.groupBy('label').count().collect()
# ... compute inverse-frequency weights and join as 'weight' column
# use weightCol in logistic regression: LogisticRegression(weightCol='weight')

## 9. Model interpretation & debugging

- Feature importance (tree-based models) and SHAP values for detailed explanation.
- Partial dependence plots for marginal effects.
- Error analysis: inspect false positives/negatives.

## 10. Production considerations

- Monitoring: data drift, label drift, performance degradation.
- Retraining strategies: periodic retrain or event‑driven retrain.
- Model packaging: containerize model, expose prediction API, use feature pipelines in production.
- Testing: unit tests for preprocessing, integration tests for full pipeline.

## 11. Checklist before deployment

- Is the metric aligned with business? Are thresholds tuned?
- Is the model robust to data distribution changes?
- Are privacy/compliance constraints satisfied?
- Have you documented data schema, expected feature ranges, and pre/post processing?

---

References & further reading
- scikit-learn documentation: https://scikit-learn.org
- Imbalanced-learn (SMOTE): https://imbalanced-learn.org
- PySpark MLlib: https://spark.apache.org/docs/latest/ml-guide.html
- Optuna: https://optuna.org