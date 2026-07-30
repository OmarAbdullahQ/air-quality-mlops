from __future__ import annotations 
 
from pathlib import Path 
 
import joblib 
import mlflow 
import mlflow.sklearn 
import pandas as pd 
from sklearn.compose import ColumnTransformer 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.impute import SimpleImputer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
 
from air_quality.features import FEATURES 

import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit


PARAM_GRIDS = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__solver": ["lbfgs", "liblinear"],
    },
    "random_forest": {
        "model__n_estimators": [100, 250, 500],
        "model__max_depth": [5, 10, 20, None],
        "model__min_samples_leaf": [1, 5, 20],
    },
    "decision_tree": {
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_leaf": [1, 10, 50],
        "model__criterion": ["gini", "entropy"],
    },
    "knn": {
        "model__n_neighbors": [3, 7, 15, 31],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
}


def run_grid_search() -> None:
    """Print the best hyperparameters per model. Does not save or log anything."""
    df = pd.read_parquet("data/processed/model_table.parquet").drop_duplicates()
    train, valid, _ = split_by_time(df)

    search_df = pd.concat([train, valid], ignore_index=True)
    X = search_df[FEATURES]
    y = search_df["high_pollution_next_hour"]

    # -1 -> always train, 0 -> the one validation fold. Same split as main(), no leakage.
    fold = np.concatenate([np.full(len(train), -1), np.zeros(len(valid), dtype=int)])
    cv = PredefinedSplit(fold)

    base_models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "decision_tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "knn": KNeighborsClassifier(),
    }

    for name, model in base_models.items():
        search = GridSearchCV(
            make_pipeline(model),
            PARAM_GRIDS[name],
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=False,   # we only want the numbers, so skip the final refit
        )
        search.fit(X, y)
        print(f"{name:20s} best_f1={search.best_score_:.4f}  {search.best_params_}")
 
 
def split_by_time(df: pd.DataFrame): 
    n = len(df) 
    train_end = int(n * 0.70) 
    valid_end = int(n * 0.85) 
    return df.iloc[:train_end], df.iloc[train_end:valid_end], df.iloc[valid_end:] 
 
 
def make_pipeline(model): 
    preprocess = ColumnTransformer([ 
        ("numeric", Pipeline([ 
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler()), 
        ]), FEATURES), 
    ]) 
    return Pipeline([("preprocess", preprocess), ("model", model)]) 
 
 
def main() -> None: 
    df = pd.read_parquet("data/processed/model_table.parquet") 
    df = df.drop_duplicates()
    train, valid, test = split_by_time(df) 
    candidates = { 
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1), 
        "random_forest": RandomForestClassifier( 
            n_estimators=250, max_depth=20, class_weight="balanced", random_state=42 ,n_jobs=-1), 
        "decision_tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "knn": KNeighborsClassifier(n_neighbors=31, weights="uniform"),
    } 
 
    mlflow.set_tracking_uri("http://localhost:5000") 
    mlflow.set_experiment("riyadh-air-quality") 
    best_name, best_pipeline, best_f1 = None, None, -1.0 
 
    for name, model in candidates.items(): 
        pipeline = make_pipeline(model) 
        pipeline.fit(train[FEATURES], train["high_pollution_next_hour"]) 
        probability = pipeline.predict_proba(valid[FEATURES])[:, 1] 
        prediction = (probability >= 0.60).astype(int) 
        metrics = { 
            "validation_f1": f1_score(valid["high_pollution_next_hour"], prediction), 
            "validation_recall": recall_score(valid["high_pollution_next_hour"], prediction), 
            "validation_roc_auc": roc_auc_score(valid["high_pollution_next_hour"], probability), 
        } 
        with mlflow.start_run(run_name=name): 
            mlflow.log_params({"model": name, "threshold": 0.60}) 
            mlflow.log_metrics(metrics) 
            # mlflow.sklearn.log_model(pipeline, artifact_path="model") 
            mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")
            

 
        if metrics["validation_f1"] > best_f1: 
            best_name, best_pipeline, best_f1 = name, pipeline, metrics["validation_f1"] 
 
    assert best_pipeline is not None 
    test_probability = best_pipeline.predict_proba(test[FEATURES])[:, 1] 
    test_prediction = (test_probability >= 0.60).astype(int) 
    print({ 
        "selected_model": best_name, 
        "test_f1": f1_score(test["high_pollution_next_hour"], test_prediction), 
        "test_recall": recall_score(test["high_pollution_next_hour"], test_prediction), 
        "test_roc_auc": roc_auc_score(test["high_pollution_next_hour"], test_probability), 
    }) 
    Path("models").mkdir(exist_ok=True) 
    joblib.dump({"pipeline": best_pipeline, "threshold": 0.60}, "models/model.joblib") 
    Path('data/monitoring').mkdir(parents=True, exist_ok=True)
    test[FEATURES + ["high_pollution_next_hour"]].to_csv( 
        "data/monitoring/reference.csv", index=False 
    ) 
 
 
if __name__ == "__main__": 
    run_grid_search()
    main()
