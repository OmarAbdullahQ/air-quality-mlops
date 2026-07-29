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
 
from air_quality.features import FEATURES 
 
 
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
    train, valid, test = split_by_time(df) 
    candidates = { 
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"), 
        "random_forest": RandomForestClassifier( 
            n_estimators=250, max_depth=10, class_weight="balanced", random_state=42 
        ), 
    } 
 
    mlflow.set_tracking_uri("http://localhost:5000") 
    mlflow.set_experiment("riyadh-air-quality") 
    best_name, best_pipeline, best_f1 = None, None, -1.0 
 
    for name, model in candidates.items(): 
        pipeline = make_pipeline(model) 
        pipeline.fit(train[FEATURES], train["high_pollution_next_hour"]) 
        probability = pipeline.predict_proba(valid[FEATURES])[:, 1] 
        prediction = (probability >= 0.50).astype(int) 
        metrics = { 
            "validation_f1": f1_score(valid["high_pollution_next_hour"], prediction), 
            "validation_recall": recall_score(valid["high_pollution_next_hour"], prediction), 
            "validation_roc_auc": roc_auc_score(valid["high_pollution_next_hour"], probability), 
        } 
        with mlflow.start_run(run_name=name): 
            mlflow.log_params({"model": name, "threshold": 0.50}) 
            mlflow.sklearn.log_model(pipeline, name="model", skops_trusted_types=["numpy.dtype"]) 
            mlflow.log_metrics(metrics) 
 
        if metrics["validation_f1"] > best_f1: 
            best_name, best_pipeline, best_f1 = name, pipeline, metrics["validation_f1"] 
 
    assert best_pipeline is not None 
    test_probability = best_pipeline.predict_proba(test[FEATURES])[:, 1] 
    test_prediction = (test_probability >= 0.50).astype(int) 
    print({ 
        "selected_model": best_name, 
        "test_f1": f1_score(test["high_pollution_next_hour"], test_prediction), 
        "test_recall": recall_score(test["high_pollution_next_hour"], test_prediction), 
        "test_roc_auc": roc_auc_score(test["high_pollution_next_hour"], test_probability), 
    }) 
    Path("models").mkdir(exist_ok=True) 
    joblib.dump({"pipeline": best_pipeline, "threshold": 0.50}, "models/model.joblib") 
    test[FEATURES + ["high_pollution_next_hour"]].to_csv( 
        "data/monitoring/reference.csv", index=False 
    ) 
 
 
if __name__ == "__main__": 
    main()