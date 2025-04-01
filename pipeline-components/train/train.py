import argparse
import os
import joblib
import json
import numpy as np
import lightgbm as lgb
import mlflow
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, train_test_split


LGB_SEARCH_SPACE = {
    'feature_pre_filter': hp.choice('feature_pre_filter', [False]),
    'objective': hp.choice('objective', ['binary']),
    'metric': hp.choice('metric', ['auc']),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 0.9),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 0.9),
    'bagging_freq': hp.choice('bagging_freq', [5]),
    'n_estimators': hp.choice('n_estimators', [1000]),
    'verbose': hp.choice('verbose', [-1]),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.05, 0.1]),
    'num_leaves': hp.randint('num_leaves', 2, 300), 
    'min_child_samples': hp.randint('min_child_samples', 50, 1000), 
    'min_child_weight': hp.loguniform('min_child_weight', -5, 4),
    'subsample': hp.uniform('subsample', 0.4, 1), 
    'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 1),
    'reg_alpha': hp.choice('reg_alpha', [0, 1e-1, 1, 2, 5, 7, 10, 100]),
    'reg_lambda': hp.choice('reg_lambda', [0, 1e-1, 1, 5, 10, 20, 100]),
    'max_depth': hp.choice('max_depth', [2, 3, 4, 5, 7, 11]),
    'device': 'cpu',
    'max_bin': 255,
    'n_jobs': -1,
}

LGB_MODEL_PARAMS = params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 3,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 1000,
    "verbose": -1,
    'device': 'cpu',
    'max_bin': 255,
    'n_jobs': -1
}

"""XGB_SEARCH_SPACE = {
    'n_estimators': 5000,
    'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.1),
    'max_depth': trial.suggest_int('max_depth', 4, 20),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 3000),
    'max_delta_setp': trial.suggest_int('max_delta_setp', 0, 10),
    'subsample': trial.suggest_uniform('subsample', 0.5, 1),
    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1),
    'lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10),
    'alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10),
    'gamma': trial.suggest_loguniform('gamma', 1e-8, 10),
    
    'objective': 'binary:logistic',
#         'eval_metric': 'auc',
    'tree_method': 'gpu_hist',
    'enable_categorical': True,
    'verbosity': 1, 
    'scale_pos_weight': y_ratio,
    'device': 'cuda',
    'n_jobs': -1,
}

CB_SEARCH_SPACE = {
    'iterations': 5000,
    'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.1),
    'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 100),
    'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.0, 10),
    'random_strength': trial.suggest_loguniform('random_strength', 1e-8, 10),
    'depth': trial.suggest_int('depth', 4, 16),
    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
    
    'scale_pos_weight': y_ratio,
    'objective': 'Logloss',
    'eval_metric': 'Logloss', # auc and custom functions are not working on gpu
    'verbose': 50, 
    'task_type': 'GPU',
    'thread_count': -1,
}"""

LGB_TRAIN_PARAMS = {"callbacks": [lgb.log_evaluation(400), lgb.early_stopping(10)]}


class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

def load_model(paths):
    estimators = []
    for model_path in paths:
       estimators.append(joblib.load(model_path))

    return VotingModel(estimators)
 

def train_cv(model_cls, params, df_train, model_path, training_params={}, shuffle=False):
    """
        trains a model with 5-fold, stratified cross-validation;
        this method intends to create a series of models trained on different splits,
        to be later used as an ensemble;
        logs results in the given mlflow experiment as the average roc_auc
        of the 5 models.
        @param model_cls: any model class with a standard scikit interface 
            (has to implement fit with eval_set argument accepting a tuple of dataframes, predict and predict_proba)
        @param params: dict of model params
        @param df_train pd.DataFrame: training dataset to be split by stratified 5-fold cv method. Should have columns 'WEEK_NUM', 'case_id' and 'target'
        @param model_path str: path to save the model to, appended with 5cv_{i} where i - fold number, None - don't write the model
        @param training_params: model training params 
        @param shuffle bool: whether the splits should be shuffled - keep False for reproducibility
    """
    weeks = df_train["WEEK_NUM"]
    y = df_train["target"]
    X = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=shuffle)
    cv_scores = []
    eval_result = {}
    # dataset: PandasDataset = mlflow.data.from_pandas(X.head())

    with mlflow.start_run():
        # mlflow.log_input(dataset, context="training")
        print('input logged')

        mlflow.set_tag("model", str(model_cls))
        mlflow.log_params(params)

        for i, (idx_train, idx_valid) in enumerate(cv.split(X, y, groups=weeks)):
            print(f'fold {i}')
            X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
            X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
            model = model_cls(**params)
            model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], **training_params) # e.g. {"callbacks": [lgb.log_evaluation(400), lgb.early_stopping(10)]}
            if model_path:
                joblib.dump(model, model_path + f'_5cv_{i}.pkl')
                artifact_path = "models"      
                mlflow.log_artifact(local_path=f'{model_path}_5cv_{i}.pkl', artifact_path=artifact_path)

            y_pred_valid = model.predict_proba(X_valid)[:,1]
            auc_score = roc_auc_score(y_valid, y_pred_valid)
            cv_scores.append(auc_score)

        mlflow.log_metric("auc", np.mean(cv_scores))

    return np.mean(cv_scores)


def eval(model, df_eval):
    weeks = df_eval["WEEK_NUM"]
    y = df_eval["target"]
    X = df_eval.drop(columns=["target", "case_id", "WEEK_NUM"])
    # dataset: PandasDataset = mlflow.data.from_pandas(df_eval)

    with mlflow.start_run():
        # mlflow.log_input(dataset, context="eval")

        mlflow.set_tag("model", str(model))

        y_pred = model.predict_proba(X)[:,1]
        auc_score = roc_auc_score(y, y_pred)
        mlflow.log_metric("auc", auc_score)


# should set mlflow experiment to hyperopt in some outer function
def hyperopt_search(train_data, search_space, train_params):
    eval_result = {}
    def objective(params):
        auc = train_cv(lgb.LGBMClassifier, params, train_data, None, train_params)
        return {"loss": -auc, "status": STATUS_OK}

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
    )

    print(best_result)


def hypersearch(args):
    credit_experiment_name = "credit-score-hyperopt"

    mlflow.set_experiment(credit_experiment_name)

    df = pd.read_parquet(args.data_path)
    base = pd.read_parquet(args.base_data_path)

    # train-eval split by week number
    eval_index = base[base.WEEK_NUM >= 81].index
    df = df[~df.index.isin(eval_index)]
    base = base[~base.index.isin(eval_index)]

    if "date_decision" in df.columns:
        df = df.drop("date_decision")
    hyperopt_search(pd.concat([df, base], axis=1), LGB_SEARCH_SPACE, LGB_TRAIN_PARAMS)


def train_and_eval(args):
    credit_experiment_name = "credit-score-train"
    hyperopt_experiment_name = "credit-score-hyperopt"

    mlflow.set_experiment(credit_experiment_name)

    df = pd.read_parquet(args.data_path)
    base = pd.read_parquet(args.base_data_path)

    # train-eval split by week number
    eval_index = base[base.WEEK_NUM >= 81].index
    train_df = df[~df.index.isin(eval_index)]
    eval_df = df[df.index.isin(eval_index)]
    base_train = base[~base.index.isin(eval_index)]
    base_eval = base[base.index.isin(eval_index)]

    # get params of the best model from hyperopt
    experiment = client.get_experiment_by_name(hyperopt_experiment_name)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.auc DESC"],
    )
    params = best_run[0].data.params
    for param in params:
        try:
            if '.' in params[param]:
                params[param] = float(params[param])
            else:
                params[param] = int(params[param])
        except Exception:
            print(param, params[param])

    params['device'] = 'cpu'
    params['max_bin'] = 255
    params['n_jobs'] = -1

    if "date_decision" in df.columns:
        df = df.drop("date_decision")

    print(params)
    train_cv(lgb.LGBMClassifier, params, pd.concat([train_df, base_train], axis=1), args.model_save_path, LGB_TRAIN_PARAMS)

    credit_experiment_name = "credit-score-eval"
    mlflow.set_experiment(credit_experiment_name)

    model = load_model([args.model_save_path + f'_5cv_{i}.pkl' for i in range(5)])
    
    # save the model into oen file for serving
    joblib.dump(model, model_path + f'_5cv_{i}.pkl')
    artifact_path = "models"      
    mlflow.log_artifact(local_path=f'{model_path}_5cv.pkl', artifact_path=artifact_path)

    if "date_decision" in eval_df.columns:
        eval_df = eval_df.drop("date_decision")
    eval(model, pd.concat([eval_df, base_eval], axis=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--base_data_path')
    parser.add_argument('--model_save_path')
    parser.add_argument('--tracking_uri')
    args = parser.parse_args()


    MLFLOW_TRACKING_URI = f"http://mlflowserver.kubeflow:5000"
    if args.tracking_uri:
        MLFLOW_TRACKING_URI = args.tracking_uri

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if args.model_save_path:
        train_and_eval(args)
    else:
        hypersearch(args)
