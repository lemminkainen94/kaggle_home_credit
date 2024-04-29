from datetime import datetime
import os
import re

from itertools import combinations
import pandas as pd
import polars as pl
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from scipy.stats import pearsonr, chi2_contingency, ttest_ind


def load_data(data_path="data/train_base.parquet", cat_cols=None) -> pd.DataFrame:
    data = pd.read_parquet(data_path)
    base = data[["case_id", "WEEK_NUM", "target"]]
    data = data[[x for x in data.columns if x not in ["case_id", "WEEK_NUM", "target"]]]
    y = base["target"]
    if cat_cols is None:
        cat_cols = list(data.select_dtypes("object").columns) + list(data.select_dtypes("bool").columns)
    
    data[cat_cols] = data[cat_cols].astype("category")
    return base, data, y

def load_data_week_range(data_path, from_week, to_week, cat_cols=None) -> pd.DataFrame:
    data = pd.read_parquet(data_path)
    data = data[(data["WEEK_NUM"] >= from_week) & (data["WEEK_NUM"] <= to_week)]
    base = data[["case_id", "WEEK_NUM", "target"]]
    data = data[[x for x in data.columns if x not in ["case_id", "WEEK_NUM", "target"]]]
    y = base["target"]
    if cat_cols is None:
        cat_cols = list(data.select_dtypes("object").columns) + list(data.select_dtypes("bool").columns)
    
    data[cat_cols] = data[cat_cols].astype("category")
    return base, data, y


def get_splits(data: pl.DataFrame, mode="train", train_size=0.7):
    case_ids = data[["case_id"]].unique()
    case_ids_train, case_ids_test = train_test_split(case_ids, train_size=train_size)
    if mode == "train":
        case_ids_valid, case_ids_test = train_test_split(case_ids_test, train_size=0.5)
        return case_ids_train, case_ids_valid, case_ids_test
    
    return case_ids_train, case_ids_test


def from_polars_to_pandas(data, case_ids: pl.DataFrame, cat_cols=None) -> pd.DataFrame:
    base = data.filter(pl.col("case_id").is_in(case_ids))[["case_id", "WEEK_NUM", "target"]].to_pandas()
    X = data.filter(pl.col("case_id").is_in(case_ids))[[x for x in data.columns if x not in ["case_id", "WEEK_NUM", "target"]]].to_pandas()
    y = data.filter(pl.col("case_id").is_in(case_ids))["target"].to_pandas()
    if cat_cols is None:
        cat_cols = list(X.select_dtypes("object").columns) + list(X.select_dtypes("bool").columns)
    
    X[cat_cols] = X[cat_cols].astype("category")
    return base, X, y


def load_data_splits(data_path: str, mode="train"):
    data = pl.read_parquet(data_path, use_pyarrow = True)
    if mode == "train":
        case_ids_train, case_ids_valid, case_ids_test = get_splits(data)
        return (
            from_polars_to_pandas(data, case_ids_train),
            from_polars_to_pandas(data, case_ids_valid),
            from_polars_to_pandas(data, case_ids_test)
        )
    case_ids_train, case_ids_test = get_splits(data, mode, train_size=0.85)
    return (
        from_polars_to_pandas(data, case_ids_train),
        from_polars_to_pandas(data, case_ids_test)
    )


def remove_single_val_cols(data, cat_cols_base):
    cols_to_drop = []
    for col in cat_cols_base:
        if data[col].nunique() == 1:
            print(col)
            cols_to_drop.append(col)
    return data.drop(columns=cols_to_drop), cols_to_drop


def fillna_single_val_cols(data, cat_cols_base):
    cols_to_drop = []
    for col in cat_cols_base:
        if data[col].nunique() == 1:
            print(col)
            if 'NA' not in data[col].cat.categories:
                data[col] = data[col].cat.add_categories('NA')
                data.loc[data[col].isnull(), col] = 'NA'


def rare_values_to_others(data: pd.DataFrame, cat_cols_base):
    for col in cat_cols_base:
        for val in data[col].unique():
            count = len(data[data[col] == val])
            if count < 100 and count > 0:
                if 'other' not in data[col].cat.categories:
                    print(col, val)
                    data[col] = data[col].cat.add_categories('other')
                data.loc[data[col] == val, col] = 'other'
                data[col] = data[col].cat.remove_categories(val)


def cat_cols_correlation_check(data, comb_categorical):
    corrs = []
    for comb in comb_categorical:
        table = pd.pivot_table(data, values='target', index=comb[0], columns=comb[1], aggfunc='count').fillna(0)
        # only proceed if the minimum cell count is at least 5, following chi2 recommendations
        if np.min(table) > 4:
            corr = np.sqrt(chi2_contingency(table)[0] / (table.values.sum() * (np.min(table.shape) - 1) ) )
            corrs.append((comb, corr))
        return corrs


def numeric_cols_correlation_check(data, combs, threshold = 0.95):
    corrs = []
    for comb in combs:
        corr = pearsonr(data[comb[0]].fillna(0), data[comb[1]].fillna(0))[0]
        if corr > threshold:
            corrs.append((comb, corr))
    return corrs


def get_corr_cols_to_rm(data, corrs):
    to_rm = []
    for (col1, col2), _ in corrs:
        if len(data[data[col1].isnull()]) > len(data[data[col2].isnull()]):
            to_rm.append(col1)
        else:
            to_rm.append(col2)
    return list(set(to_rm))


def indicate_missing_values(data: pd.DataFrame):
    return pd.concat([data, data.isnull().astype(int).add_suffix('_indicator')], axis=1)

def dummify_categorical(data: pd.DataFrame, cols):
    dummies = pd.get_dummies(data[cols], drop_first=True).astype(int)
    return pd.concat([data.drop(columns=cols), dummies], axis=1)


def fill_min(data: pd.DataFrame, numeric_cols_base):
    data[[x for x in data.columns if 'dpd' in x and x in numeric_cols_base]] = data[[x for x in data.columns if 'dpd' in x and x in numeric_cols_base]].fillna(0)
    data[['datelastinstal40dpd_247D']] = data[['datelastinstal40dpd_247D']].fillna(data['datelastinstal40dpd_247D'].min())
    data[['maxdpdinstldate_3546855D']] = data[['maxdpdinstldate_3546855D']].fillna(data['maxdpdinstldate_3546855D'].min())


def fill0(data: pd.DataFrame, numeric_cols_base):
    cols_to_fill = [x for x in data.columns if ('deduc' in x or 'balanc' in x) and x in numeric_cols_base]
    data[cols_to_fill] = data[cols_to_fill].fillna(0)


def fill_knn(data:pd.DataFrame):
    knn_imputer = KNNImputer(n_neighbors=5)
    imp_df = knn_imputer.fit_transform(data)
    return pd.DataFrame(imp_df, columns=data.columns), knn_imputer

def fill_lr(data: pd.DataFrame):
    imputer = IterativeImputer(n_nearest_features=10, skip_complete=True)
    imp_df = imputer.fit_transform(data)
    return pd.DataFrame(imp_df, columns=data.columns), imputer