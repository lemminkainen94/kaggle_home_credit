import argparse
import os
import gc
from glob import glob
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import re
import sys

import numpy as np
import pandas as pd
import polars as pl


@dataclass
class CFG:
    root_dir: Path = Path("")
    train_dir: Path = Path("data/train")
    test_dir: Path = Path("data/test")

    def __post_init__(self):
        print(self.train_dir)
        self.train_data_paths: dict = {
            "df_base": self.train_dir / "train_base.parquet",
            "depth_0": [
                self.train_dir / "train_static_cb_0.parquet",
                self.train_dir / "train_static_0_*.parquet",
            ],
            "depth_1": [
                self.train_dir / "train_applprev_1_*.parquet",
                self.train_dir / "train_tax_registry_a_1.parquet",
                self.train_dir / "train_tax_registry_b_1.parquet",
                self.train_dir / "train_tax_registry_c_1.parquet",
                self.train_dir / "train_credit_bureau_a_1_*.parquet",
                self.train_dir / "train_credit_bureau_b_1.parquet",
                self.train_dir / "train_other_1.parquet",
                self.train_dir / "train_person_1.parquet",
                self.train_dir / "train_deposit_1.parquet",
                self.train_dir / "train_debitcard_1.parquet",
            ],
            "depth_2": [
                self.train_dir / "train_credit_bureau_b_2.parquet",
                self.train_dir / "train_credit_bureau_a_2_*.parquet",
                self.train_dir / "train_applprev_2.parquet",
                self.train_dir / "train_person_2.parquet"
            ]
        }

        self.test_data_paths: dict = {
            "df_base": self.test_dir / "test_base.parquet",
            "depth_0": [
                self.test_dir / "test_static_cb_0.parquet",
                self.test_dir / "test_static_0_*.parquet",
            ],
            "depth_1": [
                self.test_dir / "test_applprev_1_*.parquet",
                self.test_dir / "test_tax_registry_a_1.parquet",
                self.test_dir / "test_tax_registry_b_1.parquet",
                self.test_dir / "test_tax_registry_c_1.parquet",
                self.test_dir / "test_credit_bureau_a_1_*.parquet",
                self.test_dir / "test_credit_bureau_b_1.parquet",
                self.test_dir / "test_other_1.parquet",
                self.test_dir / "test_person_1.parquet",
                self.test_dir / "test_deposit_1.parquet",
                self.test_dir / "test_debitcard_1.parquet",
            ],
            "depth_2": [
                self.test_dir / "test_credit_bureau_b_2.parquet",
                self.test_dir / "test_credit_bureau_a_2_*.parquet",
                self.test_dir / "test_applprev_2.parquet",
                self.test_dir / "test_person_2.parquet"
            ]
        }


@dataclass
class Aggregator:
    # Please add or subtract features yourself, be aware that too many features will take up too much space.
    exprs: dict = field(default_factory=lambda: {
        "max": lambda col: pl.max(col).alias(f"max_{col}"),
        "min": lambda col: pl.min(col).alias(f"min_{col}"),
        "first": lambda col: pl.first(col).alias(f"first_{col}"),
        "last": lambda col: pl.last(col).alias(f"last_{col}"),
        "mode": lambda col: pl.col(col).drop_nulls().mode().first().alias(f"mode_{col}"),
        "mean": lambda col: pl.mean(col).alias(f"mean_{col}"),
        "median": lambda col: pl.median(col).alias(f"median_{col}"),
        # "var": lambda col: pl.var(col).alias(f"var_{col}"),
        # "range": lambda col: (pl.col(col).max() - pl.col(col).min()).alias(f"range_{col}"),
        # "nunique": lambda col: pl.col(col).n_unique().alias(f"nunique_{col}"),
        # "nuniquetotal": lambda col: (pl.col(col).n_unique() / pl.col(col).count()).alias(f"nuniquetotal_{col}")
    })

    @classmethod
    def num_expr(cls, df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        exprs = ["min", "max", "mean", "range", "first", "last"]
        return  [exprs[expr](col) for col in cols for expr in exprs]

    @classmethod
    def date_expr(cls, df):
        cols = [col for col in df.columns if col[-1] in ("D")]

        exprs = ["min", "max", "mean", "median", "first", "last"]
        return  [exprs[expr](col) for col in cols for expr in exprs]

    @classmethod
    def str_expr(cls, df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        exprs = ["mode", "last"]
        return  [exprs[expr](col) for col in cols for expr in exprs]

    @classmethod
    def other_expr(cls, df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        exprs = ["mode", "last"]
        return  [exprs[expr](col) for col in cols for expr in exprs]

    @classmethod
    def count_expr(cls, df):
        cols = [col for col in df.columns if "num_group" in col]
        exprs = ["max", "last"]
        return  [exprs[expr](col) for col in cols for expr in exprs]

    @classmethod
    def get_exprs(cls, df):
        exprs = cls.num_expr(df) + \
                cls.date_expr(df) + \
                cls.str_expr(df) + \
                cls.other_expr(df) + \
                cls.count_expr(df)

        return exprs


class DataLoader:
    """
        Provides functionalities to load a set of parquet files into a polars or pandas dataframe
    """
    def __init__(self, config: CFG):
        self.config = config

    def load_polars(self, mode="train") -> pl.DataFrame:
        patterns = self.config.train_data_paths if mode == "train" else self.config.test_data_paths
        print(patterns["df_base"])
        df_base = self.read_file(patterns["df_base"]).with_columns(
            month_decision = pl.col("date_decision").dt.month(),
            weekday_decision = pl.col("date_decision").dt.weekday(),
        )
        
        for depth in (0, 1, 2):
            for i, pattern in enumerate(patterns[f"depth_{depth}"]):
                df_base = df_base.join(self.read_files(pattern), how="left", on="case_id", suffix=f"_{depth}_{i}")

        return df_base.pipe(Pipeline.handle_dates)

    def load_pandas(self, mode="train", cat_cols=None):
        df = self.load_polars(mode).to_pandas()
        if cat_cols is None:
            cat_cols = list(df.select_dtypes("object").columns)
        df[cat_cols] = df[cat_cols].astype("category")
        return df.pipe(Pipeline.filter_cols)

    def read_files(self, regex_path):
        chunks = []
        for path in glob(regex_path.__str__()):
            chunks.append(self.read_file(path))
        
        df = pl.concat(chunks, how="vertical_relaxed")
        del chunks
        gc.collect()
        df = df.unique(subset=["case_id"])
        return df

    def read_file(self, path, depth=None):
        print(path)
        df = pl.read_parquet(path).filter(pl.col('case_id') % 4 == 1)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1,2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df)) 
        return df


class Pipeline:
    """
        Dataframe transformations - to be used after loading to set datatypes, 
        reduce memory usage and feature engineer the final dataframe
    """
    @classmethod
    def set_table_dtypes(cls, df: pl.DataFrame) -> pl.DataFrame:
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    @classmethod
    def handle_dates(cls, df: pl.DataFrame) -> pl.DataFrame:
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))  #!!?
                df = df.with_columns(pl.col(col).dt.total_days()) # t - t-1
        df = df.drop("date_decision", "MONTH")
        return df

    @classmethod
    def filter_cols(cls, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].isnull().mean()
                if isnull > 0.95 and col in df.columns:
                    del df[col]
        
        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == "category"):
                freq = df[col].nunique()
                if (freq == 1 or freq > 200) and col in df.columns:
                    del df[col]

        return df

    @classmethod
    def reduce_mem_usage(cls, df: pd.DataFrame) -> pd.DataFrame:
        """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.        
        """
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        
        for col in df.columns:
            col_type = df[col].dtype
            if str(col_type)=="category":
                continue
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                continue
        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        
        return df


def data_proc(args):
    cfg = CFG(train_dir=Path(args.train_dir))
    dl = DataLoader(cfg)
    df_train = dl.load_pandas().pipe(Pipeline.reduce_mem_usage)
    print("train data shape:\t", df_train.shape)
    print(df_train.columns)
    del cfg, dl
    gc.collect()
    df_train[['case_id', 'WEEK_NUM', 'target']].to_parquet(Path(args.train_dir) / "base.parquet", index=False)
    df_train.to_parquet(Path(args.train_dir) / args.output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir')
    parser.add_argument('--output_name')
    args = parser.parse_args()
    data_proc(args)