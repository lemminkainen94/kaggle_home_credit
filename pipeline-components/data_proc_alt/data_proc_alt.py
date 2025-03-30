import argparse
import os, glob
import gc
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import polars as pl
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample, shuffle

from sklearn.preprocessing import (
    OneHotEncoder, PowerTransformer, OrdinalEncoder, TargetEncoder
)



class Utils:
    @staticmethod
    def cols_types(df):
        """
        Create lists of feature names dtype
        """
        date_cols, num_cols, cat_cols = [], [], []
        for col, dtype in df.dtypes.items():
            if pd.api.types.is_bool_dtype(dtype):
                cat_cols.append(col)
            elif pd.api.types.is_datetime64_dtype(dtype):
                date_cols.append(col)
            elif pd.api.types.is_numeric_dtype(dtype):
                num_cols.append(col)
            else:
                cat_cols.append(col)
                
        return date_cols, num_cols, cat_cols

    @staticmethod
    def pl_cols_types(df):
        """
        (Polars version)
        Create lists of feature names dtype
        """
        date_cols, num_cols, cat_cols = [], [], []
        
        num_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        date_cols = df.select(pl.col(pl.Date)).columns
        cat_cols = [col for col in df.columns 
                    if col not in num_cols and col not in date_cols]
                
        return date_cols, num_cols, cat_cols


class DataLoader:
    """Loads the dataset given file paths"""
    @staticmethod
    def aggregate(df, depth=2):
        """
        (Polars version)
        Aggregate depth=1 dataframe and return a depth=0 dataframe 
        """
        # Drop num_group column of appropriate depth
        df = df.drop(f'num_group{depth}')

        groupby_cols = ['case_id']
        if depth > 1:
            groupby_cols = ['case_id', f'num_group{depth-1}']

        date_cols, num_cols, cat_cols = Utils.pl_cols_types(df)
        num_cols = [col for col in num_cols if col not in groupby_cols]
        
        # Create aggregation dataframe
        all_agg = df[groupby_cols]
        all_agg = all_agg.unique(groupby_cols, maintain_order=True)

        # Aggregate categorical columns
        for col in cat_cols:
            if not df[col].is_null().all():
                cat_agg = df.group_by('case_id').agg(
                    pl.col(col).drop_nulls().mode().first().alias(f'{col}_mode')
                )
                # Drop aggregated column to free memory
                df = df.drop(col)

                # Merge with aggregated dataframe
                all_agg = all_agg.join(cat_agg, on='case_id', how='left')

                # Free memory
                del cat_agg

        for col in date_cols:
            date_agg = df.group_by('case_id').agg(
                pl.mean(col).alias(f'{col}_mean')
            )
            # Drop aggregated column to free memory
            df = df.drop(col)

            # Merge with aggregated dataframe
            all_agg = all_agg.join(date_agg, on='case_id', how='left')

            # Free memory
            del date_agg

        for col in num_cols:
            num_agg = df.group_by('case_id').agg(
                pl.mean(col).alias(f'{col}_mean')
            )
            # Drop aggregated column to free memory
            df = df.drop(col)

            # Merge with aggregated dataframe
            all_agg = all_agg.join(num_agg, on='case_id', how='left')

            # Free memory
            del num_agg

        print(f'Depth {depth} aggregation finished')    

        if depth == 1:
            return all_agg

        return DataLoader.aggregate(all_agg, depth-1)

    @staticmethod
    def create_df_from(path, file_name, depth=0):
        """
        (Polars version)
        Preprocess files in chunks 
        """
        dfs = []
        for i, file_path in enumerate(
            glob.glob(str(path / f'*{file_name}*.parquet'))
        ):
            print(file_path)
            df = pl.read_parquet(file_path).filter(pl.col('case_id') % 2 == 1)

            for col in df.columns:
                if (col[-1] == 'D') or (col == 'date_decision'):
                    df = df.with_columns(pl.col(col).cast(pl.Date))
                elif col in ['case_id', 'WEEK_NUM', 'num_group1', 'num_group2']:
                    df = df.with_columns(pl.col(col).cast(pl.Int32))
                elif 'person' in col:
                    df = df.with_columns(
                        pl.col(col).cast(pl.String).cast(pl.Categorical))
                elif 'month' in col and 'T' in col:
                    df = df.with_columns(
                        pl.col(col).cast(pl.String).cast(pl.Categorical))
            
            if depth >= 1:
                df = DataLoader.aggregate(df, depth)
                
            dfs.append(df)
            print(f'Chunk {i} added to list')
        
        return pl.concat(dfs, how='diagonal_relaxed')

    @staticmethod
    def read_prepare_all(path, files_dict):
        """
        (Polars version)
        Read, preprocess and merge all the files together
        Return a pandas dataframe
        """
        # Read base data frame
        path = Path(path)
        df_all = DataLoader.create_df_from(path, 'base')
        print(f'base created')

        # Read and DataLoader.aggregate 
        for depth, files_list in files_dict.items():
            for file in files_list:
                # Create dataframe from file chunks
                print(f'### Start read {file}')
                df = DataLoader.create_df_from(path, file, depth)
                
                # Join with the main dataframe
                df_all = df_all.join(df, how='left', on='case_id')
                print(f'=== {file} merged to df_all')
                
                # Convert to Categorical to free memory
                df_all = df_all.with_columns(
                    pl.col(pl.String).cast(pl.Categorical))
                df_all = df_all.with_columns(
                    pl.col(pl.Float64).cast(pl.Float32))
            
        # Free memory
        del df
        
        # Columns types
        date_cols, num_cols, cat_cols = Utils.pl_cols_types(df_all)
        
        # Convert to pandas in chunks to not explode memory use
        df_pd = df_all.select(pl.col(num_cols)).to_pandas()
        df_all = df_all.drop(num_cols)
        df_pd = df_pd.join(df_all.select(pl.col(date_cols)).to_pandas())
        df_all = df_all.drop(date_cols)
        df_pd = df_pd.join(df_all.select(pl.col(cat_cols)).to_pandas())
        del df_all
        print('df converted to pandas')
        
        # Create time features
        df_pd['birth_year'] = df_pd.birth_259D_mean.dt.year
        df_pd['decision_year'] = df_pd.date_decision.dt.year
        df_pd['decision_quarter'] = (
            df_pd.date_decision.dt.quarter.astype(str).astype('category'))
        df_pd['decision_month_of_year'] = (
            df_pd.date_decision.dt.month.astype(str).astype('category'))
        df_pd['decision_day_of_month'] = df_pd.date_decision.dt.day
        df_pd['decision_day_of_year'] = df_pd.date_decision.dt.dayofyear
        df_pd['decision_week_of_year'] = df_pd.date_decision.dt.isocalendar().week
        df_pd['decision_day_of_week'] = (
            (df_pd.date_decision.dt.dayofweek + 1).astype(str).astype('category'))
        
        return df_pd


class ReduceMemTransformer(BaseEstimator, TransformerMixin):
    """
    Reduce memory usage of a Pandas DataFrame by converting 
    object types to categories and downcasting numeric columns
    """
    def fit(self, df, y=None):
        # Column types
        self.object_cols, self.int_cols, self.float_cols = [], [], []
        for col, dtype in df.dtypes.items():
            if pd.api.types.is_object_dtype(dtype):
                self.object_cols.append(col)
            elif pd.api.types.is_integer_dtype(dtype):
                self.int_cols.append(col)
            elif pd.api.types.is_float_dtype(dtype):
                self.float_cols.append(col)
            
        return self

    def transform(self, df):
        # Convert object columns to category
        df[self.object_cols] = df[self.object_cols].astype('category')

        # Downcast integer columns
        df[self.int_cols] = df[self.int_cols].apply(pd.to_numeric, downcast='integer')
       
        # Downcast float columns
        df[self.float_cols] = df[self.float_cols].apply(pd.to_numeric, downcast='float')
            
        return df

    def get_feature_names_out(self, input_features=None):
        return input_features

    #def fit_transform(self, df, y=None):
    #    self.fit(df, y)
    #    return self.transform(df)


class TimeFeatTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to create time related features
    """
    def fit(self, df, y=None):
        self.ref_cols = ['birth_259D_mean', 'date_decision']
        self.original_cols = [col for col in df.columns 
                              if col not in self.ref_cols]
        return self
    
    def transform(self, df):
        # Create time delta features
        for col in self.original_cols:
            delta_col_0 = f'delta_{col}_{self.ref_cols[0]}'
            df[delta_col_0] = abs(df[col] - df[self.ref_cols[0]]).dt.days

            delta_col_1 = f'delta_{col}_{self.ref_cols[1]}'
            df[delta_col_1] = abs(df[col] - df[self.ref_cols[1]]).dt.days
            
        delta_col_0_1 = f'delta_{self.ref_cols[1]}_{self.ref_cols[0]}'      
        df[delta_col_0_1] = abs(df[self.ref_cols[1]] - df[self.ref_cols[0]]).dt.days
        
        # Drop used cols 
        df = df.drop(self.ref_cols + self.original_cols, axis=1)
        
        self.delta_cols = df.columns.to_list()
        
        return df
   
    def get_feature_names_out(self, input_features=None):
        return self.delta_cols


class NumFeatTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to create numeric related features
    """
    def fit(self, df, y=None):
        self.ref_cols = ['birth_year', 'decision_year']
        self.year_cols = []
        for col in df.columns:
            if 'year' in col and 'T' in col:
                self.year_cols.append(col)
        return self
    
    def transform(self, df):
        # Create year delta features
        for col in self.year_cols:
            delta_col_0 = f'delta_{col}_{self.ref_cols[0]}'
            df[delta_col_0] = abs(df[col] - df[self.ref_cols[0]])

            delta_col_1 = f'delta_{col}_{self.ref_cols[1]}'
            df[delta_col_1] = abs(df[col] - df[self.ref_cols[1]])

        delta_col_0_1 = f'delta_{self.ref_cols[1]}_{self.ref_cols[0]}'   
        df[delta_col_0_1] = abs(df[self.ref_cols[1]] - df[self.ref_cols[0]])
        
        df = df.drop(self.ref_cols + self.year_cols, axis=1)
        
        self.all_cols = df.columns.to_list()

        return df.astype(float)
   
    def get_feature_names_out(self, input_features=None):
        return self.all_cols


class BadColsDropTransformer(BaseEstimator, TransformerMixin): 
    """
    Transformer to drop unuseful columns
    """
    def fit(self, df, y=None):
        # Columns with many missing values 
        self.missing_values = df.isna().mean().sort_values(ascending=False)
        self.cols_to_drop = set(
            self.missing_values[self.missing_values.gt(0.95)].index
        )
        # Columns with one higly dominant value
        for col in df.columns:
            if (df[col].value_counts(normalize=True) > 0.95).any():
                self.cols_to_drop.add(col)
                
        # Columns with identical values  
        for col1 in df.columns[:-1]:
            if col1 not in self.cols_to_drop:
                for col2 in df.columns[df.columns.get_loc(col1) + 1:]:
                    if df[col1].equals(df[col2]):
                        self.cols_to_drop.add(col2)
        return self
        
    def transform(self, df):
        return df.drop(list(self.cols_to_drop), axis=1)
    
    def get_feature_names_out(self, input_features=None):
        return [col for col in input_features 
                if col not in self.cols_to_drop]


class HighCorrDropTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to drop highly correlated numerical columns
    """
    def fit(self, df, y=None):
        self.corr_matrix = df.corr()
        self.cols_to_drop = set()
        for col1 in self.corr_matrix.columns:
            for col2 in self.corr_matrix.columns:
                if col1 != col2:
                    # Check for high correlation
                    if abs(self.corr_matrix.loc[col1, col2]) >= 0.90:
                        # Check which column has more missing values
                        if df[col1].isna().sum() > df[col2].isna().sum():
                            self.cols_to_drop.add(col1)
                        else:
                            self.cols_to_drop.add(col2) 
        return self
    
    def transform(self, df):
        return df.drop(list(self.cols_to_drop), axis=1)
    
    def get_feature_names_out(self, input_features=None):
        return [col for col in input_features if col not in self.cols_to_drop]

       
class LowFreqTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to process categorical, boolean and object columns
    Fill missing and convert infrequent values
    """
    def fit(self, df, y=None):
        self.original_cols = df.columns
        self.frequencies = {}
        self.threshold = {}
        for col in df.columns:
            self.frequencies[col] = df[col].value_counts(normalize=True, 
                                                         ascending=True)
            self.threshold[col] = self.frequencies[col][
                (self.frequencies[col].cumsum() > 0.05).idxmax()
                ]
        return self
    
    def transform(self, df):
        for col in self.original_cols:
            df[col] = df[col].astype(str)
            
            infrequent_mask = (df[col].isin(
                self.frequencies[col].index[
                    self.frequencies[col] < self.threshold[col]
                ]))
            # Convert low frequency categoricals to 'infrequent'
            df.loc[infrequent_mask, col] = 'infrequent'
        return df
    
    def get_feature_names_out(self, input_features=None):
        return input_features


@dataclass
class Pipeline:
    # Pipeline to process date columns
    date_pipeline = make_pipeline(
        TimeFeatTransformer(),
        BadColsDropTransformer(),
        HighCorrDropTransformer(),
        PowerTransformer(copy=False).set_output(transform="pandas"),
        ReduceMemTransformer()
    )
    # Pipeline to process numerical columns
    num_pipeline = make_pipeline(
        NumFeatTransformer(),
        BadColsDropTransformer(),
        HighCorrDropTransformer(),
        PowerTransformer(copy=False).set_output(transform="pandas"),
        ReduceMemTransformer()
    )
    # Pipeline to process low cardinality columns
    low_card_pipeline = make_pipeline(
        BadColsDropTransformer(),
        LowFreqTransformer(),
        OneHotEncoder(
            dtype=np.int8, drop='if_binary', sparse_output=False,
            min_frequency=0.02, handle_unknown='infrequent_if_exist').set_output(transform="pandas"),
        ReduceMemTransformer()
    )
    # Pipeline to process medium cardinality columns
    med_card_pipeline = make_pipeline(
        BadColsDropTransformer(),
        LowFreqTransformer(),
        OrdinalEncoder(handle_unknown='use_encoded_value',
                       unknown_value=np.nan,
                       dtype=np.float32).set_output(transform="pandas"),
        ReduceMemTransformer()
    )
    # Pipeline to process high cardinality columns
    high_card_pipeline = make_pipeline(
        BadColsDropTransformer(),
        LowFreqTransformer(),
        TargetEncoder(target_type='binary', smooth='auto', shuffle=True),
        PowerTransformer(copy=False).set_output(transform="pandas"),
        ReduceMemTransformer()
    )

    def fit(self, X, y):
        # Separate columns by type
        self.date_cols, self.num_cols, self.cat_cols = Utils.cols_types(X)
        cat_unique = X[self.cat_cols].nunique()
        self.low_card_cols = list(cat_unique.index[cat_unique.le(12)])
        self.med_card_cols = list(cat_unique.index[cat_unique.gt(12) & cat_unique.le(200)])
        self.high_card_cols = list(cat_unique.index[cat_unique.gt(200)])


        # Define column transformer
        self.transformer = make_column_transformer(
            (self.date_pipeline, self.date_cols),
            (self.num_pipeline, self.num_cols),
            (self.low_card_pipeline, self.low_card_cols),
            (self.med_card_pipeline, self.med_card_cols),
            (self.high_card_pipeline, self.high_card_cols),
            verbose=True,
        )
     
        self.transformer.fit(X, y)

        return self

    def transform(self, X):
        X = pd.DataFrame(
            self.transformer.transform(X), 
            columns=self.transformer.get_feature_names_out(),
            index=X.index
        )

        enc_med_card_cols = []
        for col in self.med_card_cols:
            if 'pipeline-4__' + col in X.columns:
                enc_med_card_cols.append('pipeline-4__' + col)

        X[enc_med_card_cols] = X[enc_med_card_cols].astype('str').astype('category')

        return X


def data_proc(args):
    # Lists of file names
    files_dict = {
        0: ['static_0', 'static_cb_0'],
        1: ['credit_bureau_a_1', 'credit_bureau_b_1', 'applprev_1', 
            'debitcard_1', 'deposit_1', 'other_1', 'person_1', 
            'tax_registry_a_1', 'tax_registry_b_1', 'tax_registry_c_1'],
        2: ['credit_bureau_a_2', 'credit_bureau_b_2', 'applprev_2', 'person_2']
    }

    data_pipeline = Pipeline()
    X = DataLoader.read_prepare_all(args.train_dir, files_dict)
    y = X['target']

    X[['case_id', 'WEEK_NUM', 'target']].to_parquet(Path(args.train_dir) / "base.parquet", index=False)
    # Drop unuseful and duplicate columns
    cols_to_drop = [
        'birthdate_574D', 'dateofbirth_337D', 'case_id', 'MONTH', 'WEEK_NUM', 'target'
    ]
    X = X.drop(cols_to_drop, axis=1)

    data_pipeline = Pipeline()
    data_pipeline.fit(X, y)
    X = data_pipeline.transform(X)

    print(X.shape)

    X.to_parquet(Path(args.train_dir) / args.output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir')
    parser.add_argument('--output_name')
    args = parser.parse_args()
    data_proc(args)