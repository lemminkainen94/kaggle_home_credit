import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
import scipy

from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from feature_eng import (
    SingleValDropTransformer, SmartCorrDropTransformer, CatFillnaTransformer, NumFillnaTransformer,
    DomainKnowledgeFeatureEng, RareLabelTransformer, CatEncoder
)


load_dotenv()


@pytest.fixture
def test_df():
    return pd.DataFrame({
        'a': pd.Series([1, np.nan, 2, 3, 4]),
        'b': pd.Series([1, np.nan, 2, 2, 3]),
        'employedtotal': pd.Series(['LESS_ONE', 'MORE_THREE', 'MORE_ONE', 'LESS_ONE', 'LESS_ONE'], dtype="category"),
        'target': pd.Series([1, 0, 1, 0, 1]),
        'single': pd.Series([1, 1, 1, 1, 1]),
        'cat': pd.Series(['yes', 'no', 'yes', 'no', 'no'], dtype="category")
    })


@pytest.fixture
def test_fillna_df():
    cft = CatFillnaTransformer()
    nft = NumFillnaTransformer()
    
    cat_cols = list(cft.fill_empty) + list(cft.fill_false)
    zero_cols = ['deduc', 'balanc', 'hisbal', 'credacc_transactions', 'totinstallast1m', 'sumoutstandtotalest_4493215A']
    neg_1_cols = ['num_group1', 'contractssum', 'avglnamtstart24m']
    num_cols = list(nft.null_median) + list(nft.null_mean) + list(nft.null_min) +\
        list(nft.null_max) + list(zero_cols) + list(neg_1_cols)
    
    sample_cat_values = pd.Series(['yes', 'yes', 'yes', np.nan], dtype="category")
    sample_num_values = pd.Series([1, 2, 4, np.nan], dtype=float)

    keys = cat_cols + num_cols
    vals = [sample_cat_values for _ in cat_cols] + [sample_num_values for _ in num_cols]
    return pd.DataFrame(dict(zip(keys, vals)))


def test_single_val_drop_transformer(test_df):
    """check SingleValDropTransformer removes the column with one unique values"""
    svdt = SingleValDropTransformer().fit(test_df)
    pd.testing.assert_frame_equal(svdt.transform(test_df), pd.DataFrame({
        'a': pd.Series([1, np.nan, 2, 3, 4]),
        'b': pd.Series([1, np.nan, 2, 2, 3]),
        'employedtotal': pd.Series(['LESS_ONE', 'MORE_THREE', 'MORE_ONE', 'LESS_ONE', 'LESS_ONE'], dtype="category"),
        'target': pd.Series([1, 0, 1, 0, 1]),
        'cat': pd.Series(['yes', 'no', 'yes', 'no', 'no'], dtype="category")
    }))

    
def test_smart_corr_drop_transformer(test_df):
    """
        check that among two correlated columns, 
        SingleValDropTransformer removes the column with fewer unique values
    """
    scdt = SmartCorrDropTransformer().fit(test_df)
    pd.testing.assert_frame_equal(scdt.transform(test_df), pd.DataFrame({
        'a': pd.Series([1, np.nan, 2, 3, 4]),
        'target': pd.Series([1, 0, 1, 0, 1]),
        'single': pd.Series([1, 1, 1, 1, 1]),
        'employedtotal': pd.Series(['LESS_ONE', 'MORE_THREE', 'MORE_ONE', 'LESS_ONE', 'LESS_ONE'], dtype="category"),
        'cat': pd.Series(['yes', 'no', 'yes', 'no', 'no'], dtype="category")
    }))

def test_cat_fillna_transformer(test_fillna_df):
    cft = CatFillnaTransformer().fit(test_fillna_df)
    cat_cols = list(cft.fill_empty) + list(cft.fill_false)
    cat_vals_empty = pd.Series(pd.Categorical(['yes', 'yes', 'yes', ''], categories=['yes', '']))
    cat_vals_false = pd.Series(pd.Categorical(['yes', 'yes', 'yes', False], categories=['yes', False, '']))
    vals = [cat_vals_empty for x in cft.fill_empty] + [cat_vals_false for x in cft.fill_false]

    pd.testing.assert_frame_equal(cft.transform(test_fillna_df[cat_cols]), pd.DataFrame(
        dict(zip(cat_cols, vals))
    ))

def test_num_fillna_transformer(test_fillna_df):
    nft = NumFillnaTransformer().fit(test_fillna_df)

    zero_cols = ['deduc', 'balanc', 'hisbal', 'credacc_transactions', 'totinstallast1m', 'sumoutstandtotalest_4493215A']
    neg_1_cols = ['num_group1', 'contractssum', 'avglnamtstart24m']
    num_cols = list(nft.null_median) + list(nft.null_mean) + list(nft.null_min) +\
        list(nft.null_max) + list(zero_cols) + list(neg_1_cols)

    num_vals_median = pd.Series([1, 2, 4, 2], dtype=float)
    num_vals_mean = pd.Series([1, 2, 4, np.mean([1,2,4])], dtype=float)
    num_vals_min = pd.Series([1, 2, 4, 1], dtype=float)
    num_vals_max = pd.Series([1, 2, 4, 4], dtype=float)
    num_vals_zero = pd.Series([1, 2, 4, 0], dtype=float)
    num_vals_neg_1 = pd.Series([1, 2, 4, -1], dtype=float)
    vals = [num_vals_median for x in nft.null_median] +\
        [num_vals_mean for x in nft.null_mean] +\
        [num_vals_min for x in nft.null_min] +\
        [num_vals_max for x in nft.null_max] +\
        [num_vals_zero for x in zero_cols] +\
        [num_vals_neg_1 for x in neg_1_cols]

    pd.testing.assert_frame_equal(nft.transform(test_fillna_df[num_cols]), pd.DataFrame(
        dict(zip(num_cols, vals))
    ))

def test_rare_label_encoder(test_df):
    test_df = test_df.fillna({'a': 1, 'b': 1})
    rlt = RareLabelTransformer(tol=0.5).fit(test_df)

    pd.testing.assert_frame_equal(rlt.transform(test_df), pd.DataFrame({
        'a': pd.Series([1, 1, 2, 3, 4], dtype=float),
        'b': pd.Series([1, 1, 2, 2, 3], dtype=float),
        'employedtotal': pd.Series(['LESS_ONE', 'Rare', 'Rare', 'LESS_ONE', 'LESS_ONE'], dtype="category"),
        'target': pd.Series([1, 0, 1, 0, 1]),
        'single': pd.Series([1, 1, 1, 1, 1]),
        'cat': pd.Series(pd.Categorical(['Rare', 'no', 'Rare', 'no', 'no'], categories=['no', 'Rare']))
    }))

def test_cat_encoder(test_df):
    conf = {
        "replace": True,
        "encoders": ['count']
    }

    ce = CatEncoder().fit(test_df, ['employedtotal', 'cat'], conf=conf)

    print(ce.transform(test_df))

    assert True