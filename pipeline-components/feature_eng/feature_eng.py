import argparse
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
import os
import re

import pandas as pd
import polars as pl
import numpy as np
import sklearn as sk
from feature_engine.encoding import CountFrequencyEncoder, WoEEncoder, RareLabelEncoder, OneHotEncoder
from feature_engine.selection import (DropHighPSIFeatures, SelectByInformationValue, 
    SelectBySingleFeaturePerformance, SelectByTargetMeanPerformance, ProbeFeatureSelection)
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


class SingleValDropTransformer:
    def fit(self, data: pd.DataFrame):
        self.cols_to_drop = []
        for col in data.columns:
            if data[col].nunique() == 1:
                print(col)
                self.cols_to_drop.append(col)
        return self

    def transform(self, data: pd.DataFrame):
        return data.drop(columns=self.cols_to_drop)


class SmartCorrDropTransformer:
    """
        Provides fuinctionalities to drop groups of correlated numerical features;
        for each of the groups keeps the feature with the largest number of unique values
    """
    def __init__(self):
        self.uses = []

    def fit(self, X: pd.DataFrame):
        num_cols_base = list(X.select_dtypes(include=np.number).columns)
        cat_cols_base = list(X.select_dtypes("category").columns)
        nans_df = X[num_cols_base].isna()
        nans_groups = {}
        for col in num_cols_base:
            cur_group = nans_df[col].sum()
            try:
                nans_groups[cur_group].append(col)
            except:
                nans_groups[cur_group] = [col]
    
        for k,v in nans_groups.items():
            if len(v) > 1:
                    Vs = nans_groups[k]
                    grps = self.group_columns_by_correlation(X[Vs], threshold=0.8)
                    use = self.reduce_group(grps, X)
                    self.uses = self.uses+use
            else:
                self.uses=self.uses+v
            print('####### NAN count =',k)
        print(self.uses)
        print(len(self.uses))
        self.uses=self.uses+cat_cols_base
        print(len(self.uses))
        return self
    
    def transform(self, X: pd.DataFrame):
        return X[self.uses]

    def reduce_group(self, grps, X):
        """picks the features with the largest number of unique values from each group"""
        use = []
        for g in grps:
            mx = 0; vx = g[0]
            for gg in g:
                n = X[gg].nunique()
                if n>mx:
                    mx = n
                    vx = gg
            use.append(vx)
        print('Use these',use)
        return use

    def group_columns_by_correlation(self, matrix, threshold=0.8):
        """returns groups of correlated features"""
        correlation_matrix = matrix.corr()
        groups = []
        remaining_cols = list(matrix.columns)
        while remaining_cols:
            col = remaining_cols.pop(0)
            group = [col]
            correlated_cols = [col]
            for c in remaining_cols:
                if correlation_matrix.loc[col, c] >= threshold:
                    group.append(c)
                    correlated_cols.append(c)
            groups.append(group)
            remaining_cols = [c for c in remaining_cols if c not in correlated_cols]
        
        return groups


@dataclass
class CatFillnaTransformer:
    """
        Fills the missing categorical data with values based on 
        data exploration and a tiny bit of domain knowledge
        You can update with your own column name patterns to
        decide what valu to fill with for the given pattern
    """
    fill_empty: tuple = (
        'housetype',
        'bankacctype',
        'credacc_status',
        'relationshiptoclient',
        'remitter',
        'familystate',
        'cardtype',
        'typesuite',
        'empl_industry',
        'sex',
        'contaddr_smempladdr',
        'requesttype',
        'incometype',
        'credtype',
        'inittransactioncode',
        'disbursement',
        'type_25L',
        'role_1084L',
        'maritalst',
        'description',
        'education',
        'opencred',
        'paytype',
        'rejectreason',
        'cancelreason',
        'postype',
        'lastst',
        'twobodfilling',
        'contaddr_matchlist',
        'status_'
    )

    fill_false: tuple = (
        'equality',
        'isdebitcard',
        'safeguaranty',
        'isbidproduct'
    )

    def fit(self, X: pd.DataFrame):
        self.cat_cols_base = list(X.select_dtypes("category").columns)
        return self

    def transform(self, X: pd.DataFrame):
        for pat in self.fill_empty:
            print(pat)
            for col in [x for x in self.cat_cols_base if pat in x]:
                if '' not in X[col].cat.categories:
                    X[col] = X[col].cat.add_categories('')
                X[col] = X[col].fillna('')

        for pat in self.fill_false:
            print(pat)
            for col in [x for x in self.cat_cols_base if pat in x]:
                if False not in X[col].cat.categories:
                    X[col] = X[col].cat.add_categories(False)
                X[col] = X[col].fillna(False)

        for col in self.cat_cols_base:
            if '' not in X[col].cat.categories:
                X[col] = X[col].cat.add_categories('')
            X[col] = X[col].fillna('')

        return X


@dataclass
class NumFillnaTransformer:
    """
        Fills the missing numerical data with values based on 
        data exploration and a tiny bit of domain knowledge
        You can update with your own column name patterns to
        decide what valu to fill with for the given pattern
    """
    null_median: tuple = (
        'openingdate',
        'amount',
        'revolvingaccount',
        'pmtaverage',
        'pmtcount',
        'inittransactionamount_650A',
        'responsedate',
        'byoccupationinc',
        'processingdate',
        'pmtscount',
        'pmtssum_45A',
        'birthdate',
        'pmtnum',
        'numinstpaidlastcontr',
        'personindex',
        'maininc',
        'pctinstlsallpaidlat',
        'cntpmts24',
        'pctinstlsallpaidear',
        'mainoccupationinc',
        'birth',
        'cntincpaycont9m',
        'numincomingpmts',
        'lastactivateddate_801D',
        'posf'
    )

    null_mean: tuple = (
        'employedfrom',
        'maxpmtlast3m_4525190A',
        'maxlnamtstart6m_4525199A',
        'avgpmtlast12m_4525200A',
        'dtlastpmtallstes_4499206D',
        'firstclxcampaign_1125D',
        'avgdbdtollast24m_4525197P',
        'numinstpaid',
        'numinstlsallpaid',
        'numinstregular',
        'numinstpaidearl',
        'outstandingdebt',
        'currdebt',
        'maxinstallast24m',
        'avginstallast24m',
        'amtinstpaidbefduel24m',
        'sumoutstandtotal',
        'downpmt',
        'credacc_credlmt',
        'maxdebt4',
        'price_1097A',
        'eir_270L'
    )

    null_min: tuple = (
        'validfrom',
        'lastdelinqdate',
        'assignmentdate',
        'datelastunpaid',
        'approvaldate',
        'firstnonzeroinstldate',
        'firstdatedue',
        'dpd',
        'lastrejectdate_50D',
        'numinsttopaygr',
        'daysoverduetolerancedd_3976961L',
        'annuity',
        'creationdate',
        'lastapplicationdate',
        'maxdpdinstldate_3546855D'
    )

    null_max: tuple = (
        'datefirstoffer',
        'dateactivated_425D',
        'approvaldate_319D',
        'lastapprdate'
    )

    def fit(self, X: pd.DataFrame):
        self.numeric_cols_base = list(X.select_dtypes(include=np.number).columns)
        self.col_val_dict = {}
        
        self.fit_min(X)
        self.fit_max(X)
        self.fit_mean(X)
        self.fit_median(X)
        self.fit0(X)
        self.fit_neg_1(X)

        #fill the rest with median
        for col in self.numeric_cols_base:
            if col not in self.col_val_dict:
                self.col_val_dict[col] = X[col].median()

        return self

    def transform(self, X: pd.DataFrame):
        X = X.fillna(self.col_val_dict)

        # fill the rest with median value if the column has at least 5 non-null values
        # otherwise just fill with 0
        for col in X.select_dtypes(include=np.number).columns:
            if len(X[X[col].notnull()]) >= 5:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)

        return X

    def fit_min(self, data: pd.DataFrame):
        for pat in self.null_min:
            for col in [x for x in self.numeric_cols_base if pat in x]:
                self.col_val_dict[col] = data[col].min()

    def fit_max(self, data: pd.DataFrame):
        for pat in self.null_max:
            for col in [x for x in self.numeric_cols_base if pat in x]:
                self.col_val_dict[col] = data[col].max()

    def fit_mean(self, data: pd.DataFrame):
        for pat in self.null_mean:
            for col in [x for x in self.numeric_cols_base if pat in x]:
                self.col_val_dict[col] = data[col].mean()

    def fit_median(self, data: pd.DataFrame):
        for pat in self.null_median:
            for col in [x for x in self.numeric_cols_base if pat in x]:
                self.col_val_dict[col] = data[col].median()

    def fit0(self, data: pd.DataFrame):
        cols_to_fill = [
            x for x in data.columns if (
                'deduc' in x or 'balanc' in x or 'hisbal' in x or 'credacc_transactions' in x or 'totinstallast1m' in x
            ) and x in self.numeric_cols_base
        ] + ['sumoutstandtotalest_4493215A']
        for col in cols_to_fill:
            self.col_val_dict[col] = 0

    def fit_neg_1(self, data: pd.DataFrame):
        cols_to_fill = [
            x for x in data.columns if ('num_group1' in x or 'contractssum' in x or 'avglnamtstart24m' in x) and x in self.numeric_cols_base
        ]
        for col in cols_to_fill:
            self.col_val_dict[col] = -1


class DomainKnowledgeFeatureEng:
    """Feature engineering methods based off of other participants' ideas"""
    @staticmethod
    def add_cred_domain_vars(X: pd.DataFrame):
        X['debt_credit_ratio'] = X['currdebt_22A'] / X['credamount_770A']
        X['credit_annuity_ratio'] = X['annuitynextmonth_57A'] / X['credamount_770A']
        X['annuity_to_max_installment_ratio'] = X['maxinstallast24m_3658928A'] / X['annuitynextmonth_57A']
        X['avg_pmt_instl_diff'] = X['avgpmtlast12m_4525200A'] - X['avginstallast24m_3658937A']

    @staticmethod
    def fill_employedtotal(X, cols):
        for col in [x for x in cols if 'employedtotal' in x]:
            X[col] = X[col].apply(lambda x:
                1 if x == 'LESS_ONE' else 2 if x == 'MORE_ONE' else 3
            ).astype(float).fillna(0).astype(int)


class RareLabelTransformer:
    """
        Applies RareLabelEncoder 
        and (TRAIN ONLY) removes rows which have one target class for 'Rare' category after encoding
    """
    def __init__(self, cat_cols_base=None, n_categories=1, tol=0.05):
        self.rle = RareLabelEncoder(n_categories=n_categories, tol=tol, variables=cat_cols_base)
        self.cat_cols_base = cat_cols_base

    def fit(self, X: pd.DataFrame):
        self.rle.fit(X)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series=None):
        X = self.rle.transform(X)

        if not self.cat_cols_base:
            self.cat_cols_base = list(X.select_dtypes("category").columns)

        self.to_rm = []
        if y is not None:
            for col in self.cat_cols_base:
                if y[X[col] == 'Rare'].nunique() == 1:
                    self.to_rm += list(X[X[col] == 'Rare'].index)

        if y is not None:
            X = X.drop(index=self.to_rm)
            y = y.drop(index=self.to_rm)

        for col in self.cat_cols_base:
            X[col] = X[col].cat.remove_unused_categories()

        if y is not None:
            return X, y
        return X


@dataclass
class CatEncoder:
    """
        provides methods to encode categorical variables and decide whether the encodings
        should replace or be added to the dataset;
        available encoders are feature_engine.encoding:
        CountFrequencyEncoder('count'|'frequency'), WoEEncoder, OneHotEncoder
    """
    conf: dict = field(default_factory=lambda: {
        "replace": True,
        "encoders": ["woe", "count"]
    })

    encoder_dict: dict = field(default_factory=lambda: {
        "count": CountFrequencyEncoder('count'),
        "woe": WoEEncoder(),
        "onehot": OneHotEncoder()
    })

    def fit(self, X: pd.DataFrame, cols,  y: pd.Series = None, conf=None):
        if conf:
            self.conf = conf 
        self.encoders = []
        print(y)
        self.cols = cols
        for encoder in self.conf["encoders"]:
            self.encoders.append(self.encoder_dict[encoder].fit(X[self.cols], y))

        return self

    def transform(self, X: pd.DataFrame):
        for i, encoder in enumerate(self.encoders):
            X_enc = encoder.transform(X[self.cols])
            X_enc.columns = [x + '_' + self.conf["encoders"][i] for x in X_enc.columns]
            X = pd.concat([X, X_enc], axis=1)
        
        if self.conf["replace"]:
            return X.drop(columns=self.cols)
        return X


"""
ValueError: During the WoE calculation, some of the categories in the following features contained 0 in the denominator or numerator, and hence the WoE can't be calculated: credtype_322L.

sbs = SelectByShuffling(
        LogisticRegression(),
        cv=3,
        random_state=42,
      )
sbs.fit(X, y)
sbs.get_feature_names_out()


rfa = RecursiveFeatureAddition(RandomForestClassifier(random_state=42), cv=3)
rfa.fit(X, y)
rfa.get_feature_names_out()


psi = DropHighPSIFeatures()
psi.fit(X)
psi.get_feature_names_out()


sfp = SelectBySingleFeaturePerformance(LogisticRegression(), cv=3)
sfp.fit(X, y)
sfp.get_feature_names_out()

"""

def feature_engineering(args):
    df = pd.read_parquet(args.data_path)

    if args.alt_data_path:
        df_alt = pd.read_parquet(args.alt_data_path)
        df = pd.concat((df, df_alt), axis=1)

    #df = df[df['WEEK_NUM'] <= 50]
    #print(df.shape)

    base = df[['case_id', 'WEEK_NUM', 'target']]
    y = df['target']

    try: 
        df = df.drop(columns=['case_id', 'WEEK_NUM', 'target'])
    except:
        print('base columns already cleared!')

    DomainKnowledgeFeatureEng.add_cred_domain_vars(df)
    cat_cols_base = list(df.select_dtypes("category").columns)
    DomainKnowledgeFeatureEng().fill_employedtotal(df, cat_cols_base)

    svdt = SingleValDropTransformer().fit(df)
    df = svdt.transform(df)

    scdt = SmartCorrDropTransformer().fit(df)
    df = scdt.transform(df)

    cft = CatFillnaTransformer().fit(df)
    df = cft.transform(df)

    nft = NumFillnaTransformer().fit(df)
    df = nft.transform(df)

    rlt = RareLabelTransformer().fit(df)
    df, y = rlt.transform(df, y)

    cat_cols_base = list(df.select_dtypes("category").columns)

    ce = CatEncoder().fit(df, cat_cols_base, y)
    df = ce.transform(df)

    print(df.shape)

    df.to_parquet(args.engd_data_path)
    base.drop(index=rlt.to_rm).to_parquet(args.base_engd_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--alt_data_path')
    parser.add_argument('--engd_data_path')
    parser.add_argument('--base_engd_path') # for output only
    args = parser.parse_args()
    feature_engineering(args)
