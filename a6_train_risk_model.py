#!/usr/bin/env python
# coding: utf-8
from collections import Counter, defaultdict
from itertools import product, groupby
import datetime
import glob
import os
import pickle
import random
import sys
import timeit

import numpy as np
import pandas as pd
from scipy.stats import mode, spearmanr
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, cohen_kappa_score, f1_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline

from myclasses import WeightedKNNImputer, MyPreprocessing, MyCalibrator, MyXGBRanker, MyStagedWrapper, MyLogisticRegression, LTRPairwise

    
def unique_keep_order(x):
    _, idx = np.unique(x, return_index=True)
    return x[np.sort(idx)]
    
    
def get_or_coef_intercept(model):
    support = model.base_estimator.steps[1][-1].get_support()
    coef = np.zeros(len(support))
    if hasattr(model.base_estimator.steps[-1][-1], 'estimator'):
        m = model.base_estimator.steps[-1][-1].estimator
    else:
        m = model.base_estimator.steps[-1][-1]
    coef[support] = m.coef_.flatten()
    intercept = m.intercept_[0]
    return coef, intercept


def get_feature_contrib(model, X):
    if type(model.base_estimator).__name__=='MyStagedWrapper':
        coef = model.base_estimator.get_coef()
        X2 = np.array(X)
        X2[:,-6:][X2[:,-6:]==0] = -1
        fi = np.c_[model.base_estimator.model1.steps[0][-1].transform(X[:,:-6]), X2[:,-6:]]*coef
    elif type(model.base_estimator.steps[-1][-1]).__name__ in ['MyLogisticRegression', 'LTRPairwise']:
        #coef, intercept = get_or_coef_intercept(model)
        coef = model.base_estimator.steps[-1][-1].coef_.flatten()
        intercept = model.base_estimator.steps[-1][-1].intercept_[0]
        fi = model.base_estimator.steps[0][-1].transform(X)*coef
    elif type(model.base_estimator.steps[-1][-1]).__name__=='MyXGBRanker':
        explainer = shap.TreeExplainer(model.base_estimator.steps[-1][-1])
        X2 = model.base_estimator.steps[0][-1].transform(X)
        X2 = model.base_estimator.steps[1][-1].transform(X2)
        shap_values = explainer.shap_values(X2)
        support = model.base_estimator.steps[1][-1].get_support()
        fi = np.zeros((len(X), len(support)))
        fi[:,support] = shap_values
    fi[np.isnan(fi)] = 0
    return fi


def binarize_imputed_value(X, Xnames, isnan):
    cont_names = ['age', 'systolic', 'diastolic', 'temperature', 'HR', 'RespRate', 'spO2', 'cci', 'PXS']
    bin_col_ids = np.where(~np.in1d(Xnames, cont_names))[0]
    X2 = np.array(X)
    for j in bin_col_ids:
        X2[isnan[:,j], j] = np.round(X[isnan[:,j], j])
    return X2


def myimpute_opt(X, Xnames, random_state=None):
    # to select best K and weights,
    # create Xtr_true by artifically creating missing values
    # so that Xtr_imputed is closest to Xtr_true
    np.random.seed(random_state)
    missing = np.isnan(X)
    missing_ratio = np.mean(missing, axis=0)
    Xtr_true = X[~np.any(missing, axis=1)]
    
    # normalize to [0,1]
    maxmin = np.nanmax(Xtr_true, axis=0)-np.nanmin(Xtr_true, axis=0)
    Xtr_true[:, maxmin>0] = (Xtr_true[:, maxmin>0]-np.nanmin(Xtr_true[:, maxmin>0], axis=0))/maxmin[maxmin>0]
    
    Xtr_miss = np.array(Xtr_true)
    
    cxr_cols = np.char.startswith(Xnames, 'cxr')
    # not cxr columns
    for j in np.where(~cxr_cols)[0]:
        ids = np.random.choice(len(Xtr_true), int(round(len(Xtr_true)*missing_ratio[j])), replace=False)
        Xtr_miss[ids, j] = np.nan
    if np.any(cxr_cols):
        # cxr columns, set as a whole, so that missing pattern is the same
        cxr_id = np.where(cxr_cols)[0][0]
        ids = np.random.choice(len(Xtr_true), int(round(len(Xtr_true)*missing_ratio[cxr_id])), replace=False)
        for i in ids:
            Xtr_miss[i, cxr_cols] = np.nan
    missing_tr = np.isnan(Xtr_miss)
    
    Ks = [10,50,100]
    age_weights = [0.1,1,10]
    errors = []
    params = []
    for K, age_w in product(Ks, age_weights):
        #print(K, age_w)
        weights = np.ones(len(Xnames))
        weights[Xnames=='age'] = age_w
        Xtr_imputed = WeightedKNNImputer(n_neighbors=K, feature_weights=weights).fit_transform(Xtr_miss)
        err = np.mean((Xtr_true[missing_tr] - Xtr_imputed[missing_tr])**2)
        errors.append(err)
        params.append((K, age_w))
            
    # after getting the optimum paramters, apply to the whole data
    best_K, best_age_w = params[np.argmin(errors)]
    #best_K = 10
    #best_age_w = 0.1
    print('KNN errors', errors)
    print('KNN imputation K = %d'%best_K)
    print('KNN imputation age weight = %g'%best_age_w)
    weights = np.ones(len(Xnames))
    weights[Xnames=='age'] = best_age_w
    model = WeightedKNNImputer(n_neighbors=best_K, feature_weights=weights).fit(X)
    X2 = model.transform(X)
    
    # binarize binary columns
    X2 = binarize_imputed_value(X2, Xnames, missing)
    
    return X2, model


def stratified_group_k_fold(X, y, groups, K, seed=None):
    """
    https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(K)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(K):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(K):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices
        

def ppv_score(y, yp):
    tp = np.sum((y==1)&(yp==1))
    cp = np.sum(yp==1)
    return tp*1./cp
    
    
def npv_score(y, yp):
    tn = np.sum((y==0)&(yp==0))
    cn = np.sum(yp==0)
    return tn*1./cn

    
def myfit(model_type, X, y_, sids, Xnames, ncv=5, hps=None, random_state=None):#, yordinal
    hps_ = []
    cv_ids = []
    yptes = []
    yp_calib_tes = []
    te_ids = []
    model_cv = []
    cv = GroupKFold(ncv)
    n_jobs = 10
    
    direction_bound = []
    for xname in Xnames:
        if xname.endswith('_hx') or xname.startswith('cxr_') or xname=='ever_positive_upToEvent' or xname=='TobaccoUserDSC' or xname=='cci':
            direction_bound.append((0, None))
        else:
            direction_bound.append((None, None))
    
    if model_type=='logreg':
        y = np.array(y_)
        #y[has_cxr] = (y_[has_cxr]>=2).astype(int)
        #y[~has_cxr] = (y_[~has_cxr]>=1).astype(int)
        y = (y>=2).astype(int)
    else:
        y = y_
        
    allow_missing_ids = np.array([x.startswith('cxr_') or x=='ever_positive_upToEvent' for x in Xnames])
        
    #split data in train set and test set
    # outer CV loop
    for cvi, (train, test) in enumerate(tqdm(stratified_group_k_fold(X, y, sids, K=ncv, seed=random_state), total=ncv)):
        Xtr = X[train]
        Xte = X[test]
        ytr = y[train]
        yte = y[test]
        ytr_ = y_[train]
        sids_tr = sids[train]
        sids_te = sids[test]
        
        if model_type=='logreg':
            Cs = list(np.logspace(2,7,16))
            model = Pipeline(steps=(
                        ('standardizer', MyPreprocessing(allow_missing_ids)),
                        ('model', MyLogisticRegression(
                                    univariate_feature_selection=True,
                                    allow_missing_ids=allow_missing_ids,
                                    class_weight='balanced',
                                    random_state=random_state,
                                    max_iter=10000,
                                    bounds=direction_bound
                                    ))))
            if hps is None:
                model = GridSearchCV(model, {'model__C':Cs},
                                    scoring='f1_weighted', n_jobs=n_jobs, cv=cv)
            else:
                for k in hps:
                    setattr(model.steps[-1][-1], k, hps[k][cvi])

        elif model_type=='or':
            Cs = list(np.logspace(2,7,16))
        
            missing_tr = np.isnan(Xtr)
            model = Pipeline(steps=(
                        ('standardizer', MyPreprocessing(allow_missing_ids)),
                        ('model', LTRPairwise(MyLogisticRegression(
                                        random_state=random_state,
                                        max_iter=1000,
                                        bounds=direction_bound
                                        ),
                                    allow_missing_ids=allow_missing_ids,
                                    missing_val=0,
                                    univariate_feature_selection=True,
                                    class_weight='balanced', min_level_diff=2))))
            if hps is None:
                model = GridSearchCV(model,
                                    {'model__estimator__C':Cs},
                                    scoring='f1_weighted', n_jobs=n_jobs, cv=cv,)
                                    #random_state=random_state)
            else:
                for k in hps:
                    setattr(model.steps[-1][-1].estimator, k, hps[k][cvi])

        elif model_type=='xgboost':
            learning_rates = [0.1,0.2,0.3]
            max_depths = [3,5,6,10]
            reg_lambdas = [0.01,0.1,1]
            model1 = Pipeline(steps=(
                        ('standardizer', StandardScaler()),
                        ('model', MyXGBRanker(
                                    univariate_feature_selection=True,
                                    random_state=random_state,
                                    ))))
            model2 = MyXGBRanker(random_state=random_state,)
            model = MyStagedWrapper(model1, model2, allow_missing_ids)
            if hps is None:
                model = RandomizedSearchCV(model, {'model1__model__learning_rate':learning_rates,
                                             'model1__model__max_depth':max_depths,
                                             'model1__model__reg_lambda':reg_lambdas,
                                             'model2__learning_rate':learning_rates,
                                             'model2__max_depth':max_depths,
                                             'model2__reg_lambda':reg_lambdas},
                                    n_iter=500,
                                    scoring='f1_weighted', n_jobs=n_jobs, cv=cv,
                                    random_state=random_state)
            else:
                for k in hps:
                    setattr(model.steps[-1][-1], k, hps[k][cvi])
        
        if type(model) in [GridSearchCV, RandomizedSearchCV]:
            model.fit(Xtr, y=ytr, groups=sids_tr)
            #print(model.best_params_)
            if hps is None:
                hps_.append(model.best_params_)
                model = model.best_estimator_
        else:
            model.fit(Xtr, y=ytr)#, model__sample_weight=sw)
        ypte = model.predict_proba(Xte)
        yptes.extend(ypte)
        
        # calibration
        model = MyCalibrator(model)
        model.fit(Xtr, ytr_)
        
        model_cv.append(model)
        yp_calib_te = model.predict_proba(Xte)
        yp_calib_tes.extend(yp_calib_te)
        te_ids.extend(test)
        cv_ids.extend([cvi]*len(yte))
    
    # now we do one last model
    if model_type=='logreg':
        if hps is None:
            final_hp = {k:[x[k] for x in hps_] for k in hps_[0]}
        else:
            final_hp = hps
            
        # get hp that is +std, be stringent for the final model
        # approximated by one level stronger
        mode_res = mode(final_hp['model__C'])
        if mode_res.count[0]==1:
            best_C = min(final_hp['model__C'])
        else:
            best_id = Cs.index(mode_res.mode[0])
            best_C = Cs[max(0, best_id-1)]
        print(final_hp, best_C)
        
        model = Pipeline(steps=(
                    ('standardizer', MyPreprocessing(allow_missing_ids)),
                    ('model', MyLogisticRegression(
                                univariate_feature_selection=True,
                                allow_missing_ids=allow_missing_ids,
                                class_weight='balanced',
                                C=best_C,
                                random_state=random_state,
                                max_iter=10000,
                                bounds=direction_bound
                                ))))
    elif model_type=='or':
        if hps is None:
            final_hp = {k:[x[k] for x in hps_] for k in hps_[0]}
        else:
            final_hp = hps
            
        # get hp that is +std, be stringent for the final model
        # approximated by one level stronger
        mode_res = mode(final_hp['model__estimator__C'])
        if mode_res.count[0]==1:
            best_C = min(final_hp['model__estimator__C'])
        else:
            best_id = Cs.index(mode_res.mode[0])
            best_C = Cs[max(0, best_id-1)]
        print(final_hp, best_C)
        
        missing = np.isnan(X)
        model = Pipeline(steps=(
                    ('standardizer', MyPreprocessing(allow_missing_ids)),
                    ('model', LTRPairwise(MyLogisticRegression(
                                    C=best_C,
                                    random_state=random_state,
                                    max_iter=1000,
                                    bounds=direction_bound
                                    ),
                                allow_missing_ids=allow_missing_ids,
                                missing_val=0,
                                univariate_feature_selection=True,
                                class_weight='balanced', min_level_diff=2))))
            
    elif model_type=='xgboost':
        if hps is None:
            final_hp = {k:[x[k] for x in hps_] for k in hps_[0]}
        else:
            final_hp = hps
            
        best_hp = {}
        for k in hps_[0]:
            values = eval(k.split('__')[-1]+'s')
            mode_res = mode(final_hp[k])
            if mode_res.count[0]==1:
                best_hp[k] = min(final_hp[k])
            else:
                best_id = values.index(mode_res.mode[0])
                best_hp[k] = values[best_id]
            
        model1 = Pipeline(steps=(
                    ('standardizer', StandardScaler()),
                    ('model', MyXGBRanker(
                                univariate_feature_selection=True,
                                learning_rate=best_hp['model1__model__learning_rate'],
                                max_depth=best_hp['model1__model__max_depth'],
                                reg_lambda=best_hp['model1__model__reg_lambda'],
                                random_state=random_state,
                                ))))
        model2 = MyXGBRanker(
                            learning_rate=best_hp['model2__learning_rate'],
                            max_depth=best_hp['model2__max_depth'],
                            reg_lambda=best_hp['model2__reg_lambda'],
                            random_state=random_state,
                            )
        model = MyStagedWrapper(model1, model2, allow_missing_ids)
        
    model.fit(X, y)#, model__sample_weight=sw)
    
    # calibration
    model = MyCalibrator(model)
    model.fit(X, y_)
    
    return model, model_cv, final_hp, np.array(yptes), np.array(yp_calib_tes), np.array(te_ids), np.array(cv_ids)
    

def read_Xdata(Xinput_dir, start_date, end_date, return_deleted_columns=False, MRN_range=None):
    # get the list of file names to read up to `date`
    
    Xpaths = glob.glob(os.path.join(Xinput_dir, 'X_Matrix*.csv'))
    Xpaths = [x for x in Xpaths if 'practice' not in x]
    df_X = [pd.read_csv(path) for path in Xpaths]
    df_X = pd.concat(df_X, axis=0).reset_index(drop=True)
    
    assert ~np.any(pd.isna(df_X.MRN))
    cxr_missing = pd.isna(df_X['cxr_0'])
    assert all([np.all(pd.isna(df_X['cxr_%d'%i])==cxr_missing) for i in range(1,55)])
    
    ## limit to MRN_range
    if MRN_range is not None:
        ids = np.in1d(df_X.MRN, MRN_range)
        df_X = df_X[ids].reset_index(drop=True)
    
    ## use up to `date`
    df_X['Date'] = pd.to_datetime(df_X.Date)
    if end_date is not None:
        end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
        df_X = df_X[df_X.Date.dt.date<end_date.date()].reset_index(drop=True)
    if start_date is not None:
        start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
        df_X = df_X[df_X.Date.dt.date>=start_date.date()].reset_index(drop=True)
    
    ## remove automatic midnight checks at 23:59
    ids = ~((df_X.Date.dt.hour==23) & (df_X.Date.dt.minute==59))
    df_X = df_X[ids].reset_index(drop=True)
    
    df_X = df_X.drop_duplicates().reset_index(drop=True)
    
    ## use MGH only
    ids = df_X.MGH_yn
    df_X = df_X[ids].reset_index(drop=True)
    
    ## age>=18 only
    ids = df_X.age>=18
    df_X = df_X[ids].reset_index(drop=True)

    ## make infeasible values to nan, then impute
    # age: [18, 110]
    # systolic: [50,225]
    # diastolic: [25,150]
    # temperature: [94,105]
    # HR: [33,195]
    # RespRate: [8,55]
    # bmi: [9, 80]
    # spO2: [50,100]
    ids = (~pd.isna(df_X.age)) & ( (df_X.age<18) | (df_X.age>110) )
    df_X.loc[ids, 'age'] = np.nan
    ids = (~pd.isna(df_X.systolic)) & ( (df_X.systolic<50) | (df_X.systolic>225) )
    df_X.loc[ids, 'systolic'] = np.nan
    ids = (~pd.isna(df_X.diastolic)) & ( (df_X.diastolic<25) | (df_X.diastolic>150) )
    df_X.loc[ids, 'diastolic'] = np.nan
    ids = (~pd.isna(df_X.temperature)) & ( (df_X.temperature<94) | (df_X.temperature>105) )
    df_X.loc[ids, 'temperature'] = np.nan
    ids = (~pd.isna(df_X.HR)) & ( (df_X.HR<33) | (df_X.HR>195) )
    df_X.loc[ids, 'HR'] = np.nan
    ids = (~pd.isna(df_X.RespRate)) & ( (df_X.RespRate<8) | (df_X.RespRate>55) )
    df_X.loc[ids, 'RespRate'] = np.nan
    ids = (~pd.isna(df_X.bmi)) & ( (df_X.bmi<9) | (df_X.bmi>80) )
    df_X.loc[ids, 'bmi'] = np.nan
    ids = (~pd.isna(df_X.spO2)) & ( (df_X.spO2<50) | (df_X.spO2>100) )
    df_X.loc[ids, 'spO2'] = np.nan
    ids = (~pd.isna(df_X.cci)) & ( (df_X.cci<0) | (df_X.cci>24) )
    df_X.loc[ids, 'cci'] = np.nan
    
    # only keep patients with vitals
    ids = np.in1d(df_X.EventType, ['RIC', 'ED']) & \
            (~pd.isna(df_X.systolic)) & \
            (~pd.isna(df_X.diastolic)) & \
            (~pd.isna(df_X.temperature)) & \
            (~pd.isna(df_X.HR)) & \
            (~pd.isna(df_X.RespRate))
    df_X = df_X[ids].reset_index(drop=True)
    
    # convert tobacco to number
    df_X['TobaccoUserDSC'] = df_X.TobaccoUserDSC.map({
                                        'Never':0,
                                        'Passive':0,
                                        'Quit': 1,
                                        'Yes':1,
                                        'Not Asked':np.nan,})
    
    # convert bmi to bmi_low and bmi_high
    miss_bmi = pd.isna(df_X['bmi'])
    df_X['bmi_low'] = (df_X['bmi']<18.5).astype(float)
    df_X['bmi_high'] = (df_X['bmi']>35).astype(float)
    df_X.loc[miss_bmi, 'bmi_low'] = np.nan
    df_X.loc[miss_bmi, 'bmi_high'] = np.nan
    
    # delete not used columns
    to_delete_columns = ['everCOVIDpos_before', 'cci_age',
                          'first_positive_dt', 'first_positive_testType',
                          'covid_results_1', 'covid_test_dts_1', 'covid_test_types_1',
                          'covid_results_2', 'covid_test_dts_2', 'covid_test_types_2',
                          'covid_results_3', 'covid_test_dts_3', 'covid_test_types_3',
                          'covid_results_4', 'covid_test_dts_4', 'covid_test_types_4',
                          'covid_results_5', 'covid_test_dts_5', 'covid_test_types_5',
                          'systolic_dt', 'diastolic_dt', 'AccessionNBR',
                          'temperature_dt', 'HR_dt', 'RespRate_dt', 'spO2_dt',
                          'BirthDTS', 'ExamStartDTS', 'ZipCD', 'PXS',
                          'Htn', 'Dm', 'Anosmia', 'Dysgeusia',
                          'Asthma', 'Copd', 'Ais', 'Vst', 'Sah',
                          'Ich', 'Sdh', 'Hiv', 'M-acid', 'Pneumothorax',
                          'Aki', 'Ckd-iv', 'Ckd-end', 'Cad', 'Chf',
                          'Rhabdo', 'Myositis', 'Gbs', 'Inf-men-enc',
                          'Ards', 'CIN', 'CIM', 'Hem-cx', 'Renal-cx',
                          'Brain-cx', 'Mg', 'Als', 'Pls', 'PbP', 'Cmt',
                          'Sma', 'Seizure', 'Ow', 'Obesity', 'Gerd', 'Ms',
                          'Coma', 'Sfn', 'Ih', 'Ce', 'Pn', 'Tb', 'Pneumonia',
                          'Pancreatitis', 'Pvd', 'PU', 'Hypothyroidism',
                          'Anxiety', 'Ocd', 'Ar', 'Oa', 'Md', 'Parkinsons',
                          'Movement', 'Neuropathy', 'Neuromuscular', 'Hemiplesia',
                          'Hydrocephalus', 'Colitis', 'Liver', 'Hepatitis',
                          'Mi', 'Pe', 'Pericarditis', 'Myocarditis', 'CM',
                          'Arrest', 'Arrythmia', 'Bronchitis', 'Ild', 'Cf',
                          'Osa', 'Ra', 'Sarcoidosis', 'CTD',
                          'hispanic', 'race_Asian', 'race_Black', 'race_NativeAmerican',
                          'race_Other', 'race_PacificIslander', 'race_Unknown', 'race_White',
                          'bmi']
    df_X_deleted = df_X[to_delete_columns]
    df_X = df_X.drop(columns=to_delete_columns)
    
    if return_deleted_columns:
        return df_X, df_X_deleted
    else:
        return df_X
    
    
def read_Ydata(Yinput_dir, start_date, end_date, horizon):
    Ypaths = glob.glob(os.path.join(Yinput_dir, 'Y_Matrix_*.xlsx'))
    df_y = [pd.read_excel(path) for path in Ypaths]
    df_y = pd.concat(df_y, axis=0).reset_index(drop=True)
    
    ## use up to `date`
    if start_date is not None:
        start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
        df_y = df_y[df_y.Date.dt.date>=start_date.date()].reset_index(drop=True)
    if end_date is not None:
        end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
        df_y = df_y[df_y.Date.dt.date<end_date.date()+datetime.timedelta(days=horizon+1)].reset_index(drop=True)
    
    ## remove automatic midnight checks at 23:59
    ids = ~((df_y.Date.dt.hour==23) & (df_y.Date.dt.minute==59))
    df_y = df_y[ids].reset_index(drop=True)
    
    ## remove following invalid events
    """
    Hosp	MGH UCC CHELSEA	
    Hosp	MGH PERIOPERATIVE DEPT	
    Hosp	MGH IMG XR ER MG WHT1	
    Hosp	MGH IMG OBUS YAW4	
    Hosp	MGP CARD EP DEVICE YAW5B	
    Hosp	MGP IMG MR OUTSIDE
    """
    df_y['EventType'] = df_y.EventType.str.strip().str.upper()
    df_y['Department'] = df_y.Department.str.strip().str.upper()
    ids = ~((df_y.EventType=='HOSP') & np.in1d(df_y.Department, [
                                        'MGH UCC CHELSEA',
                                        'MGH PERIOPERATIVE DEPT',
                                        'MGH IMG XR ER MG WHT1',
                                        'MGH IMG OBUS YAW4',
                                        'MGP CARD EP DEVICE YAW5B',
                                        'MGP IMG MR OUTSIDE',]))
    df_y = df_y[ids].reset_index(drop=True)
    
    ## keep date only and remove duplicates
    #df_y['DateOnly']=df_y.Date.dt.date
    df_y = df_y[['MRN', 'Date','EventType','Department']]
    df_y = df_y.drop_duplicates().reset_index(drop=True)
    #df_y = df_y.rename(columns={'DateOnly':'Date'})
    
    ## use MGH only
    ids = df_y.Department.astype(str).str.contains('MGH|MGP')
    ids |= (pd.isna(df_y.Department) & (df_y.EventType=='DEATH')) # death has no department
    df_y = df_y[ids].reset_index(drop=True)
    
    #with open(os.path.join('/data/Dropbox (Partners HealthCare)/RISK_PREDICTIONS_SHARED/20200515/Xy_prospective.pickle'), 'rb') as ff: 
    #    _, df_y_pros, _, _, _ = pickle.load(ff)
    #mrns_pros = list(df_y_pros.MRN)
    
    mrns = pd.unique(df_y.MRN)
    res = []
    for mrn in tqdm(mrns):
        df_pt = df_y[df_y.MRN==mrn].sort_values('Date').reset_index(drop=True)
        ed_ric_ids = np.where(np.in1d(df_pt.EventType, ['ED', 'RIC']))[0]
        if len(ed_ric_ids)==0:
            continue
        hosp_ids = np.where(df_pt.EventType=='HOSP')[0]
        icu_ids = np.where(df_pt.EventType=='ICU')[0]
        intub_ids = np.where(df_pt.EventType=='INTUBATION')[0]
        death_ids = np.where(df_pt.EventType=='DEATH')[0]
        cc = 0
        for k, l in groupby(df_pt.EventType):
            ll = len(list(l))
            if k in ['ED', 'RIC']:
                #if mrn in mrns_pros:
                #    ed_ric_date = df_y_pros.Date[df_y_pros.MRN==mrn].iloc[0]
                #    eventtype = df_y_pros.EventType[df_y_pros.MRN==mrn].iloc[0]
                #    dept = df_y_pros.Department[df_y_pros.MRN==mrn].iloc[0]
                #else:
                ed_ric_date = df_pt.Date.iloc[cc+ll-1]
                eventtype = df_pt.EventType.iloc[cc+ll-1]
                dept = df_pt.Department.iloc[cc+ll-1]
                
                status_hosp = -1
                for jj in hosp_ids:
                    if df_pt.Date.iloc[jj]>=ed_ric_date:
                        status_hosp = (df_pt.Date.iloc[jj]-ed_ric_date).total_seconds()/86400
                        break
                
                status_icu = -1
                for jj in icu_ids:
                    if df_pt.Date.iloc[jj]>=ed_ric_date:
                        status_icu = (df_pt.Date.iloc[jj]-ed_ric_date).total_seconds()/86400
                        break
                
                status_intub = -1
                for jj in intub_ids:
                    if df_pt.Date.iloc[jj]>=ed_ric_date:
                        status_intub = (df_pt.Date.iloc[jj]-ed_ric_date).total_seconds()/86400
                        break
                
                status_death = -1
                for jj in death_ids:
                    if df_pt.Date.iloc[jj]>=ed_ric_date:
                        status_death = (df_pt.Date.iloc[jj]-ed_ric_date).total_seconds()/86400
                        break
                        
                res.append([mrn, ed_ric_date, eventtype, dept,
                            status_hosp, status_icu, status_intub, status_death])
            cc += ll
    df_y = pd.DataFrame(data=np.array(res, dtype=object),
                        columns=['MRN', 'Date', 'EventType', 'Department',
                        'PStatus(Hosp)', 'PStatus(ICU)', 'PStatus(Intub)', 'PStatus(Death)'])
    
    assert ~np.any(pd.isna(df_y.MRN))
    return df_y
    
    
def align_X_Y(df_X, df_y, horizon, for_training=False):
    df_y['key'] = df_y.MRN.astype(str)+df_y.Date.astype(str)+df_y.EventType.astype(str)+df_y.Department.astype(str)
    df_X['key'] = df_X.MRN.astype(str)+df_X.Date.astype(str)+df_X.EventType.astype(str)+df_X.Department.astype(str)
    unique_mrns = unique_keep_order(df_y.MRN.values)
    ids_X = []
    ids_y = []
    y = []
    ycols = ['Hosp', 'ICU', 'Intub', 'Death']
    ylevels = [1,    2,     3,        4]
    for mrn in tqdm(unique_mrns):
        ids = np.where(df_y.MRN==mrn)[0]
        for id_ in ids:
            id__ = np.where(df_X.key==df_y.key.iloc[id_])[0]
            if len(id__)>1:
                raise ValueError('Found %d matches when aligning X and Y'%len(id_X))
            elif len(id__)==1:
                ids_X.append(id__[0])
                ids_y.append(id_)
                
                # convert y value to 1/0 based on horizon
                y_ = []
                for yi, ycol in enumerate(ycols):
                    if 0<=df_y['PStatus('+ycol+')'].iloc[id_]<=horizon:
                        y_.append(ylevels[yi])
                    else:
                        y_.append(0)
                y.append(max(y_))
    ids_X = np.array(ids_X)
    ids_y = np.array(ids_y)
    df_X = df_X.iloc[ids_X].reset_index(drop=True)
    df_y = df_y.iloc[ids_y].reset_index(drop=True)
    y = np.array(y)
    
    # combine ICU and Intub
    y[y==3] = 2
    y[y==4] = 3
    
    df_y['y_ordinal'] = y
    
    # delete not number columns
    df_X = df_X.drop(columns=['key'])
    df_y = df_y.drop(columns=['key'])
        
    if for_training:
        # randomly select one event from each patient
        unique_mrns = unique_keep_order(df_X.MRN.values)
        unique_ids = []
        for mrn in unique_mrns:
            ids = np.where(df_X.MRN==mrn)[0]
            if len(ids)==1:
                unique_ids.append(ids[0])
            else:
                id_ = np.random.choice(ids, 1)
                unique_ids.append(id_[0])
        df_X = df_X.iloc[unique_ids].reset_index(drop=True)
        df_y = df_y.iloc[unique_ids].reset_index(drop=True)
        
        ids_X = ids_X[unique_ids]
        ids_y = ids_y[unique_ids]
        
    return df_X, df_y, ids_X, ids_y
    
    
def group_cxr(df):
    cxr_patterns = ['patchy consolidation', 'viral pneumonia', 'multifocal pneumonia',
       'multifocal viral pneumonia', 'consistent with pneumonia',
       'multifocal patchy airspace opacities', 'hazy opacities',
       'airspace opacities', 'peripheral opacities', 'peripheral opacity',
       'patchy opacity', 'confluent airspace opacities', 'confluent opacities',
       'likely pneumonia', 'diffuse opacities', 'faint interstitial opacities',
       'new patchy airspace opacities',
       '(multifocal)(\W+(?:\w+\W+){0,10}?)(opacit)',
       '(patchy opacities|patchy opacities)', 'covid 19 pneumonia',
       'patchy pneumonia', 'lobe consolidation', 'typical pattern for covid',
       'consistent with covid pneumonia', 'could be due to covid',
       'can t exclude consolidation', 'consolidative opacities',
       'concerning for infection', 'likely representing pneumonia',
       'bilateral lung opacities', 'diffuse ground glass', 'ground glass',
       'ground glass opacity', 'ground glass opacities', 'ggo', 'halo',
       'reverse halo', 'cant exclude consolidation', 'pleural effusion',
       'pulmonary edema', 'focal infiltrate', 'infiltrate', 'consolidation',
       '(consolidation)(\W+(?:\w+\W+){0,10}?)(not excluded)',
       '(infection)(\W+(?:\w+\W+){0,10}?)(not excluded)',
       '(airspace)(\W+(?:\w+\W+){0,10}?)(opacities)',
       '(seen in)(\W+(?:\w+\W+){0,5}?)(covid)',
       '(groundglass)(\W+(?:\w+\W+){0,10}?)(opacities|opacity)',
       '(concerning for)(\W+(?:\w+\W+){0,10}?)(covid)',
       '(consolidative|patchy)(\W+(?:\w+\W+){0,10}?)(opacities|opacity)',
       '(compatible with|may represent|suggesting|consistent with|in keeping with)(\W+(?:\w+\W+){0,10}?)(pneumonia)',
       '(compatible with|may represent|suggesting|consistent with|in keeping with)(\W+(?:\w+\W+){0,10}?)(bronchopneumonia)',
       '(compatible with|may represent|suggesting|consistent with|in keeping with)(\W+(?:\w+\W+){0,10}?)(viral pneumonia)',
       '(compatible with|may represent|suggesting|consistent with|in keeping with)(\W+(?:\w+\W+){0,10}?)(ards)',
       '(compatible with|may represent|suggesting|consistent with|in keeping with)(\W+(?:\w+\W+){0,10}?)(covid)']
    assert len(cxr_patterns)==55
    
    groups = {
        'Multifocal':
           ['diffuse opacities',
            'bilateral lung opacities',
            'diffuse ground glass',#-------> ground glass?
            'multifocal viral pneumonia',
            'multifocal pneumonia',
            'multifocal patchy airspace opacities',
            '(multifocal)(\W+(?:\w+\W+){0,10}?)(opacit)',
            'viral pneumonia',
            '(compatible with|may represent|suggesting|consistent with|in keeping with)(\W+(?:\w+\W+){0,10}?)(viral pneumonia)',
            'patchy opacity',
            '(patchy opacities|patchy opacities)',
            'new patchy airspace opacities',
            'patchy pneumonia',
            'patchy consolidation',
            '(consolidative|patchy)(\W+(?:\w+\W+){0,10}?)(opacities|opacity)',#],
            #'GroundGlass':   # GroundGlass is combined with Multifocal
            #[
           'ground glass',
            'ground glass opacity',
            'ground glass opacities',
            '(groundglass)(\W+(?:\w+\W+){0,10}?)(opacities|opacity)',
            'ggo',],
        #'PulmonaryEdema':  # separate group
        #   ['pulmonary edema',],
        'TypicalPatternForCovid':
           ['typical pattern for covid',
            'consistent with covid pneumonia',
            'covid 19 pneumonia',
            'could be due to covid',
            '(seen in)(\W+(?:\w+\W+){0,5}?)(covid)',
            '(concerning for)(\W+(?:\w+\W+){0,10}?)(covid)',
            '(compatible with|may represent|suggesting|consistent with|in keeping with)(\W+(?:\w+\W+){0,10}?)(covid)',#],
            #'Pneumonia':
            #[
            'consistent with pneumonia',
            'likely representing pneumonia',
            'likely pneumonia',
            'concerning for infection',
            '(infection)(\W+(?:\w+\W+){0,10}?)(not excluded)',
            'can t exclude consolidation',
            'cant exclude consolidation',
            '(compatible with|may represent|suggesting|consistent with|in keeping with)(\W+(?:\w+\W+){0,10}?)(pneumonia)',
            '(compatible with|may represent|suggesting|consistent with|in keeping with)(\W+(?:\w+\W+){0,10}?)(bronchopneumonia)',
            '(compatible with|may represent|suggesting|consistent with|in keeping with)(\W+(?:\w+\W+){0,10}?)(ards)'],
        'Consolidation':
           ['focal infiltrate',
            'infiltrate',
            'consolidation',
            '(consolidation)(\W+(?:\w+\W+){0,10}?)(not excluded)',
            'lobe consolidation',
            'confluent airspace opacities',
            'confluent opacities',
            'consolidative opacities'],
        'Peripheral/ILD':
           ['peripheral opacities',
            'peripheral opacity',
            'faint interstitial opacities',
            #'reticular opacities'
           ],
        #'PleuralEffusion':
        #   ['pleural effusion'],
        #'Halo':
        #   ['halo',
        #    'reverse halo'],
        'Unknown':
           ['hazy opacities',
            'airspace opacities',
            '(airspace)(\W+(?:\w+\W+){0,10}?)(opacities)',],
    }
    
    for gn, cxr in groups.items():
        df_group = df[['cxr_%d'%cxr_patterns.index(x) for x in cxr]]
        missing_ids = np.any(pd.isna(df_group), axis=1)
        vals = np.any(df_group.values, axis=1).astype(float)
        vals[missing_ids] = np.nan
        df['cxr_'+gn] = vals
    df = df.drop(columns=['cxr_%d'%x for x in range(len(cxr_patterns))])
    
    return df
    

def get_performance(y, yp, Nbt=1000, random_state=None, output_dir=None):
    assert len(y)==len(yp)
    np.random.seed(random_state)
    
    perf = defaultdict(list)
    enough_data = [True, True, True]
    for level in [1,2,3]:
        for i in tqdm(range(Nbt+1)):
            if i==0:
                ybt = y
                ypbt = yp
                
                min_y_count = sum([Counter(y)[x] for x in [1,2,3] if x >= level])
                if min_y_count < 3:
                    enough_data[level - 1] = False
                    
            else:
                btids = np.random.choice(len(y), len(y), replace=True)
                ybt = y[btids]
                ypbt = yp[btids]
                
                if sum([Counter(ybt)[x] for x in [1,2,3] if x >= level])<max(min_y_count//10, 1):
                    continue
                
            #y = ys_bt[i]
            #yp = yps_bt[i]
            #yp_int = np.argmax(yp, axis=1)
            #yp_acuityscore = np.sum(np.arange(yp.shape[1])*yp, axis=1)
            #yp_calib = yps_calib_bt[i]
            #perf['confusion_matrix_>=%d'%level].append(confusion_matrix(y, yp_int))
            #perf['cohen_kappa_>=%d'%level].append(cohen_kappa_score(y, yp_int))
            #perf['spearmanr_>=%d'%level].append(spearmanr(y, yp_acuityscore)[0])
            
            if enough_data[level - 1]:
                yy = (ybt>=level).astype(int)
                ypp = ypbt[:,level:].sum(axis=1)
                perf['N_>=%d'%level].append(len(yy))
                perf['auc_>=%d'%level].append(roc_auc_score(yy, ypp))
                perf['prc_>=%d'%level].append(average_precision_score(yy, ypp))
                fpr, tpr, tt = roc_curve(yy, ypp)
                tpr90_id = np.argmin(np.abs(tpr-0.9))
                thres = tt[tpr90_id]
                fpr = fpr[tpr90_id]
                ypp2 = (ypp>=thres).astype(int)
                perf['ppv_>=%d'%level].append(ppv_score(yy, ypp2))
                perf['npv_>=%d'%level].append(npv_score(yy, ypp2))
                perf['fpr_>=%d'%level].append(fpr)
                E = np.sum(ypp)
                O = np.sum(yy)
                perf['calib_EvO_>=%d'%level].append(E/O)
                perf['calib_err_perc_>=%d'%level].append((E-O)/E*100)
            else:
                perf['N_>=%d'%level].append(None)
                perf['auc_>=%d'%level].append(None)
                perf['prc_>=%d'%level].append(None)
                perf['ppv_>=%d'%level].append(None)
                perf['npv_>=%d'%level].append(None)
                perf['fpr_>=%d'%level].append(None)
                perf['calib_EvO_>=%d'%level].append(None)
                perf['calib_err_perc_>=%d'%level].append(None)
                
            
            if i==0:
                # plot
                pass
     
    # get confidence interval
    perf_ci = defaultdict(list)
    for key in perf:
        if 'confusion_matrix' in key.lower():
            continue
        perf_ci['metric'].append(key)
        perf_ci['value'].append(perf[key][0])
        if len(perf[key])==1 or key.startswith('N_') or perf[key][0] == None:
            perf_ci['lb'].append(np.nan)
            perf_ci['ub'].append(np.nan)
        else:
            perf_ci['lb'].append(np.percentile(perf[key][1:], 2.5))
            perf_ci['ub'].append(np.percentile(perf[key][1:], 97.5))
        try:
            val = '%f [%f -- %f]'%(perf_ci['value'][-1], perf_ci['lb'][-1], perf_ci['ub'][-1])
        except:
            val = 'NA [NA -- NA]'
            
    perf_ci = pd.DataFrame(data=perf_ci)
    return perf_ci
    
    

if __name__=='__main__':
    random_state = 2020
    
    ## read data
    
    upto_date = '20200503'#sys.argv[1]# up to this date
    start_date = '20200307'
    horizon = 7#int(sys.argv[2])# prediction horizon in days
    model_type = 'or'#sys.argv[3]#'or', 'rf', 'xgboost', 'catboost'
    print('\n=======================')
    print('data up to', upto_date, '(not inclusive)')
    print('horizon', horizon)
    print('model_type', model_type)
    print('=======================')
    
    Xinput_dir = '/data/Dropbox (Partners HealthCare)/COVID_RISK_PREDICTION/modeling/final_X_pre'
    Yinput_dir = '/data/Dropbox (Partners HealthCare)/COVID_RISK_PREDICTION/modeling'
    output_dir = os.path.join('figures_results', 'upto_%s_horizon%dday'%(upto_date, horizon))
    
    """
    upto_date = '20200515'
    day=datetime.date(2020,5,3)
    set1 = set(df_X.MRN[df_X.Date.dt.date<day])
    set2 = set(df_X.MRN[df_X.Date.dt.date>=day])
    common_set = (set1&set2)
    set1 = set1-common_set
    set2 = set2-common_set
    MRNs_pre  = np.array(list(set1))
    MRNs_post = np.array(list(set2))
    df = pd.DataFrame(data={
                        'MRN':np.r_[MRNs_pre, MRNs_post],
                        'prospective':np.r_[np.zeros(len(MRNs_pre)), np.ones(len(MRNs_post))]
                    })
    df[['MRN', 'prospective']].to_excel('MRNs_prospective_split.xlsx', index=False)
    """
    df = pd.read_excel('MRNs_prospective_split.xlsx')
    MRNs_pre = df.MRN[df.prospective==0].values
    
    ## generate X and y
    #"""
    df_y = read_Ydata(Yinput_dir, start_date, upto_date, horizon)
    df_X, df_X_deleted = read_Xdata(Xinput_dir, start_date, upto_date, return_deleted_columns=True, MRN_range=MRNs_pre)
    df_X, df_y, idsX, idsY = align_X_Y(df_X, df_y, horizon, for_training=True)
    df_X = group_cxr(df_X)
    df_X_deleted = df_X_deleted.iloc[idsX].reset_index(drop=True)
    
    X = df_X.drop(columns=['MRN', 'MGH_yn', 'Date', 'EventType', 'Department', 'PatientNM']).values.astype(float)
    Xnames = np.array(df_X.columns[6:]).astype(str)
    sids = df_X.MRN.values.astype(str)
    yordinal = df_y['y_ordinal'].values.flatten().astype(int)
    y = yordinal
    df_X = pd.concat([df_X, df_X_deleted], axis=1)
    
    ## remove columns that have missing values more than `thres`
    thres = 0.9
    missing_mask = np.isnan(X)
    missing_ratio = missing_mask.mean(axis=0)
    #print('missing_ratio')
    #print(np.c_[Xnames, missing_ratio])
    cols_remove_ind = missing_ratio>thres
    print('Removing columns with >%.0f%% missing values'%(thres*100,), Xnames[cols_remove_ind])
    X = X[:, ~cols_remove_ind]
    Xnames = Xnames[~cols_remove_ind]
    missing_mask = missing_mask[:, ~cols_remove_ind]
    missing_ratio = missing_ratio[~cols_remove_ind]
    
    ## remove columns that have std=0
    std = np.nanstd(X, axis=0)
    std[np.isnan(std)] = 0
    print('Removing columns with std=0', Xnames[std==0])
    X = X[:, std>0]
    Xnames = Xnames[std>0]
    missing_mask = missing_mask[:, std>0]
    missing_ratio = missing_ratio[std>0]
    
    ## impute X
    # first exclude features allowing missingness
    allow_missing_ids = np.array([x.startswith('cxr_') or x=='ever_positive_upToEvent' for x in Xnames])
    X_missing = X[:,allow_missing_ids]
    X_nomissing = X[:,~allow_missing_ids]
    X_nomissing, imputer_model = myimpute_opt(X_nomissing, Xnames[~allow_missing_ids], random_state=random_state)
    X[:,~allow_missing_ids] = X_nomissing
    with open(os.path.join(output_dir, 'imputer_model_%ddays.pickle'%horizon), 'wb') as ff:
        pickle.dump([imputer_model, Xnames], ff)
    with open(os.path.join(output_dir, 'Xy.pickle'), 'wb') as ff:
        pickle.dump([df_X, df_y, X, y, sids, Xnames], ff)
    #"""
    #with open(os.path.join(output_dir, 'Xy.pickle'), 'rb') as ff:
    #    df_X, df_y, X, y, sids, Xnames = pickle.load(ff)
    """
    ## add columns indicating missingness
    cxr_col = [x for x in Xnames if x.startswith('cxr_')]
    if len(cxr_col)>0:
        cxr_col = [cxr_col[0]]
        if missing_ratio[Xnames==cxr_col][0]==0:
            cxr_col = []
    cols_add_ismissing_ind = np.r_[Xnames[(missing_ratio>0)&(~np.char.startswith(Xnames, 'cxr'))], cxr_col]
    indicators = [missing_mask[:,Xnames==col].astype(float) for col in cols_add_ismissing_ind]
    X = np.concatenate([X]+indicators, axis=1)
    Xnames = np.r_[Xnames, ['ismissing_'+col for col in cols_add_ismissing_ind]]
    """
    
    print('%d unique MRNs'%len(set(sids)))
    print(Counter(y))
    print('%d rows'%len(X))
    print('%d cols:'%len(Xnames), Xnames)
    
    # do bootstrap to get the confidence interval around the coefs 
    random_state = 2020
    np.random.seed(random_state)
    ncv = 5
    start_time = timeit.default_timer()
        
    # fit the model using Xbt and ybt
    model, models_cv, hp, yptes, yp_calib_tes, te_ids, cv_ids = myfit(model_type, X, y, sids,
                                                   Xnames, ncv=ncv, random_state=random_state)
    ## get running time
    end_time = timeit.default_timer()
    print('Used %.1f seconds'%(end_time-start_time))
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    ## coefficent
    if type(model.base_estimator)==Pipeline:
        coef = model.base_estimator.steps[-1][-1].coef_.flatten()
    else:
        coef = model.base_estimator.get_coef()
    coef_df = pd.DataFrame(data={'Xnames':Xnames, 'coef':coef})
    coef_df.to_csv(os.path.join(output_dir, 'coef_%ddays_%s.csv'%(horizon, model_type)), index=False)
    
    ## save
    with open(os.path.join(output_dir, 'model_%ddays_%s.pickle'%(horizon, model_type)), 'wb') as ff:
        pickle.dump([model, models_cv], ff)
    sids_tes = sids[te_ids]
    ids = [list(sids_tes).index(x) for x in sids]
    cxr = (~pd.isna(df_X['cxr_Multifocal'])).values.astype(int)
    data={'MRN':sids,
          'CVFold':cv_ids[ids],
          'yordinal':y,
          'AcuityScore': np.sum(np.arange(yptes.shape[1])*yptes[ids], axis=1),
          'MGH_yn':df_X.MGH_yn,
          'Date':df_X.Date,
          'EventType':df_X.EventType,
          'Department':df_X.Department,
          'PatientNM':df_X.PatientNM,
          'CXR':cxr}
    cols = [str(x) for x in range(yptes.shape[1])]
    cols_txt = []
    for ci, col in enumerate(cols):
        cols_txt.append('%d day P(%s) (%%)'%(horizon, col))
        data.update({cols_txt[-1]: yptes[ids][:,ci]*100})
    cols = ['None', 'Hosp', 'ICU or Intub', 'Death']
    for ci, col in enumerate(cols):
        cols_txt.append('%d day calibrated P(%s) (%%)'%(horizon, col))
        data.update({cols_txt[-1]: yp_calib_tes[ids][:,ci]*100})
    df_y_yp = pd.DataFrame(data=data)
    df_y_yp = df_y_yp[['MRN', 'MGH_yn', 'Date', 'EventType', 'Department', 'PatientNM', 'CXR',
                       'CVFold', 'yordinal', 'AcuityScore']+cols_txt]
    df_y_yp.to_excel(os.path.join(output_dir, 'cv_y_yp_%ddays_%s.xlsx'%(horizon, model_type)), index=False)
    
    ## get performance
    ytes = y[te_ids]
    perf_ci = get_performance(ytes, yp_calib_tes, random_state=random_state, output_dir=output_dir)
    print(perf_ci)
    perf_ci.to_excel(os.path.join(output_dir, 'cv_perf_%ddays_%s.xlsx'%(horizon, model_type)), index=False)
    
    fi = get_feature_contrib(model, X)
    df_fi = pd.DataFrame(data=np.c_[sids.astype(object), fi.astype(object)], columns=['MRN']+list(Xnames))
    df_fi.to_excel(os.path.join(output_dir, 'feature_importance_upto%s_%ddays_%s.xlsx'%(upto_date, horizon, model_type)), index=False)
