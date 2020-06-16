#!/usr/bin/env python
# coding: utf-8
from collections import defaultdict
import datetime
import glob
import os
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import shap
from a6_train_risk_model import binarize_imputed_value, group_cxr, read_Xdata, read_Ydata, get_performance, align_X_Y, get_feature_contrib
from myclasses import WeightedKNNImputer, MyXGBRanker, MyStagedWrapper, MyLogisticRegression, MyPreprocessing, LTRPairwise


if __name__=='__main__':
    
    ## read data
    predict_start_date = '20200503'#sys.argv[1]
    predict_end_date = '20200515'#sys.argv[2]
    model_date = '20200503'#sys.argv[3]
    horizon = 7#int(sys.argv[4])# prediction horizon in days
    model_type = 'or'#sys.argv[5]#'logreg', 'dt'
    
    topK = 5
    model_dir = 'models'
    Xinput_dir = '/data/Dropbox (Partners HealthCare)/COVID_RISK_PREDICTION/modeling/final_X_pre'
    output_dir = '/data/Dropbox (Partners HealthCare)/RISK_PREDICTIONS_SHARED/'+predict_end_date
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    ## load model
    with open(os.path.join(model_dir, 'imputer_model_%ddays.pickle'%horizon), 'rb') as ff:
        imputer, Xnames = pickle.load(ff)
    with open(os.path.join(model_dir, 'model_%ddays_%s.pickle'%(horizon, model_type)), 'rb') as ff:
        model, models_CV = pickle.load(ff)
    
    get_y = True
    if get_y:
        df = pd.read_excel('MRNs_prospective_split.xlsx')
        MRNs_post = df.MRN[df.prospective==1].values
        MRN_range = MRNs_post
    else:
        MRN_range = None
    #"""
    df_X, df_X_deleted = read_Xdata(Xinput_dir, predict_start_date, predict_end_date, return_deleted_columns=True, MRN_range=MRN_range)
    df_X = group_cxr(df_X)
    
    if get_y:
        Yinput_dir = '/data/Dropbox (Partners HealthCare)/COVID_RISK_PREDICTION/modeling'
        df_y = read_Ydata(Yinput_dir, '20200307', predict_end_date, horizon)
        df_X, df_y, idsX, _ = align_X_Y(df_X, df_y, horizon)
        df_X_deleted = df_X_deleted.iloc[idsX].reset_index(drop=True)
        y = df_y.y_ordinal.values.astype(float)
    
    df_X = pd.concat([df_X, df_X_deleted], axis=1)
        
    # take the most recent date of each patient
    mrns = df_X.MRN.values.astype(int)
    dates = pd.to_datetime(df_X.Date).values
    _, idx = np.unique(mrns, return_index=True)
    unique_mrns = mrns[np.sort(idx)]
    ids = []
    for mrn in unique_mrns:
        ids_ = np.where(df_X.MRN==mrn)[0]
        ids.append(ids_[np.argmin(dates[ids_])])
    df_X = df_X.iloc[ids].reset_index(drop=True)
    if get_y:
        df_y = df_y.iloc[ids].reset_index(drop=True)
        y = y[ids]
    
    X = df_X[Xnames].values.astype(float)
    missing_mask = np.isnan(X)
    missing_ratio = missing_mask.mean(axis=0)
    
    ## impute X
    allow_missing_ids = np.array([x.startswith('cxr_') or x=='ever_positive_upToEvent' for x in Xnames])
    X_nomissing = X[:,~allow_missing_ids]
    X_nomissing = imputer.transform(X_nomissing)
    # binarize binary columns
    X_nomissing = binarize_imputed_value(X_nomissing, Xnames[~allow_missing_ids], missing_mask[:,~allow_missing_ids])
    X[:,~allow_missing_ids] = X_nomissing
    
    if get_y:
        with open(os.path.join(output_dir, 'Xy_prospective.pickle'), 'wb') as ff:
            pickle.dump([df_X, df_y, X, y, Xnames], ff)
    #"""
    #if get_y:
    #    with open(os.path.join(output_dir, 'Xy_prospective.pickle'), 'rb') as ff:
    #        df_X, df_y, X, y, Xnames = pickle.load(ff)
    
    # get predictions
    yp = model.base_estimator.predict_proba(X)
    yp_acuityscore = model.predict(X)
    yp_calib = model.predict_proba(X)
    
    if get_y:
        # get performance
        perf_ci = get_performance(y, yp_calib)
        print(perf_ci)
    
    # get feature contribution
    feature_contrib = get_feature_contrib(model, X)
    feature_rank = np.argsort(-feature_contrib, axis=1)
    topfeatures = []
    for ii in range(len(X)):
        topfeatures.append([])
        for jj in range(topK):
            fn = Xnames[feature_rank[ii,jj]]
            val = X[ii,feature_rank[ii,jj]]
            coef_ = feature_contrib[ii,feature_rank[ii,jj]]
            if coef_ <=0:
                topfeatures[-1].append('')
            else:
                topfeatures[-1].append('%s:%.4g'%(fn, val))#:%.4g, coef_
        for jj in range(topK):
            fn = Xnames[feature_rank[ii,-jj-1]]
            val = X[ii,feature_rank[ii,-jj-1]]
            coef_ = feature_contrib[ii,feature_rank[ii,-jj-1]]
            if coef_ >=0:
                topfeatures[-1].append('')
            else:
                topfeatures[-1].append('%s:%.4g'%(fn, val))
    topfeatures = np.array(topfeatures)
        
    has_cxr = (~np.isnan(X[:,Xnames=='cxr_Multifocal'].flatten())).astype(int)
    
    #"""
    # get pt names
    demo_paths = glob.glob('/data/Dropbox (Partners HealthCare)/COVID_RISK_PREDICTION/data_demographics/*Demog*.csv')
    df_demo = [pd.read_csv(dp) for dp in demo_paths]
    df_demo = pd.concat(df_demo, axis=0).reset_index(drop=True)
    mrn2name = {df_demo.MRN.iloc[i]: df_demo.PatientNM.iloc[i] for i in range(len(df_demo))}
    pt_names = [mrn2name.get(x, 'NOT_FOUND') for x in df_X.MRN]
    #"""
    
    # convert risk to %
    yp = yp * 100
    yp_calib = yp_calib * 100
    
    ycols = ['None', 'Hosp', 'ICU or Intub', 'Death']
    ylevels = [0,     1,    2,     3]
    risk_cols = []
    data={'MRN':df_X.MRN.values.astype(str),
          'PatientName':pt_names,
          'Date':df_X.Date.values.astype(str),
          'Location':df_X.EventType.values.astype(str),
          'Department':df_X.Department.values.astype(str),
          'CXR':has_cxr}
    for yi in range(yp.shape[1]):
        col = '%d day P(%d) (%%)'%(horizon, yi)
        data[col] = yp[:, yi]
        risk_cols.append(col)
    for yi in range(len(ycols)):
        col = '%d day calibrated P(%s) (%%)'%(horizon, ycols[yi])
        data[col] = yp_calib[:, ylevels[yi]]
        risk_cols.append(col)
    # add acuity score
    score_col = 'AcuityScore'
    data[score_col] = yp_acuityscore
    if get_y:
        # add groundtruth label
        data['yordinal'] = y
    df_yp = pd.DataFrame(data=data)
    
    # add an empty column
    #df_yp[' '] = np.zeros(len(df_yp))+np.nan
    # add PXS score
    df_yp['PXS Score'] = df_X.PXS
    # sort by risk
    risk_sort_ids = np.argsort(-df_yp[score_col].values.astype(float))
    X = X[risk_sort_ids]
    topfeatures = topfeatures[risk_sort_ids]
    df_X = df_X.iloc[risk_sort_ids].reset_index(drop=False)
    df_yp = df_yp.iloc[risk_sort_ids].reset_index(drop=False)
    #df_y = df_y.iloc[risk_sort_ids].reset_index(drop=False)
    # format to 1 decimal
    #df_yp[risk_cols+[score_col]] = df_yp[risk_cols+[score_col]].round(decimals=1)
    #df_yp['Yordinal']=df_y.y_ordinal.values
    # add features
    for xi, xn in enumerate(Xnames):
        df_yp[xn] = X[:,xi]
    
    col = ['MRN', 'PatientName', 'Date', 'Location', 'Department', 'CXR']
    if get_y:
        # add groundtruth label
        col.append('yordinal')
    col = col+[score_col, 'PXS Score']+risk_cols+list(Xnames)
    df_yp = df_yp[col]
    
    # create data frame of top features
    df_top_feature = pd.DataFrame(
            data=topfeatures,
            columns=['PostiveTop%d'%i for i in range(1,topK+1)]+['NegativeTop%d'%i for i in range(1,topK+1)])
    
    # save
    if get_y:
        perf_ci.to_excel(os.path.join(output_dir, 'perf_%ddays_%s.xlsx'%(horizon, model_type)), index=False)
    df_yp.to_csv(os.path.join(output_dir, 'risk_%ddays_%s_%s.csv'%(horizon, predict_end_date, model_type)), index=False, sep='|')
    df_top_feature.to_csv(os.path.join(output_dir, 'top_feature_%ddays_%s_%s.csv'%(horizon, predict_end_date, model_type)), index=False, sep='|')


