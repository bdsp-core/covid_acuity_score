from itertools import combinations
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFpr
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._logistic import _logistic_loss_and_grad
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from xgboost import XGBRanker
from mord import LogisticAT


def get_sample_weights(y, class_weight='balanced'):
    assert y.min()==0 ## assume y=[0,1,...]
    K = y.max()+1
    class_weights = compute_class_weight(class_weight, np.arange(K), y)
    sw = np.array([class_weights[yy] for yy in y])
    return sw


class WeightedKNNImputer(KNNImputer):
    def __init__(self, missing_values=np.nan, n_neighbors=5, feature_weights=None):
        super().__init__(missing_values=missing_values, n_neighbors=n_neighbors, weights='uniform', metric='nan_euclidean', copy=True, add_indicator=False)
        self.feature_weights = feature_weights
    
    def fit(self, X, y=None):
        X2 = np.array(X)
        
        # normalize to [0,1]
        self.max = np.nanmax(X2, axis=0)
        self.min = np.nanmin(X2, axis=0)
        ids = self.max==self.min
        self.max[ids] = 1
        self.min[ids] = 0
        X2 = (X2-self.min)/(self.max-self.min)
        
        # apply weights
        if self.feature_weights is None:
            self.feature_weights = 1
        X2 = X2*self.feature_weights
        
        # impute
        super().fit(X2)
        
        return self
    
    def transform(self, X):
        X2 = (X-self.min)/(self.max-self.min)
        
        # apply weights
        if self.feature_weights is None:
            self.feature_weights = 1
        X2 = X2*self.feature_weights
        
        X2 = super().transform(X2)
        
        # inverse weights
        X2 /= self.feature_weights
        # inverse normalization
        X2 = X2*(self.max-self.min) + self.min
        
        return X2


class MyPreprocessing(StandardScaler):
    def __init__(self, allow_missing_ids=None, copy=True, with_mean=True, with_std=True):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)
        self.allow_missing_ids = allow_missing_ids
    
    def fit(self, X, y=None):
        if self.allow_missing_ids is None:
            self.allow_missing_ids = np.zeros(X.shape[1]).astype(bool)
        super().fit(X[:, ~self.allow_missing_ids])
        return self
        
    def transform(self, X, copy=None):
        X2 = super().transform(X[:, ~self.allow_missing_ids])
        
        X3 = X[:, self.allow_missing_ids]
        X3[np.isnan(X3)] = -999
        X3[X3==0] = -1
        X3[X3==-999] = 0
        
        Xres = np.zeros_like(X)
        Xres[:,~self.allow_missing_ids] = X2
        Xres[:,self.allow_missing_ids] = X3
        return Xres


class MyCalibrator:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        
    def fit(self, X, y):
        yp = self.predict(X)
        self.recalibration_mapper = LogisticAT(alpha=0).fit(yp.reshape(-1,1), y)
        return self
    
    def predict(self, X):
        K = len(self.base_estimator.classes_)
        yp = np.sum(self.base_estimator.predict_proba(X)*np.arange(K), axis=1)
        return yp
        
    def predict_proba(self, X):
        yp = self.predict(X)
        yp2 = self.recalibration_mapper.predict_proba(yp.reshape(-1,1))
        return yp2


class MyStagedWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model1, model2, allow_missing_ids):
        self.model1 = model1
        self.model2 = model2
        self.allow_missing_ids = allow_missing_ids
    
    def _predict_z(self, X):
        z_ = X
        for i in range(len(self.model1.steps)-1):
            z_ = self.model1.steps[i][-1].transform(z_)
        z = self.model1.steps[-1][-1].predict_z(z_)
        return z
        
    def fit(self, X, y):
        self.model1 = clone(self.model1)
        self.model2 = clone(self.model2)
        
        # stage 1: train all data without using CXR features
        X_nomissing = X[:,~self.allow_missing_ids]
        assert ~np.any(np.isnan(X_nomissing))
        if type(self.model1.steps[-1][-1])==MyXGBRanker:
            sw = [1]
        else:
            sw = get_sample_weights(y)
        self.model1.fit(X_nomissing, y, model__sample_weight=sw)
        
        self.label_encoder = self.model1.steps[-1][-1].label_encoder
        self.classes_ = self.model1.steps[-1][-1].classes_
        
        # stage 2: train +CXR data using CXR features
        missing_ids = np.any(np.isnan(X[:,self.allow_missing_ids]), axis=1)
        if hasattr(self.model1.steps[-1][-1], 'predict_z'):
            z = self._predict_z(X[~missing_ids][:,~self.allow_missing_ids])
        elif hasattr(self.model1.steps[-1][-1], 'decision_function'):
            z = self.model1.decision_function(X[~missing_ids][:,~self.allow_missing_ids])
        else:
            z = self.model1.predict(X[~missing_ids][:,~self.allow_missing_ids])
            
        Xcxr = X[~missing_ids][:,self.allow_missing_ids]
        Xcxr[Xcxr==0] = -1
        
        X_missing = np.c_[z, Xcxr]
        if type(self.model1.steps[-1][-1])==MyXGBRanker:
            sw = [1]
        else:
            sw = get_sample_weights(y[~missing_ids])
        self.model2.fit(X_missing, y[~missing_ids], sample_weight=sw)
        
        return self
    
    def predict_proba(self, X):
        # stage 1: predict data without CXR features
        missing_ids = np.any(np.isnan(X[:,self.allow_missing_ids]), axis=1)
        if missing_ids.sum()>0:
            X_nomissing = X[missing_ids][:,~self.allow_missing_ids]
            assert ~np.any(np.isnan(X_nomissing))
            yp1 = self.model1.predict_proba(X_nomissing)
        else:
            yp1 = []
        
        if (~missing_ids).sum()>0:
            # stage 2: predict data with CXR features
            if hasattr(self.model1.steps[-1][-1], 'predict_z'):
                z = self._predict_z(X[~missing_ids][:,~self.allow_missing_ids])
            elif hasattr(self.model1.steps[-1][-1], 'decision_function'):
                z = self.model1.decision_function(X[~missing_ids][:,~self.allow_missing_ids])
            else:
                z = self.model1.predict(X[~missing_ids][:,~self.allow_missing_ids])
            Xcxr = X[~missing_ids][:,self.allow_missing_ids]
            Xcxr[Xcxr==0] = -1
            X_missing = np.c_[z, Xcxr]
            yp2 = self.model2.predict_proba(X_missing)
            
            yp = np.zeros((len(X), yp2.shape[1]))
            if len(yp1)>0:
                yp[missing_ids] = yp1
            yp[~missing_ids] = yp2
        else:
            yp = yp1
        
        return yp
        
    def predict(self, X):
        yp = self.predict_proba(X)
        yp1d = self.label_encoder.inverse_transform(np.argmax(yp, axis=1))
        return yp1d
        
    def get_coef(self):
        m = self.model1.steps[-1][-1]
        if hasattr(m, 'estimator'):
            m = m.estimator
        if hasattr(m, 'feature_importances_'):
            coef_model1 = m.feature_importances_.flatten()
        elif hasattr(m, 'coef_'):
            coef_model1 = m.coef_.flatten()
        m = self.model2
        if hasattr(m, 'estimator'):
            m = m.estimator
        if hasattr(m, 'feature_importances_'):
            coef_model2 = m.feature_importances_.flatten()
        elif hasattr(m, 'coef_'):
            coef_model2 = m.coef_.flatten()
            
        coefs = np.zeros(len(self.allow_missing_ids))
        coefs[~self.allow_missing_ids] = coef_model1
        coefs[self.allow_missing_ids] = coef_model2[1:]
        return coefs
        
    def get_intercept(self):
        m = self.model1.steps[-1][-1]
        if hasattr(m, 'estimator'):
            m = m.estimator
        if hasattr(m, 'feature_importances_'):
            intercept_model1 = 0
        elif hasattr(m, 'intercept_'):
            intercept_model1 = m.intercept_[0]
        m = self.model2
        if hasattr(m, 'estimator'):
            m = m.estimator
        if hasattr(m, 'feature_importances_'):
            intercept_model2 = 0
        elif hasattr(m, 'coef_'):
            intercept_model2 = m.intercept_[0]
        return np.r_[intercept_model1, intercept_model2]


class MyLogisticRegression(LogisticRegression):
    """
    univariate feature selection as first step
    Removes regularization on intercept
    Allows bounds
    Binary only
    L1 only
    L-BFGS-B or BFGS only
    """
    def __init__(self, univariate_feature_selection=False, allow_missing_ids=None,
                 class_weight=None, tol=1e-6, C=1.0, random_state=None, max_iter=1000, bounds=None):
        super().__init__(penalty='l1', dual=False, tol=tol, C=C,
                 fit_intercept=True, intercept_scaling=1, class_weight=class_weight,
                 random_state=random_state, solver='lbfgs', max_iter=max_iter,
                 multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None)
        self.univariate_feature_selection = univariate_feature_selection
        self.allow_missing_ids = allow_missing_ids
        self.bounds = bounds
                 
    def fit(self, X, y, sample_weight=None):
        self.label_encoder = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder.classes_
        y = self.label_encoder.transform(y)
        
        if self.allow_missing_ids is None:
            self.allow_missing_ids = np.zeros(X.shape[1]).astype(bool)
            
        if self.univariate_feature_selection:
            # univariate feature selection
            feature_selector = SelectFpr(alpha=0.05).fit(X[:,~self.allow_missing_ids], y)
            self.support = np.ones(X.shape[1]).astype(bool)
            self.support[~self.allow_missing_ids] = feature_selector.get_support()
            X = X[:, self.support]
            if self.bounds is not None:
                self.bounds = [self.bounds[ii] for ii in range(len(self.bounds)) if self.support[ii]]
        else:
            self.support = np.ones(X.shape[1]).astype(bool)
        
        def func(w, X, y, alpha, sw):
            out, grad = _logistic_loss_and_grad(w, X, y, 0, sw)
            out_penalty = alpha*np.sum(np.abs(w[:-1]))
            grad_penalty = np.r_[alpha*np.sign(w[:-1]) ,0]
            return out+out_penalty, grad+grad_penalty
        
        y2 = np.array(y)
        y2[y2==0] = -1
        w0 = np.r_[np.random.randn(X.shape[1])/10, 0.]
        if self.bounds is None:
            method = 'BFGS'
        else:
            method = 'L-BFGS-B'
        if sample_weight is None:
            if self.class_weight is not None:
                sample_weight = get_sample_weights(y, class_weight=self.class_weight)
            else:
                sample_weight = np.ones(len(X))
        sample_weight /= (np.mean(sample_weight)*len(X))
        self.opt_res = minimize(
            func, w0, method=method, jac=True,
            args=(X, y2, 1./self.C, sample_weight),
            bounds=self.bounds+[(None,None)],
            options={"gtol": self.tol, "maxiter": self.max_iter}
        )
        self.coef_ = np.zeros(len(self.support))
        self.coef_[self.support] = self.opt_res.x[:-1]
        self.coef_ = self.coef_.reshape(1,-1)
        self.intercept_ = self.opt_res.x[-1].reshape(1,)
        return self


class LTRPairwise(BaseEstimator, ClassifierMixin):
    """Learning to rank, pairwise approach
    For each pair A and B, learn a score so that A>B or A<B based on the ordering.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        It must be a classifier with a ``decision_function`` function.
    verbose : bool, optional, defaults to False
        Whether prints more information.
    """
    def __init__(self, estimator,
                    allow_missing_ids=None, missing_val=None,
                    univariate_feature_selection=False,
                    class_weight=None, min_level_diff=1, verbose=False):
        super().__init__()
        self.estimator = estimator
        self.missing_val = missing_val
        self.allow_missing_ids = allow_missing_ids
        self.univariate_feature_selection = univariate_feature_selection
        self.class_weight = class_weight
        self.min_level_diff = min_level_diff
        self.verbose = verbose
        
    #def __setattr__(self, name, value):
    #    setattr(self.estimator, name, value)
    #    super().__setattr__(name, value)
        
    def _generate_pairs(self, X, y, sample_weight):
        if np.any(self.allow_missing_ids):
            missing = np.any(X[:, self.allow_missing_ids]==self.missing_val, axis=1)
        else:
            missing = np.zeros(len(X)).astype(bool)
        missing_ids = np.where(missing)[0]
        nomissing_ids = np.where(~missing)[0]
        
        X2 = []
        y2 = []
        sw2 = []
        for i, j in combinations(nomissing_ids, 2):
            # if there is a tie, ignore it
            if np.abs(y[i]-y[j])<self.min_level_diff:
                continue
            X2.append( X[i]-X[j] )
            y2.append( 1 if y[i]>y[j] else 0 )
            if sample_weight is not None:
                sw2.append( max(sample_weight[i], sample_weight[j]) )
        
        for i, j in combinations(missing_ids, 2):
            # if there is a tie, ignore it
            if np.abs(y[i]-y[j])<self.min_level_diff:
                continue
            X2.append( X[i]-X[j] )
            y2.append( 1 if y[i]>y[j] else 0 )
            if sample_weight is not None:
                sw2.append( max(sample_weight[i], sample_weight[j]) )

        if sample_weight is None:
            sw2 = None
        else:
            sw2 = np.array(sw2)

        return np.array(X2), np.array(y2), sw2

    def fit(self, X, y, sample_weight=None):
        self.fitted_ = False
        if self.allow_missing_ids is None:
            self.allow_missing_ids = np.zeros(X.shape[1]).astype(bool)
            
        Xold = np.array(X)
        if self.univariate_feature_selection:
            # univariate feature selection
            feature_selector = SelectFpr(alpha=0.05).fit(X[:,~self.allow_missing_ids], y)
            self.support = np.ones(X.shape[1]).astype(bool)
            self.support[~self.allow_missing_ids] = feature_selector.get_support()
            X = X[:, self.support]
            self.allow_missing_ids = self.allow_missing_ids[self.support]
        else:
            self.support = np.ones(X.shape[1]).astype(bool)
        
        if sample_weight is None:
            if self.class_weight is not None:
                sample_weight = get_sample_weights(y, class_weight=self.class_weight)
            else:
                sample_weight = np.ones(len(X))
        sample_weight /= (np.mean(sample_weight)*len(X))
        
        # generate pairs
        X2, y2, sw2 = self._generate_pairs(X, y, sample_weight)
        sw2 = sw2/sw2.mean()
        if self.verbose:
            print('Generated %d pairs from %d samples'%(len(X2), len(X)))

        # fit the model
        if self.estimator.bounds is not None:
            self.estimator.bounds = [self.estimator.bounds[ii] for ii in range(len(self.estimator.bounds)) if self.support[ii]]
        self.estimator.fit(X2, y2, sample_weight=sw2)

        # get the mean of z for each level of y
        self.label_encoder = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder.classes_
        z = self.predict_z(Xold)
        self.z_means = np.array([z[y==cl].mean() for cl in self.label_encoder.classes_])

        self.coef_ = np.zeros(len(self.support))
        self.coef_[self.support] = self.estimator.coef_.flatten()
        self.coef_ = self.coef_.reshape(1,-1)
        self.intercept_ = self.estimator.intercept_
        self.fitted_ = True
        return self

    def predict_z(self, X):
        X = X[:, self.support]
        z = self.estimator.decision_function(X)
        return z

    def predict_proba(self, X):
        z = self.predict_z(X)
        dists = -(z.reshape(-1,1) - self.z_means)**2
        yp = softmax(dists, axis=1)
        return yp

    def predict(self, X):
        yp = self.predict_proba(X)
        yp1d = self.label_encoder.inverse_transform(np.argmax(yp, axis=1))
        return yp1d

    def score(self, X, y):
        yp = self.predict(X)
        return kendalltau(y, yp)[0]


class MyXGBRanker(XGBRanker):
    """
    univariate feature selection as first step
    + convert to probability
    """
    def __init__(self, univariate_feature_selection=False, allow_missing_ids=None, **kwargs):
        super().__init__(**kwargs)
        self.univariate_feature_selection = univariate_feature_selection
        self.allow_missing_ids = allow_missing_ids
        
    def fit(self, X, y, sample_weight=None):
        if self.allow_missing_ids is None:
            self.allow_missing_ids = np.zeros(X.shape[1]).astype(bool)
            
        if self.univariate_feature_selection:
            # univariate feature selection
            feature_selector = SelectFpr(alpha=0.05).fit(X[:,~self.allow_missing_ids], y)
            self.support = np.ones(X.shape[1]).astype(bool)
            self.support[~self.allow_missing_ids] = feature_selector.get_support()
            X = X[:, self.support]
        else:
            self.support = np.ones(X.shape[1]).astype(bool)
            
        # fit the model
        super().fit(X, y, [len(X)], sample_weight=sample_weight)

        # get the mean of z for each level of y
        self.label_encoder = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder.classes_
        z = super().predict(X).astype(float)
        self.z_means = np.array([z[y==cl].mean() for cl in self.label_encoder.classes_])
        return self
    
    def predict_z(self, X):
        X = X[:, self.support]
        z = super().predict(X).astype(float)
        return z
        
    def predict_proba(self, X):
        z = self.predict_z(X)
        dists = -(z.reshape(-1,1) - self.z_means)**2
        yp = softmax(dists, axis=1)
        return yp
    
    def predict(self, X):
        yp = self.predict_proba(X)
        yp1d = self.label_encoder.inverse_transform(np.argmax(yp, axis=1))
        return yp1d
    
    @property
    def feature_importances_(self):
        fi = np.zeros(len(self.support))
        fi[self.support] = super().feature_importances_.flatten()
        return fi

