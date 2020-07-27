import numpy as np

from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

import neuralnet

###################################
# Auxiliary functions
###################################

def to_density(x, bins=5, bounds=None):
    """"Turn into density based nb of bins"""
    p_x = np.histogram(x, bins=bins, density=True, range=bounds)[0]
    p_x = p_x / np.sum(p_x)
    return p_x  

def get_pairs(val):
    """Given val, returns the set of pairs 1 <= {i, j} <= val, s.t. order does not matter"""
    pairs = []
    for i in range(val):
        for j in range(i, val):
            pairs.append([i, j])

    return pairs

def normalize(v, mean=None, var=None):

    if not mean:
        mean = np.mean(v)
    if not var:
        var = np.linalg.norm(v)

    v = v - mean
    if var == 0: 
       return v
    return v / var

def denormalise(v, mean, var):
    return var*mean + mean

def get_moments(v):
    return np.mean(v), np.linalg.norm(v)


###################################
# Integrate linear model with poly feature builder class
###################################


class PolyRegreg:

    def __init__(self, degree, olsreg, pvalue=.05):
        self.degree = degree
        self.olsreg = olsreg

    def feature_transform(self, x):
        polynomial_features= PolynomialFeatures(degree=self.degree)
        x_poly = polynomial_features.fit_transform(x.reshape(-1, 1))
        return x_poly

    def predict(self, x):
        # Create polynomial features
        x_poly = self.feature_transform(x)

        # predict 
        y_pred = self.olsreg.predict(x_poly)
        return y_pred


###################################
# Regression related classes
###################################

def model_selection(x, y, max_degree=6):
    # Note to remove non-significant weights:
    # mask = results.pvalues < .05
    # x_poly_ = x_poly[:, mask]
    
    bic_scores = []
    olsreg_models = []

    for d in range(max_degree+1):

        polynomial_features= PolynomialFeatures(degree=d)
        x_poly = polynomial_features.fit_transform(x.reshape(-1, 1))

        olsreg = sm.OLS(y, x_poly).fit()
        bic_scores.append(olsreg.bic)
        olsreg_models.append(olsreg)

    opt_degree = bic_scores.index(min(bic_scores))
    opt_olsreg = olsreg_models[opt_degree]

    opt_model = PolyRegreg(degree=opt_degree, olsreg=opt_olsreg)
    
    return opt_model

def compute_residuals(X_, Y_, models, norm=False):
    residuals = []

    for i in range(len(X_)):
        y_pred = models[i].predict(X_[i])
        r = Y_[i] - y_pred
        if norm:
            r = normalize(r)
        residuals.append(r)

    return residuals


##

def get_model(x, y, model_params=None):

    if not model_params:
        model_type = 'PolyRegreg'
    else:
        model_type = model_params['model_type']


    model = None

    if model_type == 'PolyRegreg':
        model = model_selection(x, y)

    elif model_type == 'NeuralNet':
        model = neuralnet.Net(model_params)
        model.train(x, y)

    else:
        raise NameError('Unkown model:', model_type)

    return model

