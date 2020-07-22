from sklearn import metrics
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy 
from numpy import random as rand

class Metrics:
    '''
    A class implementing metrics to measure statistical dependence

    MI:
    The heuristic used for the number of bins used for discretisation is based on:

    https://stats.stackexchange.com/questions/179674/number-of-bins-when-computing-mutual-information

    Note that there is in general no optimal bin size.
    '''

# To add: Hellinger distance, TV, Wasserstein, ... ?

    ####
    # Mutural Information
    ####

    @staticmethod
    def nb_bins(nb_samples):
        """Given nb_samples, returns a reasonable discretisation size (i.e. nb of bins)"""
        return int(np.floor(np.sqrt(nb_samples/5)))

    @staticmethod
    def MI(x, y, nb_samples=None):
        """Compute the discretised empirical Mutual information between samples x and y"""

        if not nb_samples:
            nb_samples = len(x)

        bins = Metrics.nb_bins(nb_samples)
        c_xy = np.histogram2d(x, y, bins)[0]
        mi = metrics.mutual_info_score(None, None, contingency=c_xy)
        return mi


    @staticmethod
    def metric_from_str(metric_name):
        metric_dict = {
            'MI': Metrics.MI
        }

        return metric_dict[metric_name]



class LinearModel:
    ''' sklearn linear model wrapper to avoid verbose reshaping
    '''
    # TODO apdapt from notebook nonlinear -> test out method on result 
    def __init__(self, degree=1):
        self.model = LinearRegression(fit_intercept=True)

    def adapt_shape(self, *args):
        if len(args) == 1:
            return args[0].reshape(-1, 1) 
        return [x.reshape(-1, 1) for x in args]

    def fit(self, x, y):
        x_, y_ = self.adapt_shape(x, y)
        self.model.fit(x_, y_)

    def predict(self, x):
        x_ = self.adapt_shape(x)
        y_hat = self.model.predict(x_)
        return y_hat.reshape(-1)


class NoiseFactory:

    def __init__(self, noise_config, size):
        self.name = noise_config['name']
        self.var = noise_config['var']
        self.size = size
        self.noise_config = noise_config

    def sample(self):
        name = self.name
        var = self.var
        size = self.size

        if name == 'normal':
            return rand.normal(loc=0, scale=var, size=size)

        elif name == 'uniform':
            radius = np.sqrt(3 * var)
            return rand.uniform(low=-radius, high=radius, size=size)
        
        elif name == 'exp':
            return np.random.exponential(scale=1/np.sqrt(var), size=size)

        elif name == 'laplace':
            return np.random.laplace(loc=0.0, scale=var, size=size)

        elif name == 'uniform+normal':

            radius = np.sqrt(3 * var)
            x = rand.uniform(low=-radius, high=radius, size=size)

            y = rand.normal(loc=0, scale=var, size=size)
            lmbd = self.noise_config['lmbd']
            return lmbd*x + (1-lmbd)*y

        elif name == 'linspace':
            return np.linspace(-var, var, size)

        else:
            raise ValueError('Unkown noise model: ' + name)


def check_mass(s_, bound_left, bound_right, p, eps):
    """True if bound has mass p in s_"""
    
    if len(s_) == 0:
        return 'done'
    
    above_count = len([i for i in s_ if i > bound_right])
    bellow_count = len([i for i in s_ if i < bound_left])
    

    m = (above_count + bellow_count) / len(s_) 
    
    if 1 - m > p + eps:
        return 'high'
    elif 1 - m < p - eps:
        return 'low'
    else:
        return 'done'

#Note a symmetric distribution is assumed
def find_mass(s, min_s, max_s, mean, p=.9, k=10, eps=1e-3):
        
    done = False
    
    boundl = min_s
    boundr = max_s
    
    lowerBoundl = lowerBoundr = mean
    
    i = 0
    while not done:
        
        mean_boundl = (boundl + lowerBoundl) / 2
        mean_boundr = (boundr + lowerBoundr) / 2
    
    
        alpha = i*k
        i+=1

        s_ = s[alpha: alpha+k]

        mass_value = check_mass(s_, mean_boundl, mean_boundr, p, eps)
        
        if mass_value == 'high':
            boundl = mean_boundl
            boundr = mean_boundr
            
        elif mass_value == 'low':
            lowerBoundl = mean_boundl
            lowerBoundr = mean_boundr
            
        else:
            done = True
        
        
    return mean_boundl, mean_boundr 

def to_density(x, bins=5, bounds=None):
    """"Turn into density based nb of bins"""

    p_x = np.histogram(x, bins=bins, density=True, range=bounds)[0]
    p_x = p_x / np.sum(p_x)
    return p_x  

def norm_entropy(x):
    """Normalized entropy"""
    return scipy.stats.entropy(x) / np.log(len(x))
        
def filter_by_bounds(x, bounds):
    """Filter an array based within given bounds, s.t. bounds[0] is a lower bound and bounds[1]"""
    assert bounds[1] >= bounds[0]
    return [i for i in x if i > bounds[0] and i < bounds[1]]


def l1_dist_to_uniform(x):
    """Returns the l1 distance between x and uniform distribution"""
    return np.linalg.norm(x - 1/len(x), ord=1)