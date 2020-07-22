import pandas as pd
import numpy as np
import misc

class BinaryAM:
    """ Binary additive model without a cofunder (i.e. x -> y)
        x = noise_x
        y = f(x) + noise_y
    """

    def __init__(self, x_noise_config, y_noise_config, f, size):

        # Save function to be applied

        self.f = f

        # Create noise samplers for x and y

        self.x_noise = misc.NoiseFactory(x_noise_config, size)
        self.y_noise = misc.NoiseFactory(y_noise_config, size)

        # Structural equation model (SEM)

        self.x = self.x_noise.sample()
        self.y = self.f(self.x) + self.y_noise.sample()

        # Save data

        self.x_label = 'x'
        self.y_label = 'y'

        self.data = {
            self.x_label : self.x,
            self.y_label: self.y
        }

        self.data = pd.DataFrame(data=self.data)


class RegressionBasedDiscovery:
    """A simple class to perform causal discovery via regression and goodness of fit
    """

    def __init__(self, metric_name='MI'):

        self.method_name = metric_name
        self.metric = misc.Metrics.metric_from_str(metric_name)

    def compute_resiudal(self, x, y):

        model_direct = misc.LinearModel()
        model_direct.fit(x, y)

        return model_direct.predict(x) - y

    def discover(self, x, y, method='MI'):

        # Compute residuals for both the direct and reverse model

        res_direct = self.compute_resiudal(x, y)
        res_reverse = self.compute_resiudal(y, x)


        if method == 'MI':

            # Compute an independence score 

            direct_score = self.metric(x, res_direct)
            reverse_score = self.metric(y, res_reverse)

            # We are measuring the "degree" of independence
            # between input data and residual: lower is better proof
            # of causality direction

            return direct_score < reverse_score 

        elif method == 'EST':
            direct_score = sum(res_direct) / len(res_direct)
            reverse_score = sum(res_reverse) / len(res_reverse)

            print('direct: ', direct_score)
            print('reverse: ', reverse_score)

            return abs(direct_score) < abs(reverse_score), res_direct, res_reverse

        else:
            raise Exception('Unkown method:', method) 


class OnlineDiscovery:

    def __init__(self):
        pass

    @staticmethod
    def discovery_sim(x, y, eta=.02):
    
        m = 0
        s = 0
        
        M = [m]
        S = [s]
        
        w = 0
        W = [w]
        
        w_mean = 0
        W_mean = [w_mean]
        
        R = []
        
        for i in range(len(x)):
            
            # Online regression
            y_ = w*x[i]
            residual = y_ - y[i]
            w -= eta*residual*x[i]
            W.append(w)
            
            w_mean = np.mean(W) # take average of w estimate to compute residual
            residual_ =  w_mean*x[i] - y[i]
            W_mean.append(w_mean)
            
            # compute running variance and mean of residual
            k = i
            delta = residual_ - m
            m += delta / (k + 2) # to check
            s += delta * (residual_ - m)
            
            M.append(m)
            S.append(s/(k+1))
            R.append(residual_)

        data_stats = {
            'mean': M,
            'var': S,
            'W_mean': W_mean,
            'residual': R
        }
        
        return data_stats