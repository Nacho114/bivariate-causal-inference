import pandas as pd
import numpy as np
import misc


class GPsampler:
    """ A class to sample from a GP
    """

    def __init__(self, gamma):
        self.gamma = gamma

    def get_func(self):

        def gp_func(xs, gamma):
            mean = np.zeros(len(xs))
            gram = self.gram_matrix(xs, gamma)

            ys = np.random.multivariate_normal(mean, gram)

            return ys

        return lambda x: gp_func(x, self.gamma)

    def rbf_kernel(self, x1, x2, gamma=1):
        return np.exp(-1 * ((x1-x2) ** 2) / (2*gamma))

    def gram_matrix(self, xs, gamma):
        return [[self.rbf_kernel(x1,x2, gamma) for x2 in xs] for x1 in xs]

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




def am_from_param(param):
    binary_am = BinaryAM(param['Nx'], param['Ny'], 
                            param['f'], param['nb_samples'])
    x = binary_am.x
    y = binary_am.y

    return x, y