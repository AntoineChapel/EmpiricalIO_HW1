### Demand Estimation with Individual-Level Data

## Part a, ii

import numpy as np

N = 1000
J = 4
T = 50

# Generate the data
np.random.seed(123)
mu = np.array([-1.71, 0.44, -1.37, -0.91, -1.23]).reshape(-1, 1)
sigma = np.diag(np.array([3.22, 3.24, 2.87, 4.15, 1.38])).reshape(5, 5)
print(mu, '\n')
print(sigma)


# generate the random parameters
betas = np.random.multivariate_normal(mu.flatten(), sigma, N)
print(betas.shape)


## part b: first-order Markov dependence



