# %%
import numpy as np 
import pandas as pd 
import time
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp

import numpy as np
import optax
#from IPython.display import display, Latex
from warnings import filterwarnings
filterwarnings('ignore')

# %%
np.random.seed(123)

N = 1000
J = 4
T = 50

# Generate the data
np.random.seed(123)
mu = np.array([-1.71, 0.44, -1.37, -0.91, -1.23]).reshape(-1, 1)
sigma = np.diag(np.array([3.22, 3.24, 2.87, 4.15, 1.38])).reshape(5, 5)

# sigma = np.diag(np.zeros(5)).reshape(5, 5) #This was done for testing, and we do recover the true parameters well enough with the MLE homogeneous approach

print("*********************************")
print("*******TRUE PARAMETERS*******")
print("*********************************")

print(mu, '\n')
print(sigma)

print("*********************************")



# generate the random parameters
betas = np.random.multivariate_normal(mu.flatten(), sigma, N)
betas_np = betas[:, :-1]
etas_np = betas[:, -1]

# %%
###Extract data

price_transition_states = pd.read_csv(r'price_transition_states.csv')
price_transition_matrix = pd.read_csv(r'transition_prob_matrix.csv')

# %%
price_transition_matrix_np = price_transition_matrix.to_numpy()
price_transition_states_np = price_transition_states.to_numpy()

# %%
def simulate_prices(states, transition, T):
    state_indices = np.arange(states.shape[0])

    price_simu = np.zeros((T, 6)) #create a matrix to store the simulated prices
    price_simu[0] = states[0] #fix the initial vector of prices
    
    for t in range(1, T):
        preceding_state = price_simu[t-1, :] #take the preceding state
        index_preceding_state = int(preceding_state[-1] - 1) #take the index of the preceding state (-1 for 0-indexing in Python)
        index_next_state = np.random.choice(state_indices, p=(transition[index_preceding_state, :].flatten())) #draw the next state
        price_simu[t, :] = states[index_next_state] #update the price vector and store it
    return price_simu


# %%

price_50_by_6 = simulate_prices(price_transition_states_np, price_transition_matrix_np, T)
prices_50_by_4 = price_50_by_6[:, :-2] #remove the indices column

# %%
## generate Utility data
utility_np = np.zeros((1+J, N, T)) # 1 for the outside option, J for the number of products
for t in range(T):
    for i in range(N):
        utility_np[0, i, t] = np.random.gumbel() #outside option, just a random noise
        utility_np[1:, i, t] = betas_np[i, :] + etas_np[i]*prices_50_by_4[t, :] + np.random.gumbel(size=J) #utility for the J products



# %%
choice_jnp = jnp.argmax(utility_np, axis=0) #argmax to get the choice number
prices_50_by_4_jnp = jnp.array(prices_50_by_4) #convert the prices to jnp array

# Note: I converted most of the numpy objects to jax.numpy objects. JAX is a library that allows us to do parallel computing, which speeds up the computation.

# * MLE assuming homogeneous $\Theta^h$

# %%

@jit
def choice_probas(theta):
    theta_jnp = jnp.array(theta)
    betas = theta_jnp[:-1]
    eta = theta_jnp[-1]
    v_1to4_utility = betas + eta * prices_50_by_4_jnp #for a candidate theta, compute systematic utility for each time period and product
    v_default = jnp.zeros((T, 1))
    v_utility = jnp.concatenate((v_default, v_1to4_utility), axis=1)

    # Compute choice probabilities with improved numerical stability
    log_sumexps = logsumexp(v_utility, axis=1)
    probas = jnp.exp(v_utility - log_sumexps[:, None]) #get the choice probabilities for each time period and product

    return probas

# %%
@jit
def likelihood(theta): #(log)-likelihood function
    probas_theta = choice_probas(theta) #get the choice probabilities for the candidate theta
    log_likelihood = jnp.sum(jnp.log(probas_theta[jnp.arange(T), choice_jnp])) #sum the log-probabilities of the observed choices
    return -log_likelihood

grad_likelihood = jit(grad(likelihood)) ## gradient of the likelihood function

# %%
def minimize_adam(f, grad_f, x0, norm=1e9, tol=0.1, lr=0.05, maxiter=1000, verbose=0, *args): ## generic adam optimizer
  """
  Generic Adam Optimizer. Specify a function f, a starting point x0, possibly a \n
  learning rate in (0, 1). The lower the learning rate, the more stable (and slow) the convergence.
  """
  tic = time.time()
  solver = optax.adam(learning_rate=lr)
  params = jnp.array(x0, dtype=jnp.float32)
  opt_state = solver.init(params)
  iternum = 0
  while norm > tol and iternum < maxiter :
    iternum += 1
    grad = grad_f(params, *args)
    updates, opt_state = solver.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    params = jnp.asarray(params, dtype=jnp.float32)
    norm = jnp.max(jnp.abs(grad))
    if verbose > 0:
      if iternum % 100 == 0:
        print(f"Iteration: {iternum}  Norm: {norm}  theta: {params}")
    if verbose > 1:
      print(f"Iteration: {iternum}  Norm: {norm}  theta: {params}")
  tac = time.time()
  if iternum == maxiter:
    print(f"Convergence not reached after {iternum} iterations. \nTime: {tac-tic} seconds. Norm: {norm}")
  else:
    print(f"Convergence reached after {iternum} iterations. \nTime: {tac-tic} seconds. Norm: {norm}")

  return params


print("*********************************")
print("******  MLE HOMOGENEOUS   *******")
print("*********************************")

# %%
theta_MLE_homo = minimize_adam(likelihood, grad_likelihood, jnp.ones(5), lr=0.1, verbose=1, tol=0.01, maxiter=1500)
print("*********************************")

# %%

# %% [markdown]
# Formula to retrieve MLE Standard Errors
# 
# \begin{align*}
# \nabla l_{it} &= \frac{\partial l_{it} (\widehat{\theta})}{\partial \widehat{\theta}} \tag{column vector} \\
# SE(\widehat{\theta}) &= diag\Bigg[{\sqrt{\Big( \frac{1}{NT}\sum_{i=1}^N \sum_{t=1}^T \nabla l_{it} \cdot \nabla l_{it}' \Big)^{-1}}}\Bigg]
# \end{align*}

# %%
###Computation of standard errors

@jit
def likelihood_it(theta, i, t):
    """
    Computes the likelihood for an individual observation
    """
    probas_theta = choice_probas(theta)
    likelihood_it = jnp.log(probas_theta[t, choice_jnp[i, t]])
    return likelihood_it

grad_likelihood_it = jit(grad(likelihood_it)) ### Takes the gradient of the individual likelihood

@jit
def outer_grad_likelihood(theta, i, t):
    """
    Takes the outer product (column vector x row vector) of the gradient of the individual likelihood
    """
    grad_it = (grad_likelihood_it(theta, i, t)).reshape(-1, 1) 
    return grad_it@grad_it.T


#computes the outer product above for each individual and time period
grad_likelihood_it_vec = vmap(vmap(outer_grad_likelihood, in_axes=(None, 0, None)), in_axes=(None, None, 0)) 

@jit
def compute_standard_errors(theta):
    sum_outers = (1/(N*T))*(jnp.sum(grad_likelihood_it_vec(theta, jnp.arange(N), jnp.arange(T)), axis=(0, 1)))
    return jnp.diag(jnp.sqrt(jnp.linalg.inv(sum_outers)))

# %%
se = compute_standard_errors(theta_MLE_homo)
print("*********************************")
print("*********************************")
print(f'Theta: {theta_MLE_homo}')
print(f'se: {se}')
print("*********************************")

# %% [markdown]
# * MLE assuming two classes

# %%
###Two classes: instead of estimating theta, we want to estimate the weights phi_1, phi_2 of each class (?)
theta_k1 = theta_MLE_homo - se
theta_k2 = theta_MLE_homo + se
print(theta_k1)
print(theta_k2)

# %%
@jit
def choice_probas_2classes(phi1):
    phi2 = 1 - phi1
    probas_k1 = choice_probas(theta_k1)
    probas_k2 = choice_probas(theta_k2)
    probas = phi1 * probas_k1 + phi2 * probas_k2
    return probas

# %%
@jit
def likelihood_2classes(phi1): #(log)-likelihood function
    probas_theta = choice_probas_2classes(phi1) #get the choice probabilities for the candidate theta
    log_likelihood = jnp.sum(jnp.log(probas_theta[jnp.arange(T), choice_jnp])) #sum the log-probabilities of the observed choices
    return -log_likelihood


grad_likelihood_2classes = jit(grad(likelihood_2classes))

# %%
weight_1 = minimize_adam(likelihood_2classes, grad_likelihood_2classes, 0.5, verbose=False)
print("*********************************")
print("******* 2 CLASSES MLE **************")
print("*********************************")

print(f'Theta_1: {theta_k1}')
print(f'Theta_2: {theta_k2}')
print(f'Weights: {weight_1.item(), 1-weight_1.item()}')
print("*********************************")

# %% [markdown]
# * MLE assuming that $\Theta^h$ has a normal distribution across the households 

# %%
# Define the number of Monte Carlo draws
S = 5000  # Number of simulation draws

# Generate random draws for Monte Carlo integration
key = jax.random.PRNGKey(123)
mc_draws = jax.random.normal(key, (S, 5))  # 5 parameters (4 betas + 1 eta)

# %%
@jit
def mixed_logit_likelihood(theta):
    mu = theta[:5]  # Mean of the random parameters
    sigma = jnp.exp(theta[5:])
    betas_eta = mu + mc_draws * jnp.sqrt(sigma)
    
    probas_theta = vmap(choice_probas)(betas_eta)  # Shape: (S, T, 5)
    probas_theta_avg = jnp.mean(probas_theta, axis=0)  # Shape: (T, 5)
    
    log_likelihood = jnp.sum(jnp.log(probas_theta_avg[jnp.arange(T), choice_jnp]))  # Sum over T
    return -log_likelihood

grad_mixed_logit_likelihood = jit(grad(mixed_logit_likelihood))

# %%
x_start = jnp.concatenate((theta_MLE_homo, jnp.ones(5)))
print("**************************************")
print("****** Mixed Logit Estimation ********")
print("**************************************")

theta_mixed_logit = minimize_adam(mixed_logit_likelihood, grad_mixed_logit_likelihood, x_start, lr=0.1, maxiter=5000, verbose=1, tol=0.2)
print("*********************************")

# %%
### Estimate of Theta: not too bad, and the signs are correct
print(f'Theta: {theta_mixed_logit[:5]}')
print("*********************************")


# %%
### Estimate of the variance-covariance matrix. Seems hard to recover, probably also causing a larger biase in the point estimate.
print("*********************************")
print("Variance-Covariance matrix: \n")
print(jnp.diag(jnp.exp(theta_mixed_logit[5:])))
print("*********************************")

# %% [markdown]
# * Sanity Check: what if we assume the correct sigma and only estimate for theta ?
# 
# Conclusion: we do recover an unbiased estimate of $\mu$ if we are able to assume to know $\sigma$ already.

# %%
@jit
def mixed_logit_likelihood_correct(theta):
    mu = theta.flatten()  # Mean of the random parameters
    betas_eta = mu + mc_draws * jnp.sqrt(jnp.diag(sigma)) 

    def compute_probas(draw):
        return choice_probas(draw)  # Shape: (T, 5)
    
    probas_theta = vmap(compute_probas)(betas_eta)  # Shape: (S, T, 5)
    probas_theta_avg = jnp.mean(probas_theta, axis=0)  # Shape: (T, 5)
    
    log_likelihood = jnp.sum(jnp.log(probas_theta_avg[jnp.arange(T), choice_jnp]))  # Sum over T
    return -log_likelihood

grad_mixed_logit_likelihood_correct = jit(grad(mixed_logit_likelihood_correct))


# %%

print("************************************************************\n")
print("*****  Sanity Check: Mixed logit under true sigma  *********")
print("************************************************************\n")


theta_mixed_logit_correct = minimize_adam(mixed_logit_likelihood_correct, 
                                          grad_mixed_logit_likelihood_correct, 
                                          theta_MLE_homo, 
                                          lr=0.05, maxiter=5000, verbose=1, tol=0.1)

# %%

print(f'Theta: {theta_mixed_logit_correct}')

print("************************************************************\n")
