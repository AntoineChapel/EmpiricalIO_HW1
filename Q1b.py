# %%
import numpy as np 
import pandas as pd 
import time
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp

import optax
from warnings import filterwarnings
filterwarnings('ignore')

# %%
np.random.seed(123)

N = 1000
J = 4
T = 150

# Generate the data
np.random.seed(123)
mu = np.array([-1.71, 0.44, -1.37, -0.91, -1.23, 1]).reshape(-1, 1)
sigma = np.diag(np.array([3.22, 3.24, 2.87, 4.15, 1.38, 1])).reshape(6, 6)

print("*****   True Parameters     *******")
print(mu, '\n')
print(sigma)
print("**********************************")

# generate the random parameters
betas = np.random.multivariate_normal(mu.flatten(), sigma, N)
betas_np = betas[:, :-2]
etas_np = betas[:, -2]
gammas_np = betas[:, -1]

# %%
price_transition_states = pd.read_csv(r'price_transition_states.csv')
price_transition_matrix = pd.read_csv(r'transition_prob_matrix.csv')
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
price_150_by_6 = simulate_prices(price_transition_states_np, price_transition_matrix_np, T)
prices_150_by_4 = price_150_by_6[:, :-2] #remove the indices column

# %%
## generate baseline utility data (no loyalty)
utility_np = np.zeros((T, 1+J, N)) # 1 for the outside option, J for the number of products
for t in range(1, T):
    for i in range(N):
        utility_np[t, 0, i] = np.random.gumbel() #outside option, just a random noise
        utility_np[t, 1:, i] = betas_np[i, :] + etas_np[i]*prices_150_by_4[t, :] + np.random.gumbel(size=J) #utility for the J products

#utility_np_orig = utility_np.copy()

# %%
### add loyalty
state_matrix = np.zeros((T, N), dtype=int) #the state at time 0 is 0
state_matrix[1, :] = np.argmax(utility_np[0, :, :], axis=0) #initialize the state simulation

for t in range(1, T-1):
    for i in range(N):
        state_it = state_matrix[t, i]
        for j in range(1, J+1): #exclude the outside option
            utility_np[t, j, i] += gammas_np[i] * (j == state_it)
        choice = np.argmax(utility_np[t, :, i])
        if choice==0:
            state_matrix[t+1, i] = state_it ### if the outside option is chosen, the state remains the same
        else:
            state_matrix[t+1, i] = choice ### if a product is chosen, the state is updated

# %%
#utility_orig_jnp = jnp.array(utility_np_orig[100:, :, :])  #50 x 5 x 1000
utility_jnp = jnp.array(utility_np[100:, :, :])            #50 x 5 x 1000
choice_jnp = jnp.argmax(utility_np, axis=1)[100:, :]       #50 x 1000
prices_50_by_4_jnp = jnp.array(prices_150_by_4[100:, :])   #50 x 4
state_matrix_jnp = jnp.array(state_matrix[100:, :])        #50 x 1000

# %% [markdown]
# ### i: Homogeneous, Two classes, and Mixed Logit
# * Homogeneous

# %%
@jit
def ccp(theta):
    """
    Compute the choice probabilities for each time period and product for a given theta, for each possible state
    There are 4 possible states (individuals are never in state 0). For a given theta, compute the choice probabilities for each state
    Should a return a (T, J, J+1) array. That is, for each period, for each possible state, the choice probas
    """
    theta_jnp = jnp.array(theta).flatten()
    betas = theta_jnp[:-2]
    eta = theta_jnp[-2]
    gamma = theta_jnp[-1]
    
    #possible states: 0, 1, 2, 3, 4 
    v_1to4_utility_state0 = (betas + eta * prices_50_by_4_jnp).reshape(50, 1, 4)
    v_1to4_utility_state1to4 = (betas + eta * prices_50_by_4_jnp).reshape(50, 1, 4) + gamma * jnp.eye(4)
    v_utility = jnp.concatenate((v_1to4_utility_state0, v_1to4_utility_state1to4), axis=1)
    v_default = jnp.zeros((50, 5, 1))
    v_utility_full = jnp.concatenate((v_default, v_utility), axis=2)

    # Compute choice probabilities 
    log_sumexps = logsumexp(v_utility_full, axis=2, keepdims=True)
    probas = jnp.exp(v_utility_full - log_sumexps) #get the choice probabilities for each time period and product

    return probas
ccp_vec = vmap(ccp)

# %%
@jit
def likelihood(theta): #(log)-likelihood function
    probas_theta = ccp(theta) #get the choice probabilities for the candidate theta
    log_likelihood = jnp.sum(jnp.log(probas_theta[jnp.arange(50)[:, None], state_matrix_jnp, choice_jnp])) #sum the log-probabilities of the observed choices
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
        print(f"Iteration: {iternum}  Norm: {norm}  theta: {jnp.round(params, 2)}")
    if verbose > 1:
      print(f"Iteration: {iternum}  Norm: {norm}  theta: {jnp.round(params, 2)}")
  tac = time.time()
  if iternum == maxiter:
    print(f"Convergence not reached after {iternum} iterations. \nTime: {tac-tic} seconds. Norm: {norm}")
  else:
    print(f"Convergence reached after {iternum} iterations. \nTime: {tac-tic} seconds. Norm: {norm}")

  return params
print("**********************************")
print("**** Homogeneous Theta MLE  ******")
print("**********************************")


# %%
theta_MLE_homo = minimize_adam(likelihood, grad_likelihood, jnp.zeros(6), lr=0.01, verbose=0, maxiter=5000)
print("**********************************")

# %%

print(f'Theta: {theta_MLE_homo}')
# %%
###Computation of standard errors

@jit
def likelihood_it(theta, i, t):
    """
    Computes the likelihood for an individual observation
    """
    probas_theta = ccp(theta)
    likelihood_it = jnp.log(probas_theta[t, state_matrix_jnp[t, i], choice_jnp[t, i]])
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
    sum_outers = (1/(N*50))*(jnp.sum(grad_likelihood_it_vec(theta, jnp.arange(N), jnp.arange(T)), axis=(0, 1)))
    return jnp.diag(jnp.sqrt(jnp.linalg.inv(sum_outers)))

# %%
se = compute_standard_errors(theta_MLE_homo)
print(f'Standard errors: {se}')
print("**********************************")


# %% [markdown]
# * MLE 2 classes

# %%
###Two classes: instead of estimating theta, we want to estimate the weights phi_1, phi_2 of each class (?)
print("**********************************")
print("********   Two Classes   *********")
print("**********************************")

theta_k1 = theta_MLE_homo - se
theta_k2 = theta_MLE_homo + se
print(theta_k1)
print(theta_k2)
print("**********************************")

# %%
@jit
def choice_probas_2classes(phi1):
    phi2 = 1 - phi1
    probas_k1 = ccp(theta_k1)
    probas_k2 = ccp(theta_k2)
    probas = phi1 * probas_k1 + phi2 * probas_k2
    return probas

# %%
@jit
def likelihood_2classes(phi1): #(log)-likelihood function
    probas_theta = choice_probas_2classes(phi1) #get the choice probabilities for the candidate theta
    log_likelihood = jnp.sum(jnp.log(probas_theta[jnp.arange(50)[:, None], state_matrix_jnp, choice_jnp])) #sum the log-probabilities of the observed choices
    return -log_likelihood


grad_likelihood_2classes = jit(grad(likelihood_2classes))

# %%
weight_1 = minimize_adam(likelihood_2classes, grad_likelihood_2classes, 0.5, verbose=False)
print(f'Theta 1: {theta_k1}')
print(f'Theta 2: {theta_k2}')
print(f'Weights: {weight_1.item(), 1-weight_1.item()}')

print("**********************************")


# %% [markdown]
# * MLE assuming that $\Theta^h$ has a normal distribution across the households 

# %% [markdown]
# First: sanity check. Let's assume we know $\sigma$: do we recover the proper mu under the correct distributional assumption ?

# %%
# Define the number of Monte Carlo draws
S = 1000  # Number of simulation draws

# Generate random draws for Monte Carlo integration
key = jax.random.PRNGKey(123)
mc_draws = jax.random.normal(key, (S, 6))  # 5 parameters (4 betas + 1 eta)

# %%
@jit
def mixed_logit_likelihood_correct(theta):
    mu = theta.flatten()  # Mean of the random parameters
    betas_eta = mu + mc_draws * jnp.sqrt(jnp.diag(sigma))
    
    probas_theta = ccp_vec(betas_eta)  # Shape: (S, T, 6)
    probas_theta_avg = jnp.mean(probas_theta, axis=0)  # Shape: (T, 6)
    log_likelihood = jnp.sum(jnp.log(probas_theta_avg[jnp.arange(50)[:, None], state_matrix_jnp, choice_jnp])) 
    return -log_likelihood

grad_mixed_logit_likelihood_correct = jit(grad(mixed_logit_likelihood_correct))


print("************************************************************")
print("***************** Mixed Logit Theta MLE  *******************")
print("************************************************************")
print("*****  Sanity Check: Mixed Logit under true sigma  *********")
print("************************************************************")

# %%
### This does not converge, the bias is high
theta_mixed_logit_correct = minimize_adam(mixed_logit_likelihood_correct, 
                                          grad_mixed_logit_likelihood_correct, 
                                          theta_MLE_homo, 
                                          lr=0.05, maxiter=5000, verbose=0, tol=0.1)
print(theta_mixed_logit_correct)
print("***********************************************************")

# %% [markdown]
# There is bias. Maybe this is to be expected given the instructions of the exercise.

# %%
@jit
def mixed_logit_likelihood(theta):
    mu = theta[:6]  # Mean of the random parameters
    sigma = jnp.exp(theta[6:])
    betas_eta = mu + mc_draws * sigma
    probas_theta = ccp_vec(betas_eta)  # Shape: (S, T, 6)
    probas_theta_avg = jnp.mean(probas_theta, axis=0)  # Shape: (T, 6)
    log_likelihood = jnp.sum(jnp.log(probas_theta_avg[jnp.arange(50)[:, None], state_matrix_jnp, choice_jnp])) 
    return -log_likelihood

grad_mixed_logit_likelihood = jit(grad(mixed_logit_likelihood))

# %%
### This does not seem to converge well and the bias is high.
print("****************************************************************")
print("********  Mixed Logit: estimate theta and variance   ***********")
print("****************************************************************")

x_start = jnp.concatenate((theta_MLE_homo, jnp.zeros(6)))
theta_mixed_logit = minimize_adam(mixed_logit_likelihood, grad_mixed_logit_likelihood, x_start, lr=0.3, maxiter=1000, verbose=1, tol=0.2)

print(f'Theta: {theta_mixed_logit[:6]}')
print("Variance-covariance matrix:")
print(jnp.diag(jnp.exp(theta_mixed_logit[6:])))
print("****************************************************************")

# %%
theta_mixed_logit[:6] - mu.flatten() #quite strong bias

# %% [markdown]
# ### ii: Re-initialize the initial state and re-run the state computation given choice data
print("****************************************************************")
print("************  Other Approach: re-initialize state  *************")
print("****************************************************************")

# %%
state_matrix_np2 = np.array(jnp.vstack((jnp.zeros((1, 1000)), state_matrix_jnp[1:, :])))

# %%
###rerun the state based on choices. Takes 9 secs.
for i in range(N):
    for t in range(49):
        choice_it = choice_jnp[t, i]
        if choice_it != 0:
            state_matrix_np2[t+1, i] = choice_it
        else:
            state_matrix_np2[t+1, i] = state_matrix_np2[t, i]
state_matrix_jnp2 = jnp.array(state_matrix_np2).astype(int)

# %% [markdown]
# Again, sanity check. Let's assume sigma is known, and see if we can recover the true mu.

# %%
@jit
def mixed_logit_likelihood_correct2(theta):
    mu = theta.flatten()  # Mean of the random parameters
    betas_eta = mu + mc_draws * jnp.sqrt(jnp.diag(sigma))
    
    probas_theta = ccp_vec(betas_eta)  # Shape: (S, T, 6)
    probas_theta_avg = jnp.mean(probas_theta, axis=0)  # Shape: (T, 6)
    log_likelihood = jnp.sum(jnp.log(probas_theta_avg[jnp.arange(50)[:, None], state_matrix_jnp2, choice_jnp])) 
    return -log_likelihood

grad_mixed_logit_likelihood_correct2 = jit(grad(mixed_logit_likelihood_correct2))

# %%
### Much closer, but still not there, and one sign is off.

print("****************************************************************")
print("********   Sanity Check: Estimate Theta under true sigma *******")
print("****************************************************************")

theta_mixed_logit_correct2 = minimize_adam(mixed_logit_likelihood_correct2, 
                                          grad_mixed_logit_likelihood_correct2, 
                                          theta_MLE_homo, 
                                          lr=0.05, maxiter=5000, verbose=0, tol=0.1)
print(f'Theta Mixed logit under correct distribution: {theta_mixed_logit_correct2}')
print("****************************************************************")
print("****************************************************************")

# %% [markdown]
# Actual Mixed Logit estimtion of $\Theta$ and variance

# %%
@jit
def mixed_logit_likelihood2(theta):
    mu = theta[:6]  # Mean of the random parameters
    sigma = jnp.exp(theta[6:]) 
    betas_eta = mu + mc_draws * sigma
    
    probas_theta = ccp_vec(betas_eta)  # Shape: (S, T, 6)
    probas_theta_avg = jnp.mean(probas_theta, axis=0)  # Shape: (T, 6)
    
    log_likelihood = jnp.sum(jnp.log(probas_theta_avg[jnp.arange(50)[:, None], state_matrix_jnp2, choice_jnp])) 
    return -log_likelihood

grad_mixed_logit_likelihood2 = jit(grad(mixed_logit_likelihood2))

# %%
### This does not converge properly and as the norm reduces, the bias is still very high

x_start = jnp.concatenate((theta_MLE_homo, jnp.zeros(6)))
theta_mixed_logit2 = minimize_adam(mixed_logit_likelihood2, grad_mixed_logit_likelihood2, x_start, lr=0.3, maxiter=1000, verbose=1, tol=0.2)
print("****************************************************************")

# %% [markdown]
# Bias:
print(f'Theta under state reinitialization: {theta_mixed_logit2[:6]}')
print(f'Variance-covariance matrix:')
print(jnp.diag(jnp.exp(theta_mixed_logit2[6:])))
# %%

theta_mixed_logit[:6] - mu.flatten()

# %%
theta_mixed_logit2[:6] - mu.flatten()