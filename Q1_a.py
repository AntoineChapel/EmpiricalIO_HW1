### Demand Estimation with Individual-Level Data

## Part a, ii

import numpy as np 
import pandas as pd 
import time
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp
import optax



N = 1000
J = 4
T = 50

# Generate the data
np.random.seed(123)
mu = np.array([-1.71, 0.44, -1.37, -0.91, -1.23]).reshape(-1, 1)
sigma = np.diag(np.array([3.22, 3.24, 2.87, 4.15, 1.38])).reshape(5, 5)


# generate the random parameters
betas = np.random.multivariate_normal(mu.flatten(), sigma, N)

betas_np = betas[:, :-1]
etas_np = betas[:, -1]


## generate prices
price_transition_states = pd.read_csv(r'price_transition_states.csv')
price_transition_matrix = pd.read_csv(r'transition_prob_matrix.csv')

price_transition_matrix_np = price_transition_matrix.to_numpy()
price_transition_states_np = price_transition_states.to_numpy()


def simulate_prices(states, transition, T):
    state_indices = np.arange(states.shape[0])
    price_simu = np.zeros((T, 6))
    price_simu[0] = states[0]
    for t in range(1, T):
        preceding_state = price_simu[t-1, :]
        index_preceding_state = int(preceding_state[-1] - 1)
        index_next_state = np.random.choice(state_indices, p=(transition[index_preceding_state, :].flatten()))
        price_simu[t, :] = states[index_next_state]
    return price_simu


price_50_by_6 = simulate_prices(price_transition_states_np, price_transition_matrix_np, T)
prices_50_by_4 = price_50_by_6[:, :-2]


## generate Utility data
utility_np = np.zeros((J+1, N, T))
for t in range(T):
    for i in range(N):
        utility_np[0, i, t] = np.random.gumbel()
        utility_np[1:, i, t] = betas_np[i, :] + etas_np[i]*prices_50_by_4[t, :].flatten() + np.random.gumbel(size=J)

## generate choice data
choice_np = jnp.argmax(utility_np, axis=0)

prices_50_by_4_jnp = jnp.array(prices_50_by_4)


@jit
def choice_probas(theta):
    theta_jnp = jnp.array(theta)
    betas = theta_jnp[:-1]
    eta = theta_jnp[-1]
    v_1to4_utility = betas + eta * prices_50_by_4_jnp
    v_default = jnp.zeros((T, 1))
    v_utility = jnp.concatenate((v_default, v_1to4_utility), axis=1)

    # Compute choice probabilities with improved numerical stability
    log_sumexps = logsumexp(v_utility, axis=1)
    probas = jnp.exp(v_utility - log_sumexps[:, None])

    return probas


@jit
def likelihood(theta): #(log)-likelihood function
    probas_theta = choice_probas(theta)
    log_likelihood = jnp.sum(jnp.log(probas_theta[jnp.arange(T), choice_np]))
    return -log_likelihood

grad_likelihood = jit(grad(likelihood)) ## gradient of the likelihood function



def minimize_adam(f, x0, norm=1e9, tol=0.1, lr=0.05, maxiter=1000):
  tic = time.time()
  solver = optax.adam(learning_rate=lr)
  params = jnp.array(x0, dtype=jnp.float32)
  opt_state = solver.init(params)
  iternum = 0
  while norm > tol and iternum < maxiter :
    iternum += 1
    grad = grad_likelihood(params) #personally computed gradients
    updates, opt_state = solver.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    params = jnp.asarray(params, dtype=jnp.float32)
    norm = jnp.max(jnp.abs(grad))
    print(f"Iteration: {iternum}  Norm: {norm}  theta: {params}")
  tac = time.time()
  if iternum == maxiter:
    print(f"Convergence not reached after {iternum} iterations. \nTime: {tac-tic} seconds. Norm: {norm}")
  else:
    print(f"Convergence reached after {iternum} iterations. \nTime: {tac-tic} seconds. Norm: {norm}")

  return params


theta_MLE_homogeneous = minimize_adam(likelihood, np.ones(5), lr=0.2)
print(f'*********************************** \n \n theta MLE under homogeneous assumption: {theta_MLE_homogeneous} \n \n ***********************************')








