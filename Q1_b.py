import numpy as np 
import pandas as pd 
import time
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp

import numpy as np
import optax
from IPython.display import display, Latex
from warnings import filterwarnings
filterwarnings('ignore')



np.random.seed(123)

N = 1000
J = 4
T = 150

# Generate the data
np.random.seed(123)
mu = np.array([-1.7, 0.44, -1.37, -0.91, -1.23, 1]).reshape(-1, 1)
sigma = np.diag(np.array([3.22, 3.24, 2.87, 4.15, 1.38, 1])).reshape(6, 6)
# sigma = np.diag(np.zeros(6)).reshape(6, 6) #done for testing purpose, theta_MLE_homo does recover mu

print(mu, '\n')
print(sigma)


# generate the random parameters
betas = np.random.multivariate_normal(mu.flatten(), sigma, N)
betas_np = betas[:, :-2]
etas_np = betas[:, -2]
gammas_np = betas[:, -1]



price_transition_states = pd.read_csv(r'price_transition_states.csv')
price_transition_matrix = pd.read_csv(r'transition_prob_matrix.csv')
price_transition_matrix_np = price_transition_matrix.to_numpy()
price_transition_states_np = price_transition_states.to_numpy()


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




price_150_by_6 = simulate_prices(price_transition_states_np, price_transition_matrix_np, T)
prices_150_by_4 = price_150_by_6[:, :-2] #remove the indices column


## generate baseline utility data (no loyalty)
utility_np = np.zeros((T, 1+J, N)) # 1 for the outside option, J for the number of products
for t in range(1, T):
    for i in range(N):
        utility_np[t, 0, i] = np.random.gumbel() #outside option, just a random noise
        utility_np[t, 1:, i] = betas_np[i, :] + etas_np[i]*prices_150_by_4[t, :] + np.random.gumbel(size=J) #utility for the J products

#utility_np_orig = utility_np.copy()


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


#utility_orig_jnp = jnp.array(utility_np_orig[100:, :, :])  #50 x 5 x 1000
utility_jnp = jnp.array(utility_np[100:, :, :])            #50 x 5 x 1000
choice_jnp = jnp.argmax(utility_np, axis=1)[100:, :]       #50 x 1000
prices_50_by_4_jnp = jnp.array(prices_150_by_4[100:, :])   #50 x 4
state_matrix_jnp = jnp.array(state_matrix[100:, :])        #50 x 1000


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


@jit
def likelihood(theta): #(log)-likelihood function
    probas_theta = ccp(theta) #get the choice probabilities for the candidate theta
    log_likelihood = jnp.sum(jnp.log(probas_theta[jnp.arange(50)[:, None], state_matrix_jnp, choice_jnp])) #sum the log-probabilities of the observed choices
    return -log_likelihood

grad_likelihood = jit(grad(likelihood)) ## gradient of the likelihood function


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


theta_MLE_homo = minimize_adam(likelihood, grad_likelihood, jnp.zeros(6), lr=0.01, verbose=1, maxiter=5000)



