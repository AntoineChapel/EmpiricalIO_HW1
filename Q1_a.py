### Demand Estimation with Individual-Level Data

## Part a, ii

import numpy as np
import pandas as pd

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
choice_np = np.argmax(utility_np, axis=0)

print(choice_np)

print(choice_np.shape)


## part b: first-order Markov dependence



