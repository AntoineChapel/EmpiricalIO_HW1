# %%
import numpy as np
import pandas as pd

# %%
N = 1000
T = 50 

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
prices_150_by_4 = price_150_by_6[:, :-2] 

# %%
beta = np.array([-1.71,  0.44, -1.37, -0.91])
gamma = -1.23

# %%
def elasticities(beta, gamma, prices):
    vjs = beta + gamma * np.mean(prices, axis=0)
    Pjs = np.exp(vjs) / (1+np.sum(np.exp(vjs)))
    #own price elasticities:
    e_jj = (gamma * np.mean(prices, axis=0) * (1-Pjs)).reshape(1,- 1)
    e_jk = ((-gamma) * np.mean(prices, axis=0) * Pjs).reshape(1, -1)
    return np.vstack((e_jj, e_jk))

# %%
#Homogeneous
beta_MLE_hat = np.array([-1.550132, -0.23989092, -1.3096188, -0.9299984])
gamma_MLE_hat = -0.36560854

# %%
#Mixed Logit:
beta_MLE_het = np.array([-0.9368873, 0.01981145, -0.6546789, -1.1986973])
gamma_MLE_het = -1.2291056

# %%
elasticities(beta, gamma, prices_150_by_4)

# %%
elasticities(beta_MLE_hat, gamma_MLE_hat, prices_150_by_4)

# %%
elasticities(beta_MLE_het, gamma_MLE_het, prices_150_by_4)


