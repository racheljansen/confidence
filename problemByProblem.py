# PARTICLE FILTER VERSION OF MODEL (confidence judgments)

import random
import numpy as np


def sigmoid(x): # sigmoid function will come in handy
    return 1/(1 + np.exp(-x))

# The probability of getting 1 in the likelihood given theta and beta and epsilon is:
# (1- $\epsilon$) $\cdot $ prob correct + $\epsilon$ $\cdot $ prob incorrect

def loglik(theta,beta,eps):
    
    """
    takes in a theta and beta value and uses a sigmoid function to compute likelihood of answering correctly
    1 / 1 + exp (-(theta-beta))
    also takes in a probability epsilon of making an incorrect inference
    """
    assert eps >= 0, "Epsilon is a probability and must be between 0 and 1!"
    assert eps <= 1, "Epsilon is a probability and must be between 0 and 1!"


    return np.log( (1-eps) * (1 / (1 + np.exp(-(theta - beta)))) # 1-epsilon times prob right 
                  + eps * (1 / (1 + np.exp(theta - beta))) ) # plus epsilon times prob wrong

def loglikfail(theta,beta,eps): # 1-loglik
    
    """
    takes in a theta and beta value and uses a sigmoid function to compute likelihood of answering incorrectly
    1 - (1 / 1 + exp (-(theta-beta)))
    """
    assert eps >= 0, "Epsilon is a probability and must be between 0 and 1!"
    assert eps <= 1, "Epsilon is a probability and must be between 0 and 1!"
    
    return np.log( (1-eps) * (1 / (1 + np.exp(theta - beta)))
                 + eps * (1 / (1 + np.exp(-(theta - beta)))) )

def ess(weights):
    """
    Effective Sample Size (ESS) is a measure of the variance of the weights
    """
    ess = 0
    for w in weights:
        ess += w**2
    return 1/ess
        
def seqMC(betas, mut, sigt, nparticles, eps, test, threshold, sigt_dyn):
    
    """
    input:
    betas: initial betas (vector) - number of vecors of betas of length 'test'
    mut,sigt: mean and standard deviation of prior
    nparticles: number of particles
    eps: epsilon
    test: vectors of 1's and 0's that represent correct vs. incorrect
    threshold: value for resampling criterion
    sigt_dyn: the standard deviation on the dynamics of theta
    
    output:
    thetas: list of particles (list of vectors containing all particles at each time point)
    weights: list of weights corresponding to each particle
    
    
    """
    len(test) == len(betas), "Should have same number of betas, and problems"
    
    weights = [ [] for i in range(len(test)) ] # empty list of empty lists
    thetas = [ [] for i in range(len(test)) ]
    ess_list = [] # storing effective sample size values (to tune threshold parameter)
    resample = [] # storing whether resampling occurred (also to turn threshold parameter)
    
    for t in range(len(test)): # start by looping over times
        weights_t = [] # empty list for weights
        
        # (1) iteration 1: generate from the prior
        if t == 0: 
            # (a) sample a set of n particles (theta_1) from the prior
            thetas_t = np.random.normal(mut,sigt,nparticles)
            
            # (b) compute weights which are equal to the likelihood 
             
            if test[t] == 1: # if answered correctly
                for theta in thetas_t:
                    weights_t.append(loglik(theta,betas[t],eps)) 
            else: # in test[t] == 0 - if incorrect on that problem
                for theta in thetas_t:
                    weights_t.append(loglikfail(theta,betas[t],eps)) 
       
        # (2) time t >= 2 (or >=1 b/c indexing from 0)
        else:
            # (a) sample theta_t 
            thetas_t = [] 
            for i in range(nparticles):
                thetas_t.append(np.random.normal(thetas[t-1][i], sigt_dyn)) # need to sample from previous distribution 
            
            # (b) compute weights - sum of likelihoods up to and including at time t

            if resample[t-1] == 'yes': # if resampling happened prior, no longer summing previous weights 
                if test[t] == 1: # if answered correctly
                    for theta in thetas_t:
                        weights_t.append(loglik(theta,betas[t],eps)) 
                else: # in test[t] == 0 - if incorrect on that problem
                    for theta in thetas_t:
                        weights_t.append(loglikfail(theta,betas[t],eps)) 
            else: # if no resampling happened, continue by adding cumulative weights
                for w in range(nparticles):
                    if test[t] == 1: # if they are correct on problem
                        w_particle = loglik(thetas_t[w],betas[t],eps) + np.log(weights[t-1][w]) # convert previous weights to log space to take sum with log likelihoods
                    else: # in test[prob] == 0 - if incorrect on that problem
                        w_particle = loglikfail(thetas_t[w],betas[t],eps) + np.log(weights[t-1][w])

                    weights_t.append(w_particle)

        # normalize weights (divide by sum of weights) but also need to move weights out of log space
        weights_t_normed = np.exp(weights_t)/np.sum(np.exp(weights_t))

        # (c) resample if the resampling criterion is satisfied
        esst = ess(weights_t_normed) # calculate effective sample size
        ess_list.append(esst)

        if esst < threshold: # only resample if effective sample size is below threshold
            resample.append('yes')
            new_weights = [1/nparticles for i in range(nparticles)] # make weights uniform if resampling
            new_thetas = random.choices(thetas_t,weights_t_normed,k=nparticles)
            
            weights[t] = new_weights
            thetas[t] = new_thetas

        else:
            resample.append('no')

            weights[t] = weights_t_normed
            thetas[t] = thetas_t
            
    return {'thetas' : thetas,
            'weights' : weights,
            'resample': resample,
            'ess': ess_list
           }