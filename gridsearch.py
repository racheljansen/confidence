import random
import pandas as pd
import numpy as np
import timeit
import sys

import problemByProblem as pf # import series of functions associated with particle filter script


def static_sims(corrects, ix, nparticles = 10000, sigt = 1, threshold = 3000):
    """
    corrects: csv of corrects and user_IDs
    epsilons: list of eps values
    muts: list of mut values
    """

    total_start = timeit.default_timer()
    
    data = pd.read_csv(corrects, encoding = "utf-8")
    data = data.values.tolist()
    test = data[int(ix)][:-1]
    user = data[int(ix)][-1]
    
    betas = np.random.randn(len(test))
    epsilons = np.linspace(0,0.5,num=11)
    muts = np.linspace(-3,3,num=61)
    
    sigt_dyn = 0
    
    preds = [] # empty list of thetas
    particles = [] # empty list for all of particle filter output

    for eps in epsilons: # loop through epsilons
        for mut in muts: # loop through mu thetas

            start = timeit.default_timer()


            sim = pf.seqMC(betas, mut, sigt, nparticles, eps, test, threshold, sigt_dyn)
            particles.append(sim) 

            # transfer thetas into vector of vectors of thetas for a given item and sigmoid each
            prior = np.mean(np.random.normal(mut,sigt,nparticles))
            means = [pf.sigmoid(prior)] # this is the first estimate
            for i in range(len(sim['thetas'])):
                wtheta = []
                for p in range(nparticles):
                    wtheta.append(pf.sigmoid(sim['thetas'][i][p])*sim['weights'][i][p])
                means.append(np.sum(wtheta))         

            preds.append(means) 
            stop = timeit.default_timer()

            if len(preds)%50:
                print('time: ', stop - start)

            print('epsilon: ', eps)
            print('mut: ', mut)

    
    preds_all = pd.DataFrame(preds)
    preds_all = preds_all.transpose()
    preds_all.to_csv('./AlgMCsims/C/Confidence_algMC_preds' + str(user) + 'C.csv', index=False, header=True)
    
    total_end = timeit.default_timer()
    print('Total time: ', total_end - total_start)
    

def main():
    corrects = sys.argv[1];
    ix = sys.argv[2];
    static_sims(corrects,ix)
    
if __name__ == "__main__":
    main()