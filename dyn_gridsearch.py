import random
import pandas as pd
import numpy as np
import timeit
import sys

import problemByProblem as pf # import series of functions associated with growth mindset script
    
def dynamic_sims(corrects, ix):#, epsilons, muts, sigt_dyns):
    
    total_start = timeit.default_timer()
    nparticles = 10000 
    sigt = 1.
    threshold = 3000
    
    data = pd.read_csv(corrects, encoding = "utf-8")
    data = data.values.tolist()
    test = data[int(ix)][:-1]
    user = data[int(ix)][-1]
    
    betas = np.random.randn(len(test))
    
    
    epsilons = np.linspace(0,0.5,num=11)
    muts = np.linspace(-3,3,num=31) 
    sigt_dyns = np.linspace(0.01,0.2,num=20)


    preds = [] # empty list of thetas converted to percentages
    particles = [] # empty list for all of filter output

    for eps in epsilons: # loop through epsilons
        for mut in muts: # loop through mu thetas
            for sig_dyn in sigt_dyns:

                start = timeit.default_timer()

                sim = pf.seqMC(betas, mut, sigt, nparticles, eps, test, threshold, sig_dyn)
                particles.append(sim) #add all info from the chain to a list

                # transfer thetas into vector of vectors of thetas for a given item and sigmoid each
                prior = np.mean(np.random.normal(mut,sigt,nparticles))
                means = [prior] # this is the first estimate
                for i in range(len(sim['thetas'])):
                    wtheta = []
                    for p in range(nparticles):
                        wtheta.append(pf.sigmoid(sim['thetas'][i][p])*sim['weights'][i][p])
                    means.append(np.sum(wtheta))         

                stop = timeit.default_timer()

                if len(preds)%50: # if the we're at a point that's a multiple of 100
                    print('time: ', stop - start)

                print('epsilon: ', eps)
                print('mut: ', mut)
                print('sig: ', sig_dyn)

                preds.append(means)  
                
    preds_all = pd.DataFrame(preds)
    preds_all = preds_all.transpose()
    preds_all.to_csv('./Simulations/AlgMC/C/Confidence_algMC_preds' + ID + 'dynC.csv', index=False, header=True)

    total_end = timeit.default_timer()
    print('Total time: ', total_end - total_start)

def main():
    corrects = sys.argv[1];
    ix = sys.argv[2];
    #epsilons = sys.argv[3];
    #muts = sys.argv[4]
    dynamic_sims(corrects,ix)#,epsilons,muts)
    
if __name__ == "__main__":
    main()