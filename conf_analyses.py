import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisqprob
import pandas as pd

def sigmoid(x): # sigmoid function will come in handy
    return 1/(1 + np.exp(-x))

def calc_sse(sims,ix,confs):
    """
    Calculate list of SSEs, where confidence judgments are divided by 100
    sims: dataframe loaded from csv of model simulations
    ix: index from confidence list of interest
    """
    sses = []
    for i in range(sims.shape[1]): # first loop through each simulated set of values
        pred = sims[str(i)]
        sse = 0
        for s in range(len(confs[ix][:-1])):
            error = pred[s] - confs[ix][s]/100
            sse+=error**2
        sses.append(sse)
    return(sses)

def calc_sse100(sims,ix,confs):
    """
    Calculate list of SSEs, 
    sims: dataframe loaded from csv of model simulations
    ix: index from confidence list of interest
    """
    sses = []
    for i in range(sims.shape[1]): # first loop through each simulated set of values
        pred = sims[str(i)]
        sse = 0
        for s in range(len(confs[ix][:-1])):
            error = pred[s]*100 - confs[ix][s]
            sse+=error**2
        sses.append(sse)
    return(sses)

def check_params_static(ix):
    """
    To check what the epsilon and mu_theta values are for the best fit model
    input: 
    index that comes out of list of SSEs
    """
    epsilons = np.linspace(0,0.5,num=11)
    muts = np.linspace(-3,3,num=31) 

    params0 = []
    for eps in epsilons: # loop through epsilons
         for mut in muts: # loop through mu thetas
            params0.append(('eps: ', eps,'mu: ', mut))
    return(params0[ix])

def check_params_dynamic(ix):
    # To check parameter values at min sse index
    epsilons = np.linspace(0,0.5,num=11)
    muts = np.linspace(-3,3,num=31) 
    sig_dyns = [.1, .2, .4, .6, .8, 1, 1.5, 2, 2.5, 3, 4, 5, 6] #np.linspace(0.2,2,num=10)

    params1 = []
    for eps in epsilons: # loop through epsilons
         for mut in muts: # loop through mu thetas
            for sig in sig_dyns:
                params1.append(('eps: ', eps,'mu: ', mut, 'sig: ', sig))
    return(params1[ix])

def check_params_dynamic_small(ix):
    # To check parameter values at min sse index
    epsilons = np.linspace(0,0.5,num=11)
    muts = np.linspace(-3,3,num=31) 
    sig_dyns = np.linspace(0.01,0.2,num=20)

    params1 = []
    for eps in epsilons: # loop through epsilons
         for mut in muts: # loop through mu thetas
            for sig in sig_dyns:
                params1.append(('eps: ', eps,'mu: ', mut, 'sig: ', sig))
    return(params1[ix])

def model_fits(ix,corrects,confs):
    ID = corrects[ix][-1]
    
    # load the associated simulations
    sims = pd.read_csv('../../Simulations/AlgTutorSims/A/Confidence_algTutor_preds'+ ID + 'A.csv', encoding = "utf-8") 
    simsD = pd.read_csv('../../Simulations/AlgTutorSims/A/Confidence_algTutor_preds' + ID + 'Adyn.csv', encoding = "utf-8")
    simsD2 = pd.read_csv('../../Simulations/AlgTutorSims/C/Confidence_algTutor_preds' + ID + 'dynC.csv', encoding = "utf-8") # small parameter fits

    
    print(sims.shape) 
    print(simsD.shape)
    print(simsD2.shape)
    
    sses = calc_sse100(sims,ix,confs)
    ssesD = calc_sse100(simsD,ix,confs)
    #for i in range(simsD2.shape[1]): # first loop through each simulated set of values
    #    simsD2[str(i)][0] = sigmoid(simsD2[str(i)][0])
    ssesD2 = calc_sse100(simsD2,ix,confs)
    
    # calculate minimum SSE and locate index for static model
    sseMin = min(sses)
    min_ix = sses.index(min(sses))
    print('min sse: ', sseMin)
    print('index: ', min_ix)
    # see parameters for best fit static model
    params = check_params_static(min_ix)
    print(params)
    
    sseMinD = min(ssesD)
    print(sseMinD)
    sseMinD2 = min(ssesD2)
    print(sseMinD2)
    
    if sseMinD < sseMinD2: #
        # calculate minimum SSE and locate index
        sseMin_dyn = sseMinD
        min_ixD = ssesD.index(min(ssesD))
        print('min sse: ', sseMinD)
        print('index: ', min_ixD)
        # see parameters for best fit dynamic model
        paramsD = check_params_dynamic(min_ixD)
        print(paramsD)
        
        plot_sims(ix,min_ix,min_ixD,confs,sims,simsD,corrects)
    
    else:
        sseMin_dyn = sseMinD2
        min_ixD = ssesD2.index(min(ssesD2))
        print('min sse: ', sseMin_dyn)
        print('index: ', min_ixD)
        # see parameters for best fit dynamic model
        paramsD = check_params_dynamic_small(min_ixD)
        print(paramsD)
        
        plot_sims(ix,min_ix,min_ixD,confs,sims,simsD2,corrects)
    
    lrt = LRtest(21, sseMin, sseMin_dyn, 1)
    to_df(ID,params,paramsD,sseMin,min_ix,sseMin_dyn,min_ixD,lrt)
    
    return {'sseMin': sseMin,
            'min_ix': min_ix,
            'params': params,
            'sseMinD': sseMin_dyn,
            'min_ixD': min_ixD,
            'paramsD': paramsD,
            'lrt': lrt
           }

def plot_sims(dat,fit,fitD,confs,sims,simsD,corrects):
    """
    Plots the data with best fit models
    dat: index in data that we're interested (location in confs and corrects lists)
    fit: index of best-fit static model
    fitD: index of best-fit dynamic model
    legend_loc: positional argument for legend ('upper left' or 'lower left' most likely candidates)
    confs: list of lists of confidence judgments
    sims: simulations from the static model
    simsD: simulations from the dynamic model
    corrects: 1's and 0's listing corrects/incorrects across all problems
    """
    fig, ax = plt.subplots()
    
    #conf = [c/100 for c in confs[dat]]
    sims100 = [s*100 for s in sims[str(fit)]]
    simsD100 = [s*100 for s in simsD[str(fitD)]]

    ax.scatter(range(1,len(confs[dat])+1), confs[dat], label="Data",color='b',marker='^')
    ax.scatter(range(1,len(confs[dat])+1), sims100, label="Static model predictions",color='c')
    ax.scatter(range(1,len(confs[dat])+1), simsD100, label="Dynamic model predictions",color='m',marker='s')

    ax.set_facecolor('white')
    # plt.title('Simulated Performance Estimate by Actual Score', fontsize=20)
    ax.set_ylim(0,100)
    ax.set_xlim(0.5,21.5)

    plt.axhline(100, color='black')
    plt.axvline(0.5, color='black')
    plt.axhline(0, color='black')
    plt.axvline(21.5, color='black')

    ax.set_xticks(np.arange(1, 21, step=1))
    plt.xticks(np.arange(1.5,21.5,step=1), np.arange(1, 21, step=1))
    plt.tick_params(length=0,labelsize=15,pad=5)
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlabel('Problem number',fontsize = 20)
    ax.set_ylabel('Confidence',fontsize = 20)
    legend = ax.legend(loc='best', prop={'size':15}, frameon=True,facecolor='w',edgecolor='k')
    fig.set_size_inches(7,7)

    for i in range(len(corrects[dat][:-1])):
        if corrects[dat][i] == 0:
            ax.axvspan(i+1, i+2, alpha=0.25, facecolor='0.5',zorder=3,hatch='', edgecolor="k")
        else:
            ax.axvspan(i+1, i+2, alpha=0.25, facecolor='none',zorder=3,hatch='', edgecolor="k")

    plt.gcf().subplots_adjust(bottom=0.15)
    fig.savefig('figs/ModelSims/ConfFit' + str(corrects[dat][-1]) + '.pdf', dpi=1000)

    plt.show()
    

def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))    

def LRtest(n, s0, s1, dof):
    k0 = 2 # number of parameters in static model
    k1 = 3 # number of parameters in dynamic model
    L0 = -n/2 *(1 + np.log(2*np.pi*(s0/n)))
    BIC0 = k0 * np.log(n) - 2*L0
    
    L1 = -n/2 *(1 + np.log(2*np.pi*(s1/n)))
    BIC1 = k1 * np.log(n) - 2*L1
    
    LR = likelihood_ratio(L0,L1)

    p = chisqprob(LR, dof) # L1 has 1 DoF more than L0, so equal to 1 for individuals, but it turns out to the number of participants when looking across multiple participants
    
    return {'L0': L0, 
            'BIC0': BIC0,
            'L1': L1,
            'BIC1': BIC1,
            'LR': LR,
            'p': p
           }


def to_df(ID,params,paramsD,sseMin,min_ix,sseMinD,min_ixD,lrt):
    to_save = {'user_ID': ID, 
               'eps_static': params[1],
               'mu_static': params[3],
          'SSE_static': sseMin, 
          'ix_SSE_static': min_ix,
               'eps_dyn': paramsD[1],
               'mu_dyn': paramsD[3],
               'sig_dyn': paramsD[5],
          'SSE_dynamic': sseMinD, 
          'ix_SSE_dynamic': min_ixD, 
          'L0': lrt['L0'], 
          'BIC0': lrt['BIC0'], 
          'L1': lrt['L1'], 
          'BIC1': lrt['BIC1'], 
          'LR': lrt['LR'], 
          'p': lrt['p']} 

    df = pd.DataFrame([to_save])  

    df.to_csv('BestFits/sim_values_' + ID + '.csv', index=False, header=True)