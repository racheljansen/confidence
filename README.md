# confidence

Confidence project:

Paper here: https://cocosci.princeton.edu/papers/jansen_cogsci2020.pdf

problemByProblem.py: particle filter code

gridsearch.py: generates model predictions given a set of correct and incorrect values (static version of the model, sigma_dyn = 0) looking at a grid of mu and epsilon values (found in AlgTutorSims/A)

dyn_gridsearch.py: generates model predictions for dynamic model (sigma_dyn > 0) (found in AlgTutorSims/A and C - folder A has large range of values up to sigma_dyn = 6 while folder C has fine-grained values up to 2)

conf_analyses.py: finds best-fit models (static and dynamic) and generates tables of all values (in BestFits folder)plus plots overlaying best-fit models on data (Plots folder)

To run simulations (replacing 0 with whatever index from the csv of corrects and incorrects):
    
    python3 gridsearch.py AlgTutor_corrects.csv 0
    
    python3 dyn_gridsearch_alg.py AlgTutor_corrects.csv 0
