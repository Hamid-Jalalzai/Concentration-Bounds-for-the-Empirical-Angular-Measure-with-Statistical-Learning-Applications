#!/usr/bin/env python
# coding: utf-8

'''
Authors:  Stéphan CLEMENÇON, Hamid JALALZAI, Anne SABOURIN and Johan SEGERS

The purpose of this script is to compare the angular estimation on 
on a paving of the speher with various standardisation 
(unknown margins + with truncated estimator or not) 
'''

#########################################################
# Import libraries

import time
import numpy as np
from utils import *
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
#########################################################

# dimension of the problem
dim_pb = 2 # set to 5 if dim of the problem is 5 
# initialize grid size
grid_param = 5
# setting alpha dependence param
alpha  = 10

#print(time.ctime())
# uncommenting the line below may take quite a long time to run and cause Memory Leak/overload. run with large amount of RAM
#rectangle_MC = generateMonteCarlo(0,d=dim_pb, tau=0.1, N_MC=int(5e8),alpha=alpha, grid_size=grid_param, pickle=False)
#print(time.ctime())



#print("std_MC", np.std(list(rectangle_MC.values())))
#with open('rectangle_MC_dim'+str(dim_pb)+'.pickle', 'wb') as handle:
#    pickle.dump(rectangle_MC, handle, protocol=pickle.HIGHEST_PROTOCOL)


# if you did not uncomment the line above please run the following line
# to get the MC estimator with the same params
with open('rectangle_MC_dim'+str(dim_pb)+'.pickle', 'rb') as handle:
    rectangle_MC = pickle.load(handle)


# Range of sample sizes 
N_range  = [5000, int(1e4), int(5e4), int(1e5), int(5e5), int(1e6) ]

# Range of Quantile of truncated data (M values)
qt_range = [90, 95, 98]

# number of exp replications
N_exp = 50


# Initialize the different error arrays 
sup_error_V = np.empty((N_exp, len(N_range))) #Phi tilde
sup_error_hat_V = np.empty((N_exp, len(N_range))) #Phi hat
sup_error_hat_V_M = np.empty((N_exp, len(N_range))) #Phi hat + truncation

N_removed = np.empty((N_exp, len(N_range)))

# Loop over the various quantile 
for qt in qt_range:
    for i, seed in enumerate(range(N_exp)):
        if i % 10 == 0:
            print(qt, i, time.ctime())
        for j, N in enumerate(N_range):
	#may take quite a long time to run and cause Memory Leak/overload. Better to run with large amount of RAM	
            rectangle_hat_V, rectange_hat_V_M, rectangle_V, N_removed_ = generateParetoStandardization(seed,d=dim_pb, tau=0.1,
                                                                                          alpha=alpha,
                                                                                          n= N, grid_size=grid_param,
                                                                                          quantile=qt, 
                                                                                          pickle_=False)
	    
	    # Compute error with Phi MC and  Phi tilde
            sup_error_V[i, j] = compute_absolute_error(rect_train = rectangle_V, 
                                                     rectangle_MC = rectangle_MC)[0]

	    # Compute error with Phi MC and  Phi tilde with Pareto standardization 
            sup_error_hat_V[i, j] = compute_absolute_error(rect_train = rectangle_hat_V, 
                                                     rectangle_MC = rectangle_MC)[0]


	    # Compute error with Phi MC and  Phi tilde with Pareto standardization + truncation 
            sup_error_hat_V_M[i, j] = compute_absolute_error(rect_train = rectange_hat_V_M, 
                                                     rectangle_MC = rectangle_MC)[0]
            
    
            N_removed[i, j] = float(N_removed_)
    #print("N removed", N_removed.mean(axis=0))
    #saving all errors 
    np.save("sup_error_V_dim"+str(dim_pb)+".npy", sup_error_V)
    np.save("sup_error_hat_V_dim"+str(dim_pb)+".npy", sup_error_hat_V)
    np.save("sup_error_hat_V_M_dim"+str(dim_pb)+"qt_"+str(qt)+".npy", sup_error_hat_V_M)

    # Visualization

    plt.figure(figsize=(10, 8))
    plt.plot(np.array(np.log(np.sqrt(N_range))), 
             np.mean(np.log(sup_error_V), axis=0), marker="o", c="royalblue", 
             label=r'$\log(\sup_{A \in \mathcal{A}} ~~| \phi - \widetilde{\phi} | )$' )
    plt.plot(np.array(np.log(np.sqrt(N_range))), 
             np.mean(np.log(sup_error_hat_V), axis=0), marker="o", c="salmon", 
             label=r'$\log(\sup_{A \in \mathcal{A}} ~~| \phi - \widehat{\phi} | )$' )
    plt.plot(np.array(np.log(np.sqrt(N_range))), 
             np.mean(np.log(sup_error_hat_V_M), axis=0), marker="o", c="green", 
             label=r'$\log(\sup_{A \in \mathcal{A}} ~~| \phi - \widehat{\phi}^M | )$' )


    plt.plot(np.array(np.log(np.sqrt(N_range))),
             np.log(1 / np.power(N_range, 1/4)) , 'k--', label=r"$\log(1/\sqrt{k})$")
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    #plt.ylabel("log", fontsize=20)
    plt.xlabel(r"$\log(k)$", fontsize=20)
    plt.legend(fontsize=20)
    #plt.title("M discard "+str(100 - qt)+"% of extreme data")
    plt.savefig("boundingError_"+str(qt)+"_"+str(dim_pb)+".pdf", format="pdf", bbox_inches='tight')
    #plt.show()
    
print("done", time.ctime())

