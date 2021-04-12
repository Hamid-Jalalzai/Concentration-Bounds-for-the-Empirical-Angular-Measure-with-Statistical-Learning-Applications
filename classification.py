#!/usr/bin/env python
# coding: utf-8

'''
Authors:  Stéphan CLEMENÇON, Hamid JALALZAI, Anne SABOURIN and Johan SEGERS

The purpose of this script is to compare the performance of a given 
classifier with various standardisation (unknown margins + 
with truncated estimator or not) 
'''
#########################################################
# Import libraries

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from scipy.stats import pareto, dirichlet, ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import dirichlet, pareto
from utils import *

rnd_seed = 42
np.random.seed(rnd_seed)
#########################################################




#########################################################
#############   Simulation illustration   ###############
#########################################################


d = 10 #dimension of the problem
N_iter = 100 # Nb of experiment iterations

# Dirichlet params +/-
alpha_plus  = 1 
alpha_minus = 2

# kappa params
kappa_step  = 1
kappa_max   = 2

# RF hyperparameters
n_tree_rf = 100 
rf_max_depth = 10 

# Linear Model hyperparameters
lm_loss = "log"

# tree hyperparameters
tree_max_depth=10

kappa_ = np.arange(1, kappa_max , kappa_step)
score_rf_km = np.empty((N_iter, len(kappa_)))
score_lm_km = np.empty((N_iter, len(kappa_)))
score_tree_km = np.empty((N_iter, len(kappa_)))

score_rf_um = np.empty((N_iter, len(kappa_)))
score_lm_um = np.empty((N_iter, len(kappa_)))
score_tree_um = np.empty((N_iter, len(kappa_)))

score_rf_um_M   = np.empty((N_iter, len(kappa_)))
score_lm_um_M   = np.empty((N_iter, len(kappa_)))
score_tree_um_M = np.empty((N_iter, len(kappa_)))

N_removed = np.empty(N_iter)
size_diff = np.empty(N_iter)

vrb = False
test_size = np.empty((N_iter, len(kappa_)))

for i in range(N_iter):
    if i % 10 == 0:
        print(i, time.ctime())
    for j, kappa in enumerate(kappa_):
        Θ_train, hat_Θ_train, hat_Θ_train_M, Θ_test, hat_Θ_test, y_train, y_hat_train, y_hat_train_M, y_test, y_hat_test = make_data(i,
                                                                                                  n_train=1e5,
                                                                                                  n_test=1e5,
                                                                                                  d = d,
                                                                                                  kappa = kappa,     
                                                                                                  alpha_plus=alpha_plus,
                                                                                                  alpha_minus=alpha_minus,
                                                                                              tau = 0.1, quantile=95, verbose=vrb)

        
        
        test_size[i, j] = len(Θ_test)
        #we train both classifiers on the same number of observations
                
        ############## Known Margins ##############
        # Random Forest with Known Margins : rf_km
        rf_km = RandomForestClassifier(n_jobs=-1, n_estimators=n_tree_rf,
                                       max_depth=rf_max_depth, random_state=i)
        rf_km.fit(Θ_train, y_train)
        score_rf_km[i, j] = rf_km.score(Θ_test, y_test)
        
        # Linear model with Known Margins : lm_km
        lm_km = LogisticRegression(n_jobs=-1, solver='lbfgs' ,random_state=i) #loss=lm_loss,
        lm_km.fit(Θ_train, y_train)
        score_lm_km[i, j] = lm_km.score(Θ_test, y_test)
        
        # Tree model with Known Margins : tree_km
        tree_km = DecisionTreeClassifier(random_state=i, max_depth = tree_max_depth)
        tree_km.fit(Θ_train, y_train)
        score_tree_km[i, j] = tree_km.score(Θ_test, y_test)
        
        ############## Unknown Margins ##############
        #Random Forest with Unknown Margins : rf_um
        rf_um = RandomForestClassifier(n_jobs=-1, n_estimators=n_tree_rf, 
                                       max_depth=rf_max_depth, random_state=i)
        rf_um.fit(hat_Θ_train, y_hat_train)
        score_rf_um[i, j] = rf_um.score(hat_Θ_test, y_hat_test) 
        
        # Linear model with Unnown Margins : lm_um
        lm_um = LogisticRegression(n_jobs=-1, solver='lbfgs', random_state=i) #loss=lm_loss,
        lm_um.fit(hat_Θ_train, y_hat_train)
        score_lm_um[i, j] = lm_um.score(hat_Θ_test, y_hat_test)
        
        # DecisionTreeClassifier with Known Margins : tree_um
        tree_um = DecisionTreeClassifier(random_state=i, max_depth = tree_max_depth)
        tree_um.fit(hat_Θ_train, y_hat_train)
        score_tree_um[i, j] = tree_um.score(hat_Θ_test, y_hat_test)

        ############## Unknown Margins + Truncation ##############
        #Random Forest with Unknown Margins and M truncation : rf_um_M
        rf_um_M = RandomForestClassifier(n_jobs=-1, n_estimators=n_tree_rf, 
                                         max_depth=rf_max_depth, random_state=i)
        rf_um_M.fit(hat_Θ_train_M, y_hat_train_M)
        score_rf_um_M[i, j] = rf_um_M.score(hat_Θ_test, y_hat_test) 
        
        # Linear model with Unnown Margins and M truncation : lm_um_M
        lm_um_M = LogisticRegression(n_jobs=-1, solver='lbfgs', random_state=i) #loss=lm_loss, 
        lm_um_M.fit(hat_Θ_train_M, y_hat_train_M)
        score_lm_um_M[i, j] = lm_um_M.score(hat_Θ_test, y_hat_test)
        
        # DecisionTreeClassifier with Known Margins : tree_um_M
        tree_um_M = DecisionTreeClassifier(random_state=i, max_depth = tree_max_depth)
        tree_um_M.fit(hat_Θ_train_M, y_hat_train_M)
        score_tree_um_M[i, j] = tree_um_M.score(hat_Θ_test, y_hat_test)
        
    N_removed[i] = len(hat_Θ_train) - len(hat_Θ_train_M)
    size_diff[i] = len(hat_Θ_train) - len(Θ_train)


#########################################################
##############  Simulation Visualization  ###############
#########################################################


list_plot = [ ]
for i in range(1): #score_rf_km.shape[1]
    #list_plot.append(1 - score_rf_km[:,i])
    list_plot.append(1 - score_rf_um[:,i])
    list_plot.append(1 - score_rf_um_M[:,i])
    
    
# c  = "blue"
c1 = "red"
c2 = "green"

plt.figure(figsize=(11, 8))

labels = ['RF with unknown margins \n and truncation' , 
          'RF with unknown margins',
         ] # 'RF with known margins' , 

colors = ["yellowgreen",  
          "lightblue",
         ] #  "lavenderblush",

patches = [
    mpatches.Patch(color=color, label=label)
    for label, color in zip(labels, colors)]
#plt.legend(patches, labels, frameon=False, 
#           fontsize=20, loc='center left', bbox_to_anchor=(1, 0.75))

box = plt.boxplot(list_plot, notch=False, patch_artist=True, widths = 0.8)

colors = [ 'blue', 'green']#*int((len(list_plot))/3) #'red',
for patch, color in zip(box['medians'], colors):
    patch.set_color(color)


ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax.tick_params(axis = 'both', which = 'minor', labelsize = 16)

colors = [ 'lightblue', 'yellowgreen' ]#*int((len(list_plot))/3) # 'lavenderblush',
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

#plt.title("")
#plt.xticks(np.arange(len(kappa_)*3 + 1), [0] + list(np.array([i for i in zip(np.round(kappa_, decimals=2),
#                                                                                       np.round(kappa_, decimals=2),
#                                                                               np.round(kappa_, decimals=2))]).ravel()),
#           rotation = 0, fontsize=20
#          )

plt.xticks([1, 2,], [r'$\widehat g^\tau$', r'$\widehat g^{\tau, M}$',], fontsize=25) # 3




plt.yticks(fontsize=20)
plt.ylabel(r"Classification Error", fontsize=20)
#plt.xlabel(r"Multiplicative factor", fontsize=19)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='minor',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along t
plt.savefig("classifBoxplot_RF.pdf", format="pdf", bbox_inches='tight')
#plt.show()




list_plot = [ ]
for i in range(1): #score_rf_km.shape[1]
    #list_plot.append(1 - score_rf_km[:,i])
    list_plot.append(1 - score_lm_um[:,i])
    list_plot.append(1 - score_lm_um_M[:,i])
    
    
# c  = "blue"
c1 = "red"
c2 = "green"

plt.figure(figsize=(11, 8))

labels = ['RF with unknown margins \n and truncation' , 
          'RF with unknown margins',
         ] # 'RF with known margins' , 

colors = ["yellowgreen",  
          "lightblue",
         ] #  "lavenderblush",

patches = [
    mpatches.Patch(color=color, label=label)
    for label, color in zip(labels, colors)]
#plt.legend(patches, labels, frameon=False, 
#           fontsize=20, loc='center left', bbox_to_anchor=(1, 0.75))

box = plt.boxplot(list_plot, notch=False, patch_artist=True, widths = 0.8)

colors = [ 'blue', 'green']#*int((len(list_plot))/3) #'red',
for patch, color in zip(box['medians'], colors):
    patch.set_color(color)


ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax.tick_params(axis = 'both', which = 'minor', labelsize = 16)

colors = [ 'lightblue', 'yellowgreen' ]#*int((len(list_plot))/3) # 'lavenderblush',
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

#plt.title("")
#plt.xticks(np.arange(len(kappa_)*3 + 1), [0] + list(np.array([i for i in zip(np.round(kappa_, decimals=2),
#                                                                                       np.round(kappa_, decimals=2),
#                                                                               np.round(kappa_, decimals=2))]).ravel()),
#           rotation = 0, fontsize=20
#          )

plt.xticks([1, 2,], [r'$\widehat g^\tau$', r'$\widehat g^{\tau, M}$',], fontsize=25) # 3
plt.yticks(fontsize=20)
plt.ylabel(r"Classification Error", fontsize=20)
#plt.xlabel(r"Multiplicative factor", fontsize=19)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='minor',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along t
plt.savefig("classifBoxplot_lm.pdf", format="pdf", bbox_inches='tight')
#plt.show()




list_plot = [ ]
for i in range(1): #score_rf_km.shape[1]
    #list_plot.append(1 - score_rf_km[:,i])
    list_plot.append(1 - score_tree_um[:,i])
    list_plot.append(1 - score_tree_um_M[:,i])
    
    
# c  = "blue"
c1 = "red"
c2 = "green"

plt.figure(figsize=(11, 8))

labels = ['RF with unknown margins \n and truncation' , 
          'RF with unknown margins',
         ] # 'RF with known margins' , 

colors = ["yellowgreen",  
          "lightblue",
         ] #  "lavenderblush",

patches = [
    mpatches.Patch(color=color, label=label)
    for label, color in zip(labels, colors)]
#plt.legend(patches, labels, frameon=False, 
#           fontsize=20, loc='center left', bbox_to_anchor=(1, 0.75))

box = plt.boxplot(list_plot, notch=False, patch_artist=True, widths = 0.8)

colors = [ 'blue', 'green']#*int((len(list_plot))/3) #'red',
for patch, color in zip(box['medians'], colors):
    patch.set_color(color)


ax = plt.gca()
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax.tick_params(axis = 'both', which = 'minor', labelsize = 16)

colors = [ 'lightblue', 'yellowgreen' ]#*int((len(list_plot))/3) # 'lavenderblush',
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

#plt.title("")
#plt.xticks(np.arange(len(kappa_)*3 + 1), [0] + list(np.array([i for i in zip(np.round(kappa_, decimals=2),
#                                                                                       np.round(kappa_, decimals=2),
#                                                                               np.round(kappa_, decimals=2))]).ravel()),
#           rotation = 0, fontsize=20
#          )

plt.xticks([1, 2,], [r'$\widehat g^\tau$', r'$\widehat g^{\tau, M}$',], fontsize=25) # 3
plt.yticks(fontsize=20)
plt.ylabel(r"Classification Error", fontsize=20)
#plt.xlabel(r"Multiplicative factor", fontsize=19)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='minor',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along t
plt.savefig("classifBoxplot_tree.pdf", format="pdf", bbox_inches='tight')
#plt.show()





plt.figure()
plt.plot( [ks_2samp(score_rf_km[:,i], score_rf_um_M[:,i])[1] for i in range(len(kappa_))], label="km | umM" )
plt.plot( [ks_2samp(score_rf_km[:,i], score_rf_um[:,i])[1]   for i in range(len(kappa_))], label="km | um" )
plt.plot( [ks_2samp(score_rf_um[:,i], score_rf_um_M[:,i])[1] for i in range(len(kappa_))], label="um | umM" )
plt.legend()
#plt.show()


print("Random Forest")
print("km vs um M :", ks_2samp(score_rf_km.flatten(), score_rf_um_M.flatten()))
print("um vs um M :", ks_2samp(score_rf_um.flatten(), score_rf_um_M.flatten()))
print("km vs um   :", ks_2samp(score_rf_km.flatten(), score_rf_um.flatten()))




print("Linear Model")
print("km vs um M :", ks_2samp(score_lm_km.flatten(), score_lm_um_M.flatten()))
print("um vs um M :", ks_2samp(score_lm_um.flatten(), score_lm_um_M.flatten()))
print("km vs um   :", ks_2samp(score_lm_km.flatten(), score_lm_um.flatten()))



print("Tree")
print("km vs um M :", ks_2samp(score_tree_km.flatten(), score_tree_um_M.flatten()))
print("um vs um M :", ks_2samp(score_tree_um.flatten(), score_tree_um_M.flatten()))
print("km vs um   :", ks_2samp(score_tree_km.flatten(), score_tree_um.flatten()))


# Conclusion: The standardization (<i>with truncation or not</i>) do not change the performance of the classifier (all other things remaining equal)
