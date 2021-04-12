#!/usr/bin/env python
# coding: utf-8

'''
Authors:  Stéphan CLEMENÇON, Hamid JALALZAI, Anne SABOURIN and Johan SEGERS
This file contains the tool functions for all experiments.
'''

################### Import libraries ####################
import time
import pickle
import numpy as np
from scipy.stats import dirichlet, pareto
#########################################################

#########################################################
########### Pareto Standardization functions ############
#########################################################

def order(X):
    """ Returns the ranked samples in X 
    Params:
    @X (array): data matrix
    Returns:
    @R (array): ranked (sorted) samples
    """
    R = np.sort(X, axis=0) # Sort samples
    return R

def transform(R, X):
    """ Common transformation of each marginal in standard Pareto
    Params:
    @R (array): ranked training data
    @X (array): data to transform (sample by sample, independently) according to
    training transform
    Returns: Transformed data according to Pareto standardization
    """
    n,d = np.shape(X)
    n_R = np.shape(R)[0]
    T = np.zeros((n, d))
    # Apply Pareto standardization on each marginal
    for i in range(d):
        T[:, i] = np.searchsorted(R[:, i], X[:, i]) / float(n_R + 1)
    return 1. / (1-T)


#########################################################
############### Simulation core functions ###############
#########################################################
    
def build_rectangle(V, weight, grid):
    """ Build paving of the sphere with rectangles
    Params:
    @Theta  (array): input angles
    @weight (float): weight associated to angle
    @grid   (array): grid on the L_inf sphere
    Returns:
    @rect    (dict): estimated angular measure
    """
    # initialize rectangle dictionary
    rect = {}
    # loop over the angles
    for theta_i in Theta:
        key = str(theta_i.argmax()) + '-'
        # radius of the current sample
        R_i = np.linalg.norm(theta_i, ord=np.inf) 
        for l in range(len(theta_i)):
                if l != theta_i.argmax():
                    key += str(np.min(np.where(theta_i[l] / R_i <= grid )) )
        if key in rect.keys():
            rect[key] += weight
        else:
            rect[key] = weight
    return rect

def generateMonteCarlo(rand_seed, N_MC=1e8, d=2, grid_size=4,
                       alpha=4, independence=False,
                       tau=0, verbose=False, pickle=True):
    """ Approximate the angular measure with Monte Carlo procedure
    Params:
    @N_MC          (int): MC sample size
    @d             (int): dimension
    @grid_size     (int): paving size of the L_inf sphere
    @alpha   (float > 1): Dirichlet concentration param
    @independence (bool): central or axis concentration param
    @tau         (float): min angular region from the axis to avoid on the L_inf sphere
    @verbose      (bool): bool to print output
    @pickle       (bool): dump the generated samples
    Returns:
        None if pickle is True or rectangle (dict) containing the angular measure 
    """
    if independence:
        # extreme features may be large independently
        if alpha:
            alpha_ = np.ones(d)/alpha
        else:
            alpha_ = np.ones(d)/d
    else:
        # extremes features are large simultaneously
        if alpha:
            alpha_ = np.ones(d)*alpha
        else:  
            alpha_ = np.ones(d)*d
    # initialize paving of the L_inf sphere
    grid = np.linspace(tau, 1, num=grid_size+1) 
    # initialize rectangle dictionary
    rectangle = dict()
    # samples radius and norm for MC estimation
    theta = dirichlet.rvs(alpha=alpha_, size=N_MC, random_state=rand_seed)
    R  = pareto.rvs(b=1, size=N_MC, random_state=rand_seed)
    # polar decomposition X = R*theta
    X_MC = R.reshape(-1, 1) * theta
    # norms of all the generated samples
    norm_X_MC = np.linalg.norm(X_MC, axis=1, ord=np.inf)

    # display information
    if verbose:
        print("MC sample size   :", N_MC) 
        print("dimension        :", d)
        print("grid_size of cube:", grid_size)
        print("-------------------------------------------------------")
        
    # loop over the extremes generated samples
    for idx, theta_i in enumerate(theta[norm_X_MC >= 1]):
        # display information
        if idx % int(1e6) == 0:
            if verbose:
                print(idx, time.ctime())
        # current face
        key = str(theta_i.argmax()) + '-'
        # loop over the faces of the current sample
        for l in range(len(theta_i)):
                if l != theta_i.argmax():
                    key += str(np.min(np.where(theta_i[l] / np.linalg.norm(theta_i, ord=np.inf) <= grid )))
        # update value of MC estimate
        if key in rectangle.keys():
            rectangle[key] += d/N_MC
        else:
            rectangle[key] = d/N_MC
    if pickle: # save result
        with open('rectangle.'+'N_MC='+str(np.format_float_scientific(N_MC))+
                  '.d='+str(d)+'.alpha='+str(alpha_)+'.tau='+str(tau)+
                  '.grid_size='+str(grid_size)+'.pickle', 'wb') as handle:
            pickle.dump(rectangle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else: # return rectangle dictionary
        return rectangle


def generateParetoStandardization(rand_seed, n=100, d=2, grid_size=4, alpha=4,
                       independence=False, tau=0,quantile = 66, verbose=False, pickle_=True,):

    """
    Estimate the influence of the Pareto standardization
    - Phi       (with known margins)  
    - hat Phi   (based on the Pareto standardization)
    - hat_Phi_M (based on the Pareto standardization + Truncation)
    Params:
        @rand_seed     (int): random_seed for reproducibility
	@n             (int): number of generated data
        @d             (int): sample size
        @grid_size     (int): the paving size of the L_inf sphere
        @alpha   (float > 1): Dirichlet concentration param
        @independence (bool): central or axis concentration param
        @tau         (float): min angular region from the axis to avoid on the L_inf sphere
        @quantile  ([0,100]): build M such that a ratio quantile of extreme points are kept
        @verbose      (bool): bool to print output
        @pickle       (bool): dump the generated samples
    Returns
        None if pickle is True or rectangle (dict) containing the angular measure 
    """
     ####################################
    # sanity check to make sure that given n is an int
    n = int(n)
    # setting k to define the extreme region thrshld
    k = np.sqrt(n)
    # sanity check 
    if np.sqrt(n)*tau <= d:
        print("error n/k * \tau > d is false:", np.sqrt(n)*tau,  d)
    if independence:
	## extreme features may be large independently
        if alpha:
            alpha_ = np.ones(d) * 1 / (alpha)
        else:
            alpha_ = np.ones(d) * 1/(d)
    else:
	# extreme features are large independently
        if alpha:
            alpha_ = np.ones(d) * (alpha)
        else:  
            alpha_  = np.ones(d) * (d)
    grid = np.linspace(tau, 1, num=grid_size+1) #np.arange(tau, 1+1/grid_size , 1/grid_size) 

    # initialize all rectangle dictionaries
    # true input data
    rectangle_V = dict()
    # Pareto standardized data
    rectangle_hat_V = dict()
    # Pareto standardized data + Truncation
    rectange_hat_V_M = dict()
    
    # sample radius and angular components for simulation study
    theta = dirichlet.rvs(alpha=alpha_, size=n, random_state=rand_seed)
    R  = pareto.rvs(b=1, size=n, random_state=rand_seed)

    X = R.reshape(-1, 1) * theta
    V = d * X#[np.min(X, axis=1) > 1]
    hat_V = transform(order(X), X)
    
    norm_V = np.linalg.norm(V, axis=1, ord=np.inf)
    norm_hat_V = np.linalg.norm(hat_V, axis=1, ord=np.inf)

    is_extreme_V    = norm_V >= n / k
    is_extreme_hatV = norm_hat_V >= n / k
        
    # all samples with norms smaller than a given quantile
    M = np.percentile(norm_hat_V[is_extreme_hatV], q=quantile)
    
    if M < 1:
        print("Warning M < 1:", M)
    is_smaller = norm_hat_V <= M 
    
    is_V_tau_valide = np.min(V / norm_V.reshape(-1, 1), axis=1) >= tau
    is_hat_V_tau_valide = np.min(hat_V / norm_hat_V.reshape(-1,1), axis=1) >= tau
    
    N_removed = np.sum(is_extreme_hatV * is_hat_V_tau_valide) - np.sum(is_extreme_hatV * is_hat_V_tau_valide * is_smaller) 
    
    if verbose:
        print("The dimension of the problem              :", d)
        print("The size of the grid for the cube         :", grid_size)
        print("The number of points to compute phi by MC :", N_MC)
        print("-------------------------------------------------------")

    ## Computing the mass on V = d * X[X > 1]
    for idx, V_i in enumerate(V[is_extreme_V * is_V_tau_valide]):
        key = str(V_i.argmax()) + '-'
        for l in range(len(V_i)):
                if l != V_i.argmax():
                    key += str(np.min(np.where(V_i[l] / np.linalg.norm(V_i, ord=np.inf) <= grid )))
        if key in rectangle_V.keys():
            rectangle_V[key] += 1 / k #* is_tau_valide
        else:
            rectangle_V[key] = 1 / k #* is_tau_valide
        
    #-----------------------------------------------------------------------
    
    ## Computing the mass on hat V with the regular estimator
    for idx, V_i in enumerate(hat_V[is_extreme_hatV * is_hat_V_tau_valide]):
        key = str(V_i.argmax()) + '-'
        for l in range(len(V_i)):
                if l != V_i.argmax():
                    key += str(np.min(np.where(V_i[l] / np.linalg.norm(V_i, ord=np.inf) <= grid )))
        if key in rectangle_hat_V.keys():
            rectangle_hat_V[key] += 1 / k 
        else:
            rectangle_hat_V[key] = 1 / k         
            
     #-----------------------------------------------------------------------
    
    ## Computing the mass on hat V with the truncated estimator
    for idx, V_i in enumerate(hat_V[is_extreme_hatV * is_hat_V_tau_valide * is_smaller]):
        key = str(V_i.argmax()) + '-'
        for l in range(len(V_i)):
                if l != V_i.argmax():
                    key += str(np.min(np.where(V_i[l] / np.linalg.norm(V_i, ord=np.inf) <= grid )))
        if key in rectange_hat_V_M.keys():
            rectange_hat_V_M[key] += (M/(M-1)) * (1 / k) 
        else:
            rectange_hat_V_M[key] =(M/(M-1)) * (1 / k) 
    if pickle_:
        with open('rectangleV.'+'N_MC='+str(np.format_float_scientific(N_MC))+'.d='+str(d)+'.alpha='+str(alpha_)+'.tau='+str(tau)+'.grid_size='+str(grid_size)+'.pickle', 'wb') as handle:
            pickle.dump(rectangle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        return rectangle_hat_V,rectange_hat_V_M, rectangle_V, N_removed


def compute_absolute_error(rectangle_MC, rect_train):
    """Compute the absolute error between the angular measures
       given by rectangle_MC and rect_train	
    Params:
    @rectangle_MC (dict): MC approximation of angular measure
    @rect_train   (dict): The estimated angular measure with the same grid
    Returns:
    @sup_absolute_error_train (float): supremum absolute error value 
    @absolute_error_train     (float): sum of all absolute errors over rectangles
    @key_sup                    (str): rectangle where the supremum error is reacher
    """
    # initialize errors
    sup_absolute_error_train = 0
    absolute_error_train     = 0
    # intialize rectangle where the sup is reached
    key_sup = "" 
    # loop over MC_rectangle
    for key in rectangle_MC.keys():
        if key in rect_train.keys():
            absolute_error_train += np.abs(rect_train[key] - rectangle_MC[key])
            # if current value is greater than sup
            if np.abs(rect_train[key] - rectangle_MC[key]) > sup_absolute_error_train:
                # update supremum
                sup_absolute_error_train = np.abs(rect_train[key] - rectangle_MC[key])
                # update rectangle face
                key_sup = str(key)
        else:
            absolute_error_train+= rectangle_MC[key]
            # if current value is greater than sup
            if rectangle_MC[key] > sup_absolute_error_train:
                # update supremum
                sup_absolute_error_train = rectangle_MC[key]    
                # update rectangle face
                key_sup = str(key)
    # loop over rectangle of train samples
    for key in rect_train:
        if not key in rectangle_MC.keys():
            absolute_error_train += rect_train[key]
            # if current value is greater than sup
            if rect_train[key] > sup_absolute_error_train:
                # update supremum
                sup_absolute_error_train = rect_train[key]
                # update rectangle face
                key_sup = str(key)           
    return sup_absolute_error_train, absolute_error_train, key_sup


def make_data(rand_seed, n_train, n_test,
              d, tau, quantile=95, kappa=1.,
              alpha_plus=None, alpha_minus=None,
              verbose=False):
    """ Generate data of three different types:
    - Theta       (with known margins)  
    - hat_Theta   (based on the Pareto standardization)
    - hat_Theta_M (based on the Pareto standardization + Truncation)
    Params:
    @rand_seed    (int): random_seed for reproducibility 
    @n            (int): number of samples to generate 
    @d            (int): dimension of the samples
    @tau        (float): angular region to avoid
    @quantile     (int): [0, 100] quantile of the ||\widehat T(X_i)|| to remove samples
    @kappa      (float): multiplicative factor
    alpha_plus  (float): Dirichlet concentration coef for the data labeled +1
    alpha_minus (float): Dirichlet concentration coef for the data labeled -1
    
    Returns:
    @Theta_train      : Angle of train extreme samples based on V = d R * Theta such that R * Theta > 1 
    @hat_Theta_train  : Angle of train extreme samples based on \hat V = \hat T(X)
    @hat_Theta_train_M:
    @Theta_test       : Angle of test  extreme samples based on V = d R * Theta such that R * Theta > 1 
    @hat_Theta_test   : Angle of test  extreme samples based on \hat V = \hat T(X)
    @y_train          : label corresponding to extreme train samples V
    @y_hat_train      : label corresponding to extreme train samples \hat V
    @y_hat_train_M    :
    @y_test           : label of extreme test samples V 
    @y_hat_test       : label of extreme test samples \hat V 
    """
    # threshold for selection extremes
    k = np.sqrt(n_train)    
    # sanity check to have samples on the sphere
    if not (n_train/k) * tau > d or verbose:
        print("Condition (n_train / k) * tau > d is ", (n_train / k) * tau > d)
    n_train = int(n_train/2) # because we generate data labeled +1 and data labeled -1
    n_test  = int(n_test/2) # because we generate data labeled +1 and data labeled -1
    # weights of Dirichlet for data labeled +1
    if alpha_plus:
        alpha_plus_ = np.ones(d)*alpha_plus
    else:
        alpha_plus_ = np.ones(d)/d
    # weights of Dirichlet for data labeled -1
    if alpha_minus:
        alpha_minus_ = np.ones(d)*alpha_minus
    else:  
        alpha_minus_ = np.ones(d)*d
    # Generate angular vectors: train data 
    theta_plus_train  = dirichlet.rvs(alpha=alpha_plus_,
                                      size=n_train,
                                      random_state=rand_seed)    
    theta_minus_train = dirichlet.rvs(alpha=alpha_minus_,
                                      size=n_train,
                                      random_state=rand_seed)
    # Generate angular vectors: test data 
    theta_plus_test   = dirichlet.rvs(alpha=alpha_plus_,  
                                      size=n_test, 
                                      random_state=rand_seed + 700) #+ 700 for test samples
    theta_minus_test  = dirichlet.rvs(alpha=alpha_minus_, 
                                      size=n_test, 
                                      random_state=rand_seed + 700)
    
    # Generate Radius of train data
    R_plus_train   = pareto.rvs(b=1, size=n_train, random_state=rand_seed)
    R_minus_train  = pareto.rvs(b=1, size=n_train, random_state=rand_seed + 123) #+ 123 to change radius
    
    # Generate Radius of test data
    R_plus_test   = pareto.rvs(b=1, size=n_test, random_state=rand_seed + 700)
    R_minus_test  = pareto.rvs(b=1, size=n_test, random_state=rand_seed + 123 + 700)
    
    # Build X = R * theta for train data
    X_plus_train  = R_plus_train.reshape(-1, 1)  * theta_plus_train
    X_minus_train = R_minus_train.reshape(-1, 1) * theta_minus_train
    
    # Build X = R * theta for test data
    X_plus_test  = R_plus_test.reshape(-1, 1)  * theta_plus_test
    X_minus_test = R_minus_test.reshape(-1, 1) * theta_minus_test
    
    # Build V = d * X for train data
    V_plus_train  = d * X_plus_train#[np.min(X_plus_train, axis=1) > 1]
    V_minus_train = d * X_minus_train#[np.min(X_minus_train, axis=1) > 1]
    V_train = np.vstack((V_plus_train, V_minus_train))
    
    is_min_V_train_g_d = np.min(V_train, axis=1) > d 
    
    # Build V = d * X for test data
    V_plus_test  = d * X_plus_test#[np.min(X_plus_test, axis=1) > 1]
    V_minus_test = d * X_minus_test#[np.min(X_minus_test, axis=1) > 1]
    V_test  = np.vstack((V_plus_test, V_minus_test))
    
    # sanity check
    is_min_V_test_g_d = np.min(V_test, axis=1) > d
    
    # Labels for both V_train and V_test 
    y_train = np.hstack((np.ones(n_train), np.zeros(n_train)))
    y_test  = np.hstack((np.ones(n_test),  np.zeros(n_test)))
    
    #label for both train and test for all data
    y_hat_train = np.hstack((np.ones(n_train), np.zeros(n_train)))
    y_hat_test  = np.hstack((np.ones(n_test), np.zeros(n_test)))
    
    # building X_train and X_test
    X_train = np.vstack((X_plus_train, X_minus_train))
    X_test  = np.vstack((X_plus_test, X_minus_test))
    
    # Pareto standardization
    order_X = order(X_train)
    hat_V_train = transform(order_X, X_train)
    hat_V_test  = transform(order_X, X_test)
    
    #Computing norms for train and test
    norm_V_train = np.linalg.norm(V_train, axis=1, ord=np.inf)
    norm_hat_V_train = np.linalg.norm(hat_V_train, axis=1, ord=np.inf)
    
    norm_V_test = np.linalg.norm(V_test, axis=1, ord=np.inf)
    norm_hat_V_test = np.linalg.norm(hat_V_test, axis=1, ord=np.inf) 
    
    # Assessing train samples with norms greater than n / k
    is_extreme_V_train     = norm_V_train >  (2 * n_train) / k # we multiply by 2 cause we divided by 2 before
    is_extreme_hat_V_train = norm_hat_V_train > (2 * n_train) / k
    
    # M value
    M = np.percentile(norm_hat_V_train[is_extreme_hat_V_train], q=quantile)
    if M < 1:
        print("Warning M < 1:", M)
    
    # Assessing train samples with norms smaller than M 
    #(M being a quantile of ||\hat V||[||\hat V|| > n/k])
    is_smaller = norm_hat_V_train <= M 
    
    # Computing angular vectors
    Theta_train = V_train / norm_V_train.reshape(-1, 1)
    Theta_test  = V_test  / norm_V_test.reshape(-1, 1)
     
    hat_Theta_train = hat_V_train / norm_hat_V_train.reshape(-1,1)
    hat_Theta_test  = hat_V_test  / norm_hat_V_test.reshape(-1,1)
    
    # Assessing train samples which are tau far from axes
    is_tau_valid_V_train = np.min(Theta_train, axis=1) > tau
    is_tau_valid_hat_V_train = np.min(hat_Theta_train, axis=1) > tau
    
    # Assessing test samples which are tau far from axes
    is_tau_valid_V_test = np.min(Theta_test, axis=1) > tau
    is_tau_valid_hat_V_test = np.min(hat_Theta_test, axis=1) > tau
    
    # Finding the samples which verify both conditions for F known and F unknown
    #bool_condition_train = (is_extreme_V_train * is_extreme_hat_V_train) * (is_tau_valid_V_train * is_tau_valid_hat_V_train)
    
    #defining the angular train samples on the truncated subspace
    hat_Theta_train_M = hat_Theta_train[is_extreme_hat_V_train * is_tau_valid_hat_V_train * is_smaller ]
    y_hat_train_M = y_hat_train[is_extreme_hat_V_train * is_tau_valid_hat_V_train * is_smaller ]
    
    # defining the angular train samples verifying all conditions    
    Theta_train = Theta_train[is_extreme_V_train * is_tau_valid_V_train]#[bool_condition_train] 
    hat_Theta_train = hat_Theta_train[is_extreme_hat_V_train * is_tau_valid_hat_V_train]
  
    is_extreme_V_test     = norm_V_test >= kappa * (2 * n_train) / k # we multiply by 2 cause we divided by 2 before
    is_extreme_hat_V_test = norm_hat_V_test >= kappa * (2 * n_train) / k
    
    #bool_condition_test  = (is_extreme_V_test  * is_extreme_hat_V_test)  * (is_tau_valid_V_test  * is_tau_valid_hat_V_test)
    
    hat_Theta_test = hat_Theta_test[is_extreme_hat_V_test * is_tau_valid_hat_V_test]
    Theta_test     = Theta_test[is_extreme_V_test * is_tau_valid_V_test]
    
    if verbose:
        # csq of (n_train / k) * tau > d 
        print((is_min_V_train_g_d[is_extreme_V_train * is_tau_valid_V_train]).mean()) 
        print((is_tau_valid_V_train * is_extreme_V_train == is_tau_valid_hat_V_train * is_extreme_hat_V_train).mean())
        
        print("shapes")
        print("Theta_train.shape:",Theta_train.shape)
        print("hat_Theta_train.shape:",hat_Theta_train.shape)
        print("Theta_test.shape:",Theta_test.shape)
        print("hat_Theta_test.shape:",hat_Theta_test.shape)  
    
    # Focusing on extreme samples
    y_train = y_train[is_extreme_V_train * is_tau_valid_V_train]
    y_hat_train = y_hat_train[is_extreme_hat_V_train * is_tau_valid_hat_V_train]
    
    y_test = y_test[is_extreme_V_test * is_tau_valid_V_test]
    y_hat_test = y_hat_test[is_extreme_hat_V_test * is_tau_valid_hat_V_test]
       
    return Theta_train, hat_Theta_train, hat_Theta_train_M, Theta_test, hat_Theta_test, y_train, y_hat_train, y_hat_train_M, y_test, y_hat_test
