import numpy as np


def compute_survival(survival_data):
    """
    Calculates Kaplan Meyer survival curves
    
    Parameters
    ----------
        survival_data: 
            a Numpy matrix with two columns.
            Each row of the matrix corresponds to an individual. 
            The first columns contains the time, since enrollment, of an event related 
            to the individual. The event can be detah or lost to follow up
            If the event is death, the correspomnding element in the second column is 1
            If the event is not death, the correspomnding element in the second column is 0
            The patients do not need to be sorted in any way
    Returns
    --------
        A dictionary with the following keys

        KM_times. Kaplan Meyer times. It is an array with all the times
                 of deaths in chronological order (first element is zero)
        KM_curve. Kaplan Meyer curve. It is an array with the values of 
                  the Kaplan Meyer survival curve S_hat (first element is 1)
        KM_times_staircase. Similar to KM_times, but it is an array that 
                            can be used for plotting time versus survival 
                            in the typical "staircase" plots.
        KM_curve_staircase. Similar to KM_curve, but it is an array that 
                            can be used for plotting time versus survival in 
                            the typical "staircase" plots. 
        median_survival_time. The median survival time, defined as the first time the KM curve drops below 0.5. It is 0 if the curve never drops below 0.5
        all_times. It is an array with all the times of event (deaths or otherwise)
        n_i: It is an array with the total number of 
             individuals still alive just before the corresponding 
             time in the array 'all_times'
        deaths: It is an array with the total number of individuals who dies at the corresponding time in the array 'all_times'
        lost: It is an array with the total number of individuals who are lost to follow up at the corresponding time in the array 'all_times'
    """

    unsorted_time_of_events  =survival_data[:,0]
    unsorted_type_of_events = survival_data[:,1]
    #argosrt will internally sort the array and give the original indices in order 
    sorted_time_indices = np.argsort(unsorted_time_of_events)
    time_of_events = np.zeros(len(sorted_time_indices))
    type_of_events = np.zeros(len(sorted_time_indices))
    for i in range (0,len(sorted_time_indices)):
        sorted_index = sorted_time_indices[i]
        time_of_events[i] =unsorted_time_of_events[sorted_index]
        if (unsorted_type_of_events[sorted_index]==1):
            type_of_events[i]=1
        
    N = len(time_of_events)
    total_surviving = N
    n_i = [N]
    times_of_death=[0.0]
    times_of_death_plot=[0.0]
    S_hat = [1.0]
    S_hat_plot = [1.0]
    d_i = [0]
    lost_i = [0]
    i=0
    all_times = [0.0]
    while(i<N):
        time_of_interest = time_of_events[i]
        #determine number of events at this time
        n_events = np.count_nonzero(time_of_events == time_of_interest)
        deaths_at_i=0
        lost_at_i = 0
        for j in range (i,i+n_events):
            if (type_of_events[j]==1):#There was a death
                deaths_at_i=deaths_at_i+1
            else:
                lost_at_i=lost_at_i+1
           
        if (deaths_at_i>0):            
            S_hat_plot.append(S_hat[-1])
            times_of_death_plot.append(time_of_interest)
            new_surv_frac = float(n_i[-1]-deaths_at_i)/n_i[-1]
            new_value = S_hat[-1]*new_surv_frac
            S_hat.append(new_value)
            S_hat_plot.append(new_value)
            times_of_death.append(time_of_interest)
            times_of_death_plot.append(time_of_interest)
            
        all_times.append(time_of_interest)

        i=i+deaths_at_i+lost_at_i

        total_surviving = total_surviving - deaths_at_i - lost_at_i   
        n_i.append(total_surviving)
        d_i.append(deaths_at_i)
        lost_i.append(lost_at_i)

    median_surv = 0
    for i in range (0,len(S_hat)):
        if (S_hat[i] < 0.5):
            median_surv = times_of_death[i]
            break

    return  {
        'KM_times' : times_of_death,
        'KM_curve' : S_hat,
        'KM_times_staircase' : times_of_death_plot,
        'KM_curve_staircase' : S_hat_plot,
        'median_survival_time' : median_surv,
        'all_times' : all_times,
        'n_i' : n_i,
        'deaths' : d_i,
        'lost' : lost_i
    }

def compare_survivals(survival_1, survival_2):
    '''
    This function takes in two survival curves in the format that is returned by 
    the compute_survival functions. It goes through them and calculates the values
    of u_L and s_L^2 that are necessary to perform the log-rank test.
    u_L and s_L^2 are returned in a dictionary (u_L and  s_2_l are the keys)
    '''
    #First, obtain, from the two, all the times where we need to do something
    to_be_added = []
    for i in range(0, len(survival_2['all_times'])):
        position = np.where(np.isclose(survival_1['all_times'],survival_2['all_times'][i],1e-4))
        if (len(position[0])==0):#if not there, we add it
            to_be_added.append(survival_2['all_times'][i])
    
    all_times = survival_1['all_times'] + to_be_added
    all_times.sort()
    
    u_l=0.0
    s_2_l=0.0
    n_1_i = survival_1['n_i'][0]
    n_2_i = survival_2['n_i'][0]
    d_1_i = 0
    d_2_i = 0
    lost_1_i = 0
    lost_2_i = 0
    for i in range(1,len(all_times)):#Note we ignore t=0.
        #Check whether this time in the first, second or both
        pos_2 = np.where(np.isclose(survival_2['all_times'],all_times[i],1e-4))
        pos_1 = np.where(np.isclose(survival_1['all_times'],all_times[i],1e-4))
        if len(pos_2[0])>0 :
            d_2_i  = survival_2['deaths'][pos_2[0][0]]
            n_2_i  = survival_2['n_i'][pos_2[0][0]-1]
            lost_2_i = survival_2['lost'][pos_2[0][0]]
        else:
            n_2_i = n_2_i - d_2_i - lost_2_i
            d_2_i = 0
            lost_2_i =0
                
        if len(pos_1[0])>0 :
            d_1_i  = survival_1['deaths'][pos_1[0][0]]
            n_1_i  = survival_1['n_i'][pos_1[0][0]-1]
            lost_1_i  = survival_1['lost'][pos_1[0][0]]
        else:
            n_1_i = n_1_i - d_1_i - lost_1_i
            d_1_i  = 0
            lost_1_i  = 0
    
        if (d_2_i>0 or d_1_i>0):#u_L is computed only when there is a death
            d_total_i = d_2_i + d_1_i
            n_total_i = n_2_i + n_1_i
            f_i = float(d_total_i)/n_total_i;#float is key otherwise it may do an integer division
            e_i = n_2_i*f_i
            o_minus_e = d_2_i - e_i
            u_l = u_l + o_minus_e
            if (n_total_i>1):
                s_2_l = s_2_l + (float(n_1_i)*n_2_i*d_total_i*(n_total_i-d_total_i))/(n_total_i*n_total_i*(n_total_i-1))
                
    return {'u_L' : u_l,
            's_2_l' : s_2_l}

def cox_newton_raphson(survival_data, X, max_iter=50, tol=1e-8):
    """
    Fit a Cox proportional hazards model using Newton-Raphson.

    Parameters
    ----------
    survival_data: 
        a Numpy matrix with two columns.
        Each row of the matrix corresponds to an individual. 
        The first columns contains the time, since enrollment, of an event related 
        to the individual. The event can be detah or lost to follow up
        If the event is death, the correspomnding element in the second column is 1
        If the event is not death, the correspomnding element in the second column is 0
        The patients do not need to be sorted in any way

    X : array-like, shape (n, p)
        Covariate matrix. p is the number of covariates. 
        For two groups, p=1 and X can be a single column
        with 0 = control and 1 = treatment.

    max_iter : int
        Maximum Newton-Raphson iterations.

    tol : float
        Convergence tolerance.

    Returns
    -------
    beta : ndarray, shape (p,)
        Estimated Cox coefficients.

    hr : ndarray, shape (p,)
        Hazard ratios, exp(beta).

    se : ndarray, shape (p,)
        Approximate standard errors.

    loglik : float
        Final partial log-likelihood.
    """

    time = survival_data[:,0]
    event = survival_data[:,1] 
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape # n rows, p columns
    beta = np.zeros(p)
    risk_mask = np.zeros(n,dtype=bool)
    tied_event_mask = np.zeros(n,dtype=bool)
    unique_times = np.sort(np.unique(time[event == 1]))
    small_delta = 1e-9 #Small delta to improve numerical stability of nearly singular hessian matrices
    for iteration in range(max_iter):
        loglik = 0.0
        U_beta = np.zeros(p) #This is U(beta) in the slides->the derivative of l(beta) ->the one to be put to 0
        hessian = np.zeros((p, p))

        # Loop over unique observed events
        for unique_t in unique_times:

            # Risk set: everyone still at risk at unique_t (we will mark those as "true")
            for r in range(0,n):
                if (time[r] >= unique_t):
                    risk_mask[r] = True
                else:
                    risk_mask[r] = False
            
            X_risk = X[risk_mask] #Isolate only those at risk (this is "i belonging to Rj" in the slides)

            x_i_transp_beta = X_risk.dot(beta) #x_i * beta in the slides
            weights = np.exp(x_i_transp_beta) #Exponential of the above
            weights_col = weights.reshape(len(weights),1)#make this aray (n,1) for multiplication below
            #Risk summations 
            sum_exp_term = np.sum(weights_col) #Summation of the pure exponential term
            S1 = np.sum(X_risk * weights_col, axis=0) #Multiplies each column of X_risk by weights_col
            S2 = (X_risk.T).dot(X_risk * weights_col) #The term 

            x_bar = S1 / sum_exp_term #see slides

            #Breslow tie-breaker, count deaths
            for pt in range(0,n):
                if (time[pt] == unique_t):#TODO make it robust for non-integers
                    tied_event_mask[pt] = True
                else:
                    tied_event_mask[pt] = False
            X_event = X[tied_event_mask]
            d_t = X_event.shape[0] #number of deaths inferred by the shape

            x_event_sum = np.sum(X_event, axis=0)
            loglik += np.sum(X_event.dot(beta)) - d_t * np.log(sum_exp_term) 

            hessian_term = S2 / sum_exp_term - np.outer(x_bar, x_bar) #Term in Hessian summation formula

            U_beta += x_event_sum - d_t * x_bar #See formula in the slides
            hessian -= d_t * hessian_term

        hessian_stable = hessian - small_delta * np.eye(p) #stabilize hessian for nearly singular cases (AI suggestion here)
        # Newton-Raphson step:
        # beta_new = beta - inv(H) * U_beta
        # Faster solve instead of explicitly calculating inv(H)
        step = np.linalg.solve(hessian_stable, U_beta)
        beta_new = beta - step

        if np.max(np.abs(U_beta)) < tol:#Checking that U_beta is zero #TODO possible to add convergence on |beta_new-beta|
            beta = beta_new
            break

        beta = beta_new

    # Observed information is -Hessian at convergence
    variance = np.linalg.inv(-hessian)

    ret = {
        'hazard_ratios' : np.exp(beta),
        'standard_errors' : np.sqrt(np.diag(variance)),
        'beta' : beta,
        'log_likelihood' : loglik
    }
    return ret




