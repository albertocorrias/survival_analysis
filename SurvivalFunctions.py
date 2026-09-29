import numpy as np
from scipy import stats

def kaplan_meyer(survival_data, alpha=0.95,exponential_greenwood=True):
    """
    Calculates Kaplan Meyer survival curves
    
    Parameters
    ----------
        survival_data: 
            a Numpy matrix with two columns.
            Each row of the matrix corresponds to an individual. 
            The first columns contains the time, since enrollment, of an event related 
            to the individual. The event can be detah or lost to follow up (censored)
            If the event is death, the corresponding element in the second column is 1
            If the event is not death, the corresponding element in the second column is 0
            The patients do not need to be sorted in any way
        alpha: fraction corresponding to the desired percentage confidence intervals.
                defaults to 0.95 (95% confience intervals).
        exponential_greenwood: If True, the confidence intervals are calculated accoridng to the 
                            exponential Greenwood formula. If False, 
                            the standard Greenwood formula is used. Default is True
    Returns
    --------
        A dictionary with the following keys

        KM_times. Kaplan Meyer times. It is an array with all the times
                 of deaths in chronological order (first element is zero)
        KM_curve. Kaplan Meyer curve. It is an array with the values of 
                  the Kaplan Meyer survival curve S_hat (first element is 1)
        KM_upper_bound. The upper bound of the 100*alpha percent confidence interval for KM_curve. Max is 1
        KM_lower_bound. The upper bound of the 100*alpha percent confidence interval for KM_curve. Min is 0
        KM_times_staircase. Similar to KM_times, but it is an array that 
                            can be used for plotting time versus survival 
                            in the typical "staircase" plots.
        KM_curve_staircase. Similar to KM_curve, but it is an array that 
                            can be used for plotting time versus survival in 
                            the typical "staircase" plots. 
        median_survival_time. The median survival time, defined as the first 
                              time the KM curve drops below 0.5. 
                              It is 0 if the curve never drops below 0.5
        times_censored. The times when a censored observation is. Useful for plotting
        s_hat_censored. The value of the KM curve at the censored time. Useful for plotting
        all_times. It is a list with all the times of event (deaths or otherwise)
        n_i. It is a list with the total number of 
             individuals still alive just before the corresponding 
             time in the array 'all_times'
        deaths. It is a list with the total number of individuals who dies at the 
                corresponding time in the array 'all_times'
        lost. It is a list  with the total number of individuals 
        who are lost to follow up at the corresponding time in the array 'all_times'
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
    times_lost = [] #Times of cenosred data (for plotting)
    s_hat_lost = [] #Corresponding values of of KM curve
    upper_bound = [1.0]
    lower_bound = [1.0]

    se_terms =[] #Stores d_i/(ni*(ni-di))
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
            S_hat_plot.append(S_hat[-1])#First append for staircase effect
            times_of_death_plot.append(time_of_interest)#First append for staircase effect
            new_surv_frac = float(n_i[-1]-deaths_at_i)/n_i[-1]            
            new_value = S_hat[-1]*new_surv_frac
            S_hat.append(new_value)
            S_hat_plot.append(new_value)#Second append for staircase effect
            times_of_death.append(time_of_interest)
            times_of_death_plot.append(time_of_interest)#Second append for staircase effect

            #Calculate confidence intervals
            z_val = stats.norm.ppf(1 - (1-alpha)/2)
            #First we store the term for the SE
            if (n_i[-1]>0 and np.fabs(n_i[-1] - deaths_at_i)>1e-8):
                se_terms.append(deaths_at_i/(n_i[-1]*(n_i[-1]- deaths_at_i)))
            #Summation in the Greenwood formulas
            #see https://www.math.wustl.edu/%7Esawyer/handouts/greenwood.pdf
            summ=0
            for k in range(0,len(se_terms)):
                summ = summ + se_terms[k]
            if (exponential_greenwood == False):
                stand_err = new_value*np.sqrt(summ) #Greenwood formula
                upper_bound.append(min(1, new_value + z_val*stand_err))
                lower_bound.append(max(0, new_value - z_val*stand_err))
            else: #Use exponential Greenwood
                if (np.abs(new_value - 1) > 1e-6 and new_value > 0):#Avoid runtime warnings
                    stand_err = np.sqrt((1/(np.log(new_value)**2))*summ)
                    c_plus = np.log(-np.log(new_value)) + z_val*stand_err
                    c_minus = np.log(-np.log(new_value)) - z_val*stand_err
                    upper_bound.append(np.exp(-np.exp(c_minus)))
                    lower_bound.append(np.exp(-np.exp(c_plus)))
        
        if (lost_at_i > 0):
            times_lost.append(time_of_interest)
            s_hat_lost.append(S_hat_plot[-1])
            
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
        'KM_times' : np.array(times_of_death),
        'KM_curve' : np.array(S_hat),
        'KM_times_staircase' : np.array(times_of_death_plot),
        'KM_curve_staircase' : np.array(S_hat_plot),
        'KM_upper_bound' : np.array(upper_bound),
        'KM_lower_bound' : np.array(lower_bound),
        'median_survival_time' : median_surv,
        'times_censored' : np.array(times_lost),
        's_hat_censored' : np.array(s_hat_lost),
        'all_times' : all_times,
        'n_i' : np.array(n_i),
        'deaths' : np.array(d_i),
        'lost' : np.array(lost_i)
    }

def log_rank_test(survival_1, survival_2):
    """
    Performs a log-rank test between two Kaplan Meyer survival curves.

    NOTE: The test statistic of the log-rank test is computed with survival_2 as the reference:
    if 'u_L' is positive, it means more deaths than expected for "survival_2".
    
    Therefore, assuming the p_value is small enough for the scenario of interest, 
    then, if the sign of 'u_L' is positive, it means data support survival_1 improves over survival_2.

    Parameters
    ----------
        survival_1: 
            This is expected to be a dictionary returned by the kaplan_meyer function
        survival_2:
            This is expected to be a dictionary returned by the kaplan_meyer function
    Returns
    --------
        A dictionary with the following keys

        p_value:
          The p-value of the log-rank test (2-sided)

        u_L:
         The numerator of the test statistic of the log-rank test

        s_2_l:
         The square of the denominator of the test statistic of the log-rank test
    """

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
            f_i = float(d_total_i)/n_total_i#float is key otherwise it may do an integer division
            e_i = n_2_i*f_i
            o_minus_e = d_2_i - e_i
            u_l = u_l + o_minus_e
            if (n_total_i>1):
                s_2_l = s_2_l + (float(n_1_i)*n_2_i*d_total_i*(n_total_i-d_total_i))/(n_total_i*n_total_i*(n_total_i-1))
    
    z_stat = u_l/np.sqrt(s_2_l)
    p_value = 2.0*(1.0-stats.norm.cdf(abs(z_stat)))
    return {'u_L' : u_l,
            's_2_l' : s_2_l,
            'p_value' : p_value}

def cox_prop_haz(survival_data, X, alpha=0.05, max_iter=50, tol=1e-8):
    """
    Fit a Cox proportional hazards model using Newton-Raphson.

    Parameters
    ----------
    survival_data: 
        a Numpy matrix with two columns and n rows.
        Each row of the matrix corresponds to an individual. 
        The first columns contains the time, since enrollment, of an event related 
        to the individual. The event can be detah or lost to follow up (censored)
        If the event is death, the corresponding element in the second column is 1
        If the event is not death, the corresponding element in the second column is 0
        The patients do not need to be sorted in any way

    X : array-like, shape (n, p)
        Covariate matrix. p is the number of covariates. 
        For two groups, p=1 and X can be a single column
        with 0 = control and 1 = treatment.

    alpha: float
        The desired significance level associated with the confidence intervals. 
        If 95% confidence intervals are wanted, then alpha=0.05. Defaults to 0.05.

    max_iter : int
        Maximum Newton-Raphson iterations. Defaults to 50.

    tol : float
        Convergence tolerance for the Newton Raphson iterations. Defaults to 1e-8.

    Returns
    -------
    beta : ndarray, shape (p,)
        Estimated Cox coefficients.

    hr : ndarray, shape (p,)
        Hazard ratios, exp(beta).

    upper_bounds : ndarray, shape (p,)
        The upper bounds of the Hazard Ratios (i.e., the upper bound of exp(beta) )
    
    lower_bounds : ndarray, shape (p,)
        The lower  bounds of the Hazard Ratios (i.e., the lower bound of exp(beta) )

    se : ndarray, shape (p,)
        Approximate standard errors of the coefficients beta.

    loglik : float
        Final partial log-likelihood.
    """

    time = survival_data[:,0]
    event =  np.round(survival_data[:,1]).astype(int) #Rounded to nearest integer, then cast to make sure == operator works
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape # n rows, p columns
    beta = np.zeros(p)
    risk_mask = np.zeros(n,dtype=bool)
    tied_event_mask = np.zeros(n,dtype=bool)
    unique_times = np.sort(np.unique(time[event == 1]))#Here is where the fact that death=1 in the second column is assumed

    numerical_stability_delta = 1e-9 #Small delta to improve numerical stability of nearly singular hessian matrices
    float_tol = 1e-12 #Anpther tolerance to check equality of two floating point times
    
    #Main Newton-Raphson iterative loop
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
            weights_col = weights.reshape(len(weights),1)#make this array (n,1) for multiplication below
            #Risk summations 
            sum_exp_term = np.sum(weights_col) #Summation of the pure exponential term
            S1 = np.sum(X_risk * weights_col, axis=0) #Multiplies each column of X_risk by weights_col
            
            x_bar = S1 / sum_exp_term #see slides

            #Breslow tie-breaker, count deaths
            for pt in range(0,n):
                if (np.fabs(time[pt] - unique_t)<float_tol):
                    tied_event_mask[pt] = True
                else:
                    tied_event_mask[pt] = False
            X_event = X[tied_event_mask]
            d_t = X_event.shape[0] #number of deaths inferred by the shape

            x_event_sum = np.sum(X_event, axis=0)
            loglik += np.sum(X_event.dot(beta)) - d_t * np.log(sum_exp_term) #l(beta) in the slides

            num_hess = (X_risk.T).dot(X_risk * weights_col) #The term at the numerator of the first term of the Hessian
            hessian_term = num_hess / sum_exp_term - np.outer(x_bar, x_bar) #Term in Hessian summation formula, see slides

            U_beta += x_event_sum - d_t * x_bar #See formula in the slides
            hessian -= d_t * hessian_term

        hessian_stable = hessian - numerical_stability_delta * np.eye(p) #stabilize hessian for nearly singular cases (AI suggestion here)
        # Newton-Raphson step. Conceptually, this is
        # beta_new = beta - inv(H) * U_beta
        #Calling solve is much faster though and does the same thing
        step = np.linalg.solve(hessian_stable, U_beta)
        beta_new = beta - step

        #Update beta
        beta = beta_new
        
        if np.max(np.abs(U_beta)) < tol:#Checking that U_beta is zero #TODO possible to add convergence on |beta_new-beta| before updating
            break

    # Observed information is -Hessian at convergence
    variance = np.linalg.inv(-hessian)
    standard_errors  =np.sqrt(np.diag(variance))

    exp_upper_bounds = np.exp(beta + np.abs(stats.norm.ppf(0.5*alpha))*standard_errors)
    exp_lower_bounds = np.exp(beta - np.abs(stats.norm.ppf(0.5*alpha))*standard_errors)
    ret = {
        'hazard_ratios' : np.exp(beta),
        'standard_errors' : standard_errors,
        'beta' : beta,
        'log_likelihood' : loglik,
        'upper_bounds' : exp_upper_bounds,
        'lower_bounds' : exp_lower_bounds
    }
    return ret




