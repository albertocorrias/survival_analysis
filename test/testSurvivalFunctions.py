import unittest
import numpy as np
from SurvivalFunctions import kaplan_meier, log_rank_test, cox_prop_haz

class TestSurvivalFunctions(unittest.TestCase):
    
    def test_PlutonianExample(self):
        
        #plutonian example of Stanton Glantz, page 236
        unsorted_time_of_d_or_lost = [7,12,7,12,11,8,9,6,7,2]
        unsorted_type_of_event = [1,1,1,0,0,1,1,1,0,1]
        survival_data = np.column_stack([unsorted_time_of_d_or_lost,unsorted_type_of_event ])
        survival_pluto = kaplan_meier(survival_data, exponential_greenwood = False)#Book uses standard Greenwood for ci

        #Checking times of deaths
        self.assertEqual(len(survival_pluto['KM_times']),7)
        self.assertAlmostEqual(survival_pluto['KM_times'][0],0,4)
        self.assertAlmostEqual(survival_pluto['KM_times'][1],2.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times'][2],6.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times'][3],7.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times'][4],8.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times'][5],9.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times'][6],12.0,4)
        
        #checking survival curve
        self.assertEqual(len(survival_pluto['KM_curve']),7)
        self.assertAlmostEqual(survival_pluto['KM_curve'][0],1.0,4)
        self.assertAlmostEqual(survival_pluto['KM_curve'][1],0.9,2)
        self.assertAlmostEqual(survival_pluto['KM_curve'][2],0.8,2)
        self.assertAlmostEqual(survival_pluto['KM_curve'][3],0.6,2)
        self.assertAlmostEqual(survival_pluto['KM_curve'][4],0.48,2)
        self.assertAlmostEqual(survival_pluto['KM_curve'][5],0.36,2)
        self.assertAlmostEqual(survival_pluto['KM_curve'][6],0.18,2)

        self.assertAlmostEqual(survival_pluto['median_survival_time'], 8.0,4)

        #Checking time of all events
        self.assertEqual(len(survival_pluto['all_times']),8)
        self.assertAlmostEqual(survival_pluto['all_times'][0],0.0,4)
        self.assertAlmostEqual(survival_pluto['all_times'][1],2.0,4)
        self.assertAlmostEqual(survival_pluto['all_times'][2],6.0,4)
        self.assertAlmostEqual(survival_pluto['all_times'][3],7.0,4)
        self.assertAlmostEqual(survival_pluto['all_times'][4],8.0,4)
        self.assertAlmostEqual(survival_pluto['all_times'][5],9.0,4)
        self.assertAlmostEqual(survival_pluto['all_times'][6],11.0,4)
        self.assertAlmostEqual(survival_pluto['all_times'][7],12.0,4)

        #Checking n at start of intervals
        self.assertEqual(len(survival_pluto['n_i']),8)
        self.assertAlmostEqual(survival_pluto['n_i'][0],10,4)
        self.assertAlmostEqual(survival_pluto['n_i'][1],9,4)
        self.assertAlmostEqual(survival_pluto['n_i'][2],8,4)
        self.assertAlmostEqual(survival_pluto['n_i'][3],5,4)
        self.assertAlmostEqual(survival_pluto['n_i'][4],4,4)
        self.assertAlmostEqual(survival_pluto['n_i'][5],3,4)
        self.assertAlmostEqual(survival_pluto['n_i'][6],2,4)
        self.assertAlmostEqual(survival_pluto['n_i'][7],0,4)

        #Checking deaths at intervals
        self.assertEqual(len(survival_pluto['deaths']),8)
        self.assertAlmostEqual(survival_pluto['deaths'][0],0,4)
        self.assertAlmostEqual(survival_pluto['deaths'][1],1,4)
        self.assertAlmostEqual(survival_pluto['deaths'][2],1,4)
        self.assertAlmostEqual(survival_pluto['deaths'][3],2,4)
        self.assertAlmostEqual(survival_pluto['deaths'][4],1,4)
        self.assertAlmostEqual(survival_pluto['deaths'][5],1,4)
        self.assertAlmostEqual(survival_pluto['deaths'][6],0,4)
        self.assertAlmostEqual(survival_pluto['deaths'][7],1,4)

        #Checking lost at intervals
        self.assertEqual(len(survival_pluto['lost']),8)
        self.assertAlmostEqual(survival_pluto['lost'][0],0,4)
        self.assertAlmostEqual(survival_pluto['lost'][1],0,4)
        self.assertAlmostEqual(survival_pluto['lost'][2],0,4)
        self.assertAlmostEqual(survival_pluto['lost'][3],1,4)
        self.assertAlmostEqual(survival_pluto['lost'][4],0,4)
        self.assertAlmostEqual(survival_pluto['lost'][5],0,4)
        self.assertAlmostEqual(survival_pluto['lost'][6],1,4)
        self.assertAlmostEqual(survival_pluto['lost'][7],1,4)       
 
        #Checking plot arrays
        self.assertEqual(len(survival_pluto['KM_times_staircase']),13)
        self.assertAlmostEqual(survival_pluto['KM_times_staircase'][0],0.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times_staircase'][1],2.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times_staircase'][2],2.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times_staircase'][3],6.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times_staircase'][4],6.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times_staircase'][5],7.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times_staircase'][6],7.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times_staircase'][7],8.0,4)             
        self.assertAlmostEqual(survival_pluto['KM_times_staircase'][8],8.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times_staircase'][9],9.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times_staircase'][10],9.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times_staircase'][11],12.0,4)
        self.assertAlmostEqual(survival_pluto['KM_times_staircase'][12],12.0,4)    
        
        self.assertEqual(len(survival_pluto['KM_curve_staircase']),13)
        self.assertAlmostEqual(survival_pluto['KM_curve_staircase'][0],1.0,4)
        self.assertAlmostEqual(survival_pluto['KM_curve_staircase'][1],1.0,4)
        self.assertAlmostEqual(survival_pluto['KM_curve_staircase'][2],0.9,4)
        self.assertAlmostEqual(survival_pluto['KM_curve_staircase'][3],0.9,4)
        self.assertAlmostEqual(survival_pluto['KM_curve_staircase'][4],0.8,4)
        self.assertAlmostEqual(survival_pluto['KM_curve_staircase'][5],0.8,4)
        self.assertAlmostEqual(survival_pluto['KM_curve_staircase'][6],0.6,4)
        self.assertAlmostEqual(survival_pluto['KM_curve_staircase'][7],0.6,4)             
        self.assertAlmostEqual(survival_pluto['KM_curve_staircase'][8],0.48,4)
        self.assertAlmostEqual(survival_pluto['KM_curve_staircase'][9],0.48,4)
        self.assertAlmostEqual(survival_pluto['KM_curve_staircase'][10],0.36,4)
        self.assertAlmostEqual(survival_pluto['KM_curve_staircase'][11],0.36,4)
        self.assertAlmostEqual(survival_pluto['KM_curve_staircase'][12],0.18,4)

        #Checking censored times
        self.assertEqual(len(survival_pluto['times_censored']),3)
        self.assertAlmostEqual(survival_pluto['times_censored'][0],7.0)
        self.assertAlmostEqual(survival_pluto['times_censored'][1],11.0)
        self.assertAlmostEqual(survival_pluto['times_censored'][2],12.0)

         #Checking s_hat at censored times
        self.assertEqual(len(survival_pluto['s_hat_censored']),3)
        self.assertAlmostEqual(survival_pluto['s_hat_censored'][0],0.6)
        self.assertAlmostEqual(survival_pluto['s_hat_censored'][1],0.360) #Value at the previous death (t=9, s_hat=0.36)
        self.assertAlmostEqual(survival_pluto['s_hat_censored'][2],0.18)

        #Check upper bound (Page 236, Stanton glantz book)
        self.assertEqual(len(survival_pluto['KM_upper_bound']),7)
        self.assertEqual(survival_pluto['KM_upper_bound'][0], 1) #We always put that by definition
        self.assertEqual(survival_pluto['KM_upper_bound'][1], 1)
        self.assertEqual(survival_pluto['KM_upper_bound'][2], 1)
        self.assertAlmostEqual(survival_pluto['KM_upper_bound'][3], 0.904,3)
        self.assertAlmostEqual(survival_pluto['KM_upper_bound'][4], 0.801,3)
        self.assertAlmostEqual(survival_pluto['KM_upper_bound'][5], 0.676,3)
        self.assertAlmostEqual(survival_pluto['KM_upper_bound'][6], 0.475,3)

        #Check lower bound (Page 236, Stanton glantz book)
        self.assertEqual(len(survival_pluto['KM_lower_bound']),7)
        self.assertEqual(survival_pluto['KM_lower_bound'][0], 1) #We always put that by definition
        self.assertAlmostEqual(survival_pluto['KM_lower_bound'][1], 0.714,3)
        self.assertAlmostEqual(survival_pluto['KM_lower_bound'][2], 0.552,3)
        self.assertAlmostEqual(survival_pluto['KM_lower_bound'][3], 0.296,3)
        self.assertAlmostEqual(survival_pluto['KM_lower_bound'][4], 0.159,3)
        self.assertAlmostEqual(survival_pluto['KM_lower_bound'][5], 0.044,3)
        self.assertAlmostEqual(survival_pluto['KM_lower_bound'][6], 0)

    def test_AutologousExample(self):
        #Autologous transplant example of Stanton Glantz, page 239
        autologous_time_of_events = [1,1,1,2,2,3,4,5,6,7,8,8,10,12,12,14,17,20,27,27,28,30,30,36,38,40,45,50,50,50,63,132,132]
        autologous_type_of_events = [1,1,1,1,1,1,1,1,1,1,1,1,1 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ,0 ,1 ,1 ,1 ,0 ,0, 0 ]
        survival_data = np.column_stack([autologous_time_of_events,autologous_type_of_events ])
        survival_autologous = kaplan_meier(survival_data, exponential_greenwood = True) #Lifelines uses only exponential greenwood
        
        #Checking times of deaths
        self.assertEqual(len(survival_autologous['KM_times']),18)
        self.assertAlmostEqual(survival_autologous['KM_times'][0],0.0,4)
        self.assertAlmostEqual(survival_autologous['KM_times'][1],1.0,4)
        self.assertAlmostEqual(survival_autologous['KM_times'][2],2.0,4)
        self.assertAlmostEqual(survival_autologous['KM_times'][3],3.0,4)
        self.assertAlmostEqual(survival_autologous['KM_times'][4],4.0,4)
        self.assertAlmostEqual(survival_autologous['KM_times'][5],5.0,4)
        self.assertAlmostEqual(survival_autologous['KM_times'][6],6.0,4)
        self.assertAlmostEqual(survival_autologous['KM_times'][7],7.0,4)
        
        #checking survival curve
        self.assertEqual(len(survival_autologous['KM_curve']),18)
        self.assertAlmostEqual(survival_autologous['KM_curve'][0],1.0,4)
        self.assertAlmostEqual(survival_autologous['KM_curve'][1],0.909,3)
        self.assertAlmostEqual(survival_autologous['KM_curve'][2],0.848,2)
        self.assertAlmostEqual(survival_autologous['KM_curve'][3],0.817,2)
        self.assertAlmostEqual(survival_autologous['KM_curve'][4],0.787,2)
        self.assertAlmostEqual(survival_autologous['KM_curve'][5],0.757,2)
        self.assertAlmostEqual(survival_autologous['KM_curve'][6],0.727,2)
        self.assertAlmostEqual(survival_autologous['KM_curve'][17],0.145,2)

        # from lifelines import KaplanMeierFitter
        # kmf = KaplanMeierFitter()
        # kmf.fit(autologous_time_of_events, event_observed=autologous_type_of_events)  # or, more succinctly, kmf.fit(T, E)
        # print(kmf.survival_function_['KM_estimate']) #and removing duplicates....
        # print(kmf.median_survival_time_) #17.0
        # print(kmf.confidence_interval_) #and removing duplicates....

        lifelines_correct = np.array([1.000000,0.909091, 0.848485, 0.818182, 0.787879, 0.757576, 0.727273, 0.696970, 0.636364, 0.606061, 0.545455, 0.515152, 0.484848,\
                                     0.420202, 0.387879, 0.323232, 0.290909, 0.145455])

        lifelines_correct_upper_95 = [1.000000, 0.969741, 0.933961, 0.913902, 0.892741, 0.870634, 0.847691, 0.823995, \
            0.774568, 0.748918, 0.695875, 0.668512, 0.640597, 0.580315, 0.549152, 0.484674, 0.451282, 0.311110]

        lifelines_correct_lower_95 = [1.000000, 0.744053, 0.673591, 0.639372, 0.605942, 0.573273, 0.541325, 0.510060, \
            0.449452, 0.420060, 0.363027, 0.335371, 0.308287, 0.251233, 0.223929, 0.171861, 0.147176, 0.041452]

        self.assertEqual(len(survival_autologous['KM_times']),len(lifelines_correct_upper_95))
        self.assertEqual(len(survival_autologous['KM_times']),len(lifelines_correct_lower_95))
        self.assertEqual(len(survival_autologous['KM_times']),len(lifelines_correct))
        for i in range(0,len(survival_autologous['KM_times'])):
            self.assertAlmostEqual(survival_autologous['KM_curve'][i],lifelines_correct[i],3)
            self.assertAlmostEqual(survival_autologous['KM_lower_bound'][i],lifelines_correct_lower_95[i],3)
            self.assertAlmostEqual(survival_autologous['KM_upper_bound'][i],lifelines_correct_upper_95[i],3)

        self.assertAlmostEqual(17.0,survival_autologous['median_survival_time'])
        
    def test_AllogenicExample(self):
        allogenic_time_of_events = [1,2,3,4,6,7,12,15,20,21,24,30,60,85,85,86,87,90,100,119,132]
        allogenic_type_of_events = [1,1,1,1,1,1,1 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0  ,0  ,0 ]
        survival_data = np.column_stack([allogenic_time_of_events, allogenic_type_of_events])
        survival_allogenic = kaplan_meier(survival_data)
        
        #Checking times of deaths
        self.assertEqual(len(survival_allogenic['KM_times']),9)
        self.assertAlmostEqual(survival_allogenic['KM_times'][0],0.0,4)
        self.assertAlmostEqual(survival_allogenic['KM_times'][1],1.0,4)
        self.assertAlmostEqual(survival_allogenic['KM_times'][2],2.0,4)
        self.assertAlmostEqual(survival_allogenic['KM_times'][3],3.0,4)
        self.assertAlmostEqual(survival_allogenic['KM_times'][4],4.0,4)
        self.assertAlmostEqual(survival_allogenic['KM_times'][5],6.0,4)
        self.assertAlmostEqual(survival_allogenic['KM_times'][6],7.0,4)
        self.assertAlmostEqual(survival_allogenic['KM_times'][7],12.0,4)
        self.assertAlmostEqual(survival_allogenic['KM_times'][8],24.0,4)
        
        #checking survival curve
        self.assertEqual(len(survival_allogenic['KM_curve']),9)
        self.assertAlmostEqual(survival_allogenic['KM_curve'][0],1.0,4)
        self.assertAlmostEqual(survival_allogenic['KM_curve'][1],0.952,3)
        self.assertAlmostEqual(survival_allogenic['KM_curve'][2],0.904,2)
        self.assertAlmostEqual(survival_allogenic['KM_curve'][3],0.857,2)
        self.assertAlmostEqual(survival_allogenic['KM_curve'][4],0.809,2)
        self.assertAlmostEqual(survival_allogenic['KM_curve'][5],0.762,2)
        self.assertAlmostEqual(survival_allogenic['KM_curve'][6],0.714,2)
        self.assertAlmostEqual(survival_allogenic['KM_curve'][7],0.666,2)
        self.assertAlmostEqual(survival_allogenic['KM_curve'][8],0.605,2)

        self.assertAlmostEqual(survival_allogenic['median_survival_time'], 0) #Curve never drops below 0.5, test defualt value of 0
        
    def test_CompareAutologousAllogenic(self):
        #autologous versus allogenic example. Stanton Glantz, page 240
        allogenic_time_of_events = [1,2,3,4,6,7,12,15,20,21,24,30,60,85,85,86,87,90,100,119,132]
        allogenic_type_of_events = [1,1,1,1,1,1,1 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0  ,0  ,0 ]
        survival_data = np.column_stack([allogenic_time_of_events, allogenic_type_of_events])
        survival_allogenic = kaplan_meier(survival_data)
        
        autologous_time_of_events = [1,1,1,2,2,3,4,5,6,7,8,8,10,12,12,14,17,20,27,27,28,30,30,36,38,40,45,50,50,50,63,132,132]
        autologous_type_of_events =     [1,1,1,1,1,1,1,1,1,1,1,1,1 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ,0 ,1 ,1 ,1 ,0 ,0, 0 ]
        survival_auto = np.column_stack([autologous_time_of_events,autologous_type_of_events])
        survival_autologous = kaplan_meier(survival_auto)
        
        results = log_rank_test(survival_allogenic, survival_autologous)

        self.assertAlmostEqual(results['u_L'],6.575,2)
        self.assertAlmostEqual(results['s_2_l'],7.884,2)
        self.assertAlmostEqual(results['p_value'],0.01925,2)

        results_other_way = log_rank_test(survival_autologous, survival_allogenic)

        self.assertAlmostEqual(results_other_way['u_L'],-6.575,2)
        self.assertAlmostEqual(results_other_way['s_2_l'],7.884,2)
        self.assertAlmostEqual(results_other_way['p_value'],0.01925,2)
        

    def test_log_rank_against_lifelines(self):
        time_control = [14,15,16,18,19,20,21,21,25,26,28,30,60,85,85,86,87,90,\
                            100,119,132]
        type_control = [1,0,1,1,0,1,1 ,0 ,1 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,0\
                            ,1 ,1]
        time_treatment = [1,2,3,4, 5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,\
                             28,29,30,36,38,40,45,48,49,50,52,90,95]
        type_treatment = [1,1,1,1,1,1,1,1,1,1,1,1,1 ,1 ,1 ,1 ,1 ,0 ,1 ,\
                                 1 ,1 ,1 ,1 ,1 ,0 ,0 ,0 ,1 ,1 ,1 ,0 ,1, 0 ]

        km_control = kaplan_meier(np.column_stack([time_control, type_control]))
        km_treatment = kaplan_meier(np.column_stack([time_treatment, type_treatment]))
        lr_control_vs_treat = log_rank_test(km_control,km_treatment)
        self.assertAlmostEqual(lr_control_vs_treat["p_value"], 0.0053,4)#0.0053 is correct, see below
        # from lifelines.statistics import logrank_test
        # results = logrank_test(
        #     durations_A=time_control,
        #     durations_B=time_treatment,
        #     event_observed_A=type_control,
        #     event_observed_B=type_treatment,
        # )
        # print(f"P-value: {results.p_value:.4f}") #0.0053 was printed

    def test_cox_textbook(self):
        time = np.array([2, 3, 4, 5, 6, 7])
        event = np.array([1, 0, 1, 1, 0, 1])
        survival_data = np.column_stack([time, event])
        group = np.array([0, 0, 1, 1, 0, 1])
        
        result = cox_prop_haz(survival_data, group)
        
        self.assertAlmostEqual(result['beta'][0], 0.12853,4)
        self.assertAlmostEqual(result['hazard_ratios'][0], 1.13715,4)
        self.assertAlmostEqual(result['standard_errors'][0], 1.2535,4)
        self.assertAlmostEqual(result['log_likelihood'], -4.2713,4)

        #No ties here, the two methods should give the same answer. Try with Breslow, no efron
        result = cox_prop_haz(survival_data, group, use_efron=False)
        
        self.assertAlmostEqual(result['beta'][0], 0.12853,4)
        self.assertAlmostEqual(result['hazard_ratios'][0], 1.13715,4)
        self.assertAlmostEqual(result['standard_errors'][0], 1.2535,4)
        self.assertAlmostEqual(result['log_likelihood'], -4.2713,4)

    def test_cox_against_lifelines_with_ties(self):
        #Data set with some ties (e.g., t=90)
        time_control = [14,15,16,18,19,20,21,21.5,25,26,28,30,60,85,85.5,86,87,90,\
                            100,119,132]
        type_control = [1,0,1,1,0,1,1 ,0 ,1 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,0\
                            ,1 ,1]
        time_treatment = [1,2,3,4, 5,6,7,8,9,10,11,12,13,14,15,16,19,20,21.8,22,\
                             28,29,30,36,38,40,45,48,49,50,52,90,95]

        type_treatment = [1,1,1,1,1,1,1,1,1,1,1,1,1 ,1 ,1 ,1 ,1 ,0 ,1 ,\
                                 1 ,1 ,1 ,1 ,1 ,0 ,0 ,0 ,1 ,1 ,1 ,0 ,1, 0 ]
        group = np.concatenate((np.zeros(len(time_control)),np.ones(len(time_treatment))))
        # Time: Duration until event or censoring
        time = np.concatenate((time_control,time_treatment))
        # Event: 1 if event occurred (death/failure), 0 if censored
        event = np.concatenate((type_control,type_treatment))

        # import pandas as pd
        # from lifelines import CoxPHFitter
        
        # #Combine NumPy arrays into a pandas DataFrame (Required by lifelines)
        # data = pd.DataFrame({
        #     'treatment': group,
        #     'duration': time,
        #     'event': event
        # })

        # # Initialize and fit the Cox Proportional Hazards model
        # cph = CoxPHFitter()
        # cph.fit(data, duration_col='duration', event_col='event')
        # cph.print_summary()
        #GENERATED OUTPUT
        # <lifelines.CoxPHFitter: fitted with 54 total observations, 14 right-censored observations>
        #             duration col = 'duration'
        #                 event col = 'event'
        #     baseline estimation = breslow
        # number of observations = 54
        # number of events observed = 40
        # partial log-likelihood = -123.31
        #         time fit was run = 2026-10-01 05:28:28 UTC

        # ---
        #         coef exp(coef)  se(coef)  coef lower 95%  coef upper 95% exp(coef) lower 95% exp(coef) upper 95%
        # covariate                                                                                                  
        # treatment  0.97      2.65      0.36            0.26            1.69                1.30                5.41

        #         cmp to    z    p  -log2(p)
        # covariate                            
        # treatment    0.00 2.67 0.01      7.04
        # ---
        # Concordance = 0.63
        # Partial AIC = 248.62
        # log-likelihood ratio test = 7.88 on 1 df
        # -log2(p) of ll-ratio test = 7.64

        myresult = cox_prop_haz(np.column_stack([time,event]), group)#use Efron like lifelines
        self.assertAlmostEqual(myresult['log_likelihood'],-123.31,2)#partial log-likelihood  in the lifelines output above
        self.assertAlmostEqual(myresult['hazard_ratios'][0],2.65,2) #exp(coef) in the lifelines output above
        self.assertAlmostEqual(myresult['beta'][0],0.97,2) #exp(coef) in the lifelines output above
        self.assertAlmostEqual(myresult['upper_bounds'][0],5.41,2) #exp(coef) upper in the lifelines above
        self.assertAlmostEqual(myresult['lower_bounds'][0],1.30,2) #exp(coef) lower in the lifelines above




    def test_cox_against_lifelines_multiple_covariates(self):
        
        #The following code was run after pip install lifelines

        # from lifelines.datasets import load_regression_dataset
        # from lifelines import CoxPHFitter
        # import pandas as pd
        # regression_dataset = load_regression_dataset() # a Pandas DataFrame
        # # Using Cox Proportional Hazards model
        # cph = CoxPHFitter()
        # cph.fit(regression_dataset, 'T', event_col='E')
        # cph.print_summary()
        # #Convert to numpy for storing
        # matrix = regression_dataset.to_numpy()
        # np.savetxt("test/data/lifeline_data.txt", matrix)

        #GENERATED OUTPUT (see https://lifelines.readthedocs.io/en/latest/Quickstart.html#survival-regression)
        #
        #lifelines.CoxPHFitter: fitted with 200 total observations, 11 right-censored observations>
        #               duration col = 'T'
        #                    event col = 'E'
        #        baseline estimation = breslow
        #    number of observations = 200
        #    number of events observed = 189
        #    partial log-likelihood = -807.62
        #            time fit was run = 2026-08-05 10:12:38 UTC

        #           coef exp(coef)  se(coef)  coef lower 95%  coef upper 95% exp(coef) lower 95% exp(coef) upper 95%
        #    covariate                                                                                                  
        #    var1       0.22      1.25      0.07            0.08            0.37                1.08                1.44
        #    var2       0.05      1.05      0.08           -0.11            0.21                0.89                1.24
        #    var3       0.22      1.24      0.08            0.07            0.37                1.07                1.44

         #           cmp to    z      p  -log2(p)
         #   covariate                              
         #   var1         0.00 2.99 <0.005      8.49
         #   var2         0.00 0.61   0.54      0.89
         #   var3         0.00 2.88 <0.005      7.97
         #   ---
         #   Concordance = 0.58
         #   Partial AIC = 1621.24
         #   log-likelihood ratio test = 15.54 on 3 df
         #   -log2(p) of ll-ratio test = 9.47

        
        lifeline_data = np.genfromtxt('test/data/lifeline_data.txt')
        var_1 = lifeline_data[:,0]
        var_2 = lifeline_data[:,1]
        var_3 = lifeline_data[:,2]
        times = lifeline_data[:,3]
        events = lifeline_data[:,4]
        for i in range(len(events)):
            events[i] = int(events[i])
        survival_data = np.column_stack([times, events])
        vars_l = np.column_stack([var_1,var_2,var_3])
        
        result = cox_prop_haz(survival_data,vars_l)
        self.assertAlmostEqual(result['beta'][0],0.22,2)#coef in the lifelines output
        self.assertAlmostEqual(result['beta'][1],0.05,2)#coef in the lifelines output
        self.assertAlmostEqual(result['beta'][2],0.22,2)#coef in the lifelines output
        self.assertAlmostEqual(result['standard_errors'][0],0.07,2)#se coef in the lifelines output
        self.assertAlmostEqual(result['standard_errors'][1],0.08,2)#se coef in the lifelines output
        self.assertAlmostEqual(result['standard_errors'][2],0.08,2)#se coef in the lifelines output
        self.assertAlmostEqual(result['log_likelihood'],-807.62,2)#partial log-likelihood  in the lifelines output
        self.assertAlmostEqual(result['hazard_ratios'][0],1.25,2) #exp(coef) in the lifelines output
        self.assertAlmostEqual(result['hazard_ratios'][1],1.05,2) #exp(coef) in the lifelines output
        self.assertAlmostEqual(result['hazard_ratios'][2],1.24,2) #exp(coef) in the lifelines output
        self.assertAlmostEqual(result['upper_bounds'][0],1.44,2) #exp(coef) upper in the lifelines output
        self.assertAlmostEqual(result['upper_bounds'][1],1.24,2) #exp(coef) upper in the lifelines output
        self.assertAlmostEqual(result['upper_bounds'][2],1.44,2) #exp(coef) upper in the lifelines output
        self.assertAlmostEqual(result['lower_bounds'][0],1.08,2) #exp(coef) lower in the lifelines output
        self.assertAlmostEqual(result['lower_bounds'][1],0.89,2) #exp(coef) lower in the lifelines output
        self.assertAlmostEqual(result['lower_bounds'][2],1.07,2) #exp(coef) lower in the lifelines output

    def test_two_deaths_at_one_time(self):
        time = np.array([2, 3.5, 4, 4, 12, 15, 25, 30, 32, 36, 39, 42]) #Note two deaths at t=4
        event = np.array([1, 0,  1, 1,  1,  0,  1,  1,  0,  1,  1,  1])#1 is death, 0 is censored 
        survival_data = np.column_stack([time, event])
        km = kaplan_meier(survival_data)
        km_times = km["KM_times"]
        km_curve = km["KM_curve"]
        km_upper_bound = km["KM_upper_bound"]
        km_lower_bound = km["KM_lower_bound"]
        
        # from lifelines import KaplanMeierFitter
        # kmf = KaplanMeierFitter()
        # kmf.fit(time, event_observed=event)  # or, more succinctly, kmf.fit(T, E)
        # print(kmf.survival_function_['KM_estimate']) #Output was
        # 0.0     1.000000
        # 2.0     0.916667
        # 3.5     0.916667
        # 4.0     0.733333
        # 12.0    0.641667
        # 15.0    0.641667
        # 25.0    0.534722
        # 30.0    0.427778
        # 32.0    0.427778
        # 36.0    0.285185
        # 39.0    0.142593
        # 42.0    0.000000
        #Removing censored-only points 3.5, 13, and 32 (should not be in KM curve)
        lifelines_time_correct = np.array([0.0,2.0, 4.0, 12.0, 25.0, 30.0, 36.0, 39.0, 42.0])
        lifelines_km_correct = np.array([1.0,0.916667, 0.733333, 0.641667, 0.534722, 0.427778, 0.285185, 0.142593, 0.0])
        self.assertEqual(len(lifelines_km_correct),len(lifelines_time_correct))
        self.assertEqual(len(km_curve),len(lifelines_time_correct))
        self.assertEqual(len(km_times),len(lifelines_time_correct))
        for i in range(0,len(km_curve)):
            self.assertAlmostEqual(lifelines_time_correct[i], km_times[i],4)
            self.assertAlmostEqual(lifelines_km_correct[i], km_curve[i],4)
        
        #print(kmf.median_survival_time_) #Output was 30.0
        self.assertAlmostEqual(km['median_survival_time'],30)#

        #print(kmf.confidence_interval_) #output was...
        # 0.0                 1.000000                1.000000
        # 2.0                 0.538977                0.987826
        # 3.5                 0.538977                0.987826
        # 4.0                 0.378961                0.905617
        # 12.0                0.302250                0.848294
        # 15.0                0.302250                0.848294
        # 25.0                0.212410                0.776504
        # 30.0                0.138735                0.694157
        # 32.0                0.138735                0.694157
        # 36.0                0.052140                0.586906
        # 39.0                0.008296                0.453081
        # 42.0                0.000000                0.000000
        #Removing times of only censored data (3.5,13,32)
        lifelines_upper_correct = np.array([1.0, 0.987826, 0.905617, 0.848294, 0.776504, 0.694157, 0.586906, 0.453081, 0.0])
        lifelines_lower_correct = np.array([1.0, 0.538977, 0.378961, 0.302250, 0.212410, 0.138735, 0.052140, 0.008296,  0.0])
        self.assertEqual(len(lifelines_upper_correct),len(lifelines_lower_correct))
        self.assertEqual(len(km_lower_bound),len(lifelines_lower_correct))
        self.assertEqual(len(km_upper_bound),len(lifelines_upper_correct))
        for i in range(0,len(lifelines_upper_correct)):
            self.assertAlmostEqual(lifelines_upper_correct[i], km_upper_bound[i],4)
            self.assertAlmostEqual(lifelines_lower_correct[i], km_lower_bound[i],4)

