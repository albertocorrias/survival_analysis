import unittest
import numpy as np
from SurvivalFunctions import kaplan_meyer, log_rank_test, cox_prop_haz

#[times_of_death,S_hat,all_times, n_i, d_i, lost_i, times_of_death_plot, S_hat_plot]
class TestSurvivalFunctions(unittest.TestCase):
    
    def test_PlutonianExample(self):
        
        #plutonian example of Stanton Glantz, page 236
        unsorted_time_of_d_or_lost = [7,12,7,12,11,8,9,6,7,2]
        unsorted_type_of_event = [1,1,1,0,0,1,1,1,0,1]
        survival_data = np.column_stack([unsorted_time_of_d_or_lost,unsorted_type_of_event ])
        survival_pluto = kaplan_meyer(survival_data)

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
        self.assertAlmostEqual(survival_pluto['s_hat_censored'][1],0.360) #VAlueat the previous death (t=9, s_hat=0.36)
        self.assertAlmostEqual(survival_pluto['s_hat_censored'][2],0.18)

    def test_AutologousExample(self):
        #Autologous transplant example of Stanton Glantz, page 239
        autologous_time_of_events = [1,1,1,2,2,3,4,5,6,7,8,8,10,12,12,14,17,20,27,27,28,30,30,36,38,40,45,50,50,50,63,132,132]
        autologous_type_of_events =     [1,1,1,1,1,1,1,1,1,1,1,1,1 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ,0 ,1 ,1 ,1 ,0 ,0, 0 ]
        survival_data = np.column_stack([autologous_time_of_events,autologous_type_of_events ])
        survival_autologous = kaplan_meyer(survival_data)
        
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

        lifelines_correct = np.array([1.000000,0.909091, 0.848485, 0.818182, 0.787879, 0.757576, 0.727273, 0.696970, 0.636364, 0.606061, 0.545455, 0.515152, 0.484848,\
                                     0.420202, 0.387879, 0.323232, 0.290909, 0.145455])
        
        self.assertEqual(len(survival_autologous['KM_times']),len(lifelines_correct))
        for i in range(0,len(survival_autologous['KM_times'])):
            self.assertAlmostEqual(survival_autologous['KM_curve'][i],lifelines_correct[i],3)

        self.assertAlmostEqual(17.0,survival_autologous['median_survival_time'])
        
    def test_AllogenicExample(self):
        allogenic_time_of_events = [1,2,3,4,6,7,12,15,20,21,24,30,60,85,85,86,87,90,100,119,132]
        allogenic_type_of_events = [1,1,1,1,1,1,1 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0  ,0  ,0 ]
        survival_data = np.column_stack([allogenic_time_of_events, allogenic_type_of_events])
        survival_allogenic = kaplan_meyer(survival_data)
        
        #Checking times of deaths
        self.assertEqual(len(survival_allogenic['KM_times']),9);
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
        self.assertEqual(len(survival_allogenic['KM_curve']),9);
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
        survival_allogenic = kaplan_meyer(survival_data)
        
        autologous_time_of_events = [1,1,1,2,2,3,4,5,6,7,8,8,10,12,12,14,17,20,27,27,28,30,30,36,38,40,45,50,50,50,63,132,132]
        autologous_type_of_events =     [1,1,1,1,1,1,1,1,1,1,1,1,1 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ,0 ,1 ,1 ,1 ,0 ,0, 0 ]
        survival_auto = np.column_stack([autologous_time_of_events,autologous_type_of_events])
        survival_autologous = kaplan_meyer(survival_auto)
        
        results = log_rank_test(survival_allogenic, survival_autologous)

        self.assertAlmostEqual(results['u_L'],6.575,2)
        self.assertAlmostEqual(results['s_2_l'],7.884,2)
        self.assertAlmostEqual(results['p_value'],0.01925,2)

        results_other_way = log_rank_test(survival_autologous, survival_allogenic)

        self.assertAlmostEqual(results_other_way['u_L'],-6.575,2)
        self.assertAlmostEqual(results_other_way['s_2_l'],7.884,2)
        self.assertAlmostEqual(results_other_way['p_value'],0.01925,2)
        
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

    def test_against_lifelines(self):
        
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

    def test_gh_readme(self):
        time = np.array([2, 3, 4, 5, 6, 7])
        event = np.array([1, 0, 1, 1, 0, 1])#1 is death, 0 is censored 
        survival_data = np.column_stack([time, event])
        km = kaplan_meyer(survival_data)
        print(km['median_survival_time'])
        

