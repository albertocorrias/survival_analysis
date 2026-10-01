# Survival Analysis functions

This repository contains some Python functionalities 
intended for analysis of survival data (also known as time-to-event data).

There are 3 core functionalities

  * Calculating Kaplan Meier survival curves
  * Comparing two Kaplan Meier suvival curves through the log-rank test
  * Computing hazard ratios (and their confidence intervals) through the Cox proportional hazards model

# Important note

This code was developed for educational purposes. The code favours clarity over efficiency, elegance or brevity.
Some comments in the code refers to lecture notes distributed as part of a course. 
For professional-grade survival software packages, [lifelines](https://github.com/camdavidsonpilon/lifelines) is recommended.


# Sample usage

```python
import numpy as np
from SurvivalFunctions import kaplan_meier, log_rank_test, cox_prop_haz

time = np.array([2, 3, 4, 5, 6, 7])
event = np.array([1, 0, 1, 1, 0, 1])#1 is death, 0 is censored 
survival_data = np.column_stack([time, event])
km = kaplan_meier(survival_data)
print(km['median_survival_time'])
```
