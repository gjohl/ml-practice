# Kalman filter
Notes from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

## 1. g-h filters 
### Combining predictions and measurements
With measurements from 2 noisy sensors, we can combine them to get a much more accurate measurement. **Never throw data away.** 

If we have a prediction model for how the world evolves (e.g. predicted daily weight gain) we can combine this with noisy measurements in the same way.
Treat the prediction as a noisy sensor and combine as before, so that our estimate is some way between the prediction and the estimate.
How close it is to one or the other depends on the relative accuracy of each, and this is a hyperparameter we can set. 

```
prediction =  x_est + dx*dt 

where:
`x_est` is an initial constant that gets updated
`dx` is a predicted daily weight gain rate
`dt` is the time step
```

### g filter
We can create a single parameter filter, where the parameter `g` corresponds to how strongly we 
trust the prediction over the measurement
```
residual = prediction - measurement
estimate = measurement + g * residual
```
- `g=0 => estimate=measurement`
- `g=1 => estimate=prediction`
- for other values of `g`, the estimate will be somewhere between the measurement and prediction
 
![gh_filter.png](../_images/kalman_filter/gh_filter.png)

### g-h filter
We can go one step further and use our noisy estimates of weight to refine our the daily weight gain w used in our prediction
```
dx_t+1 = dx_t + h * residual / dt
```

This gives us the g-h filter - a generic filter which allows us to set a parameter g for the weight measurement confidence
and h for the weight change measurement confidence.

We can then update the estimates for weight and weight change on every time step 
(or every measurement if time steps are irregular). 

This insight forms the basis for Kalman filters, where we will set g and h dynamically on each time step. 

### Conclusion
Key takeaways:
- Multiple data points are more accurate than one data point, so throw nothing away no matter how inaccurate it is.
- Always choose a number part way between two data points to create a more accurate estimate.
- Predict the next measurement and rate of change based on the current estimate and how much we think it will change.
- The new estimate is then chosen as part way between the prediction and next measurement scaled by how accurate each is.
- The filter is only as good as the mathematical model used to express the system.
- Filters are designed, not selected ad hoc. No choice of g and h applies to all scenarios.

Terminology:
- System - the object we want to estimate
- State, `x` - the current configuration of the system. Hidden.
- Measurement, `z` - measured value of the state from a noisy sensor. Observable.
- State estimate - our filter's estimate of the state.
- Process model - the model we use to predict the next state based on the current state.
- System propagation - the predict step
- Measurement update - the update step
- Epoch - one iteration of system propagation and measurement update.


## 2. Discrete Bayes filter