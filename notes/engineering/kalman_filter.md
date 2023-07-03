# Kalman filter
Notes from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

## 1. g-h filters 
### Combining predictions and measurements
With measurements from 2 noisy sensors, we can combine them to get a much more accurate measurement. **Never throw data away.** 

If we have a prediction model for how the world evolves (e.g. predicted daily weight gain) we can combine this with noisy measurements in the same way.
Treat the prediction as a noisy sensor and combine as before, so that our estimate is some way between the prediction and the estimate.
How close it is to one or the other depends on the relative accuracy of each, and this is a hyperparameter we can set. 

```
prediction =  c + r*t 

where:
`c` is an initial constant
`r` is a predicted daily weight gain rate
`t` is the time step
```

### g filter
We can create a single parameter filter, where the parameter `g` corresponds to how strongly we 
trust the prediction over the measurement
```
estimate = measurement + g(prediction - measurement)
```
- `g=0 => estimate=measurement`
- `g=1 => estimate=prediction`
- for other values of `g`, the estimate will be somewhere between the measurement and prediction
 
![gh_filter.png](../_images/kalman_filter/gh_filter.png)

### g-h filter
We can go one step further and use our noisy estimates of weight to refine our the daily weight gain w used in our prediction
```
r_t*1 = r_t + h*(measurement - prediction)/ delta_t
```

This gives us the g-h filter - a generic filter which allows us to set a parameter g for the weight measurement confidence and h for the weight change measurement confidence.

We can then update the estimates for weight and weight change on every time step (or every measurement if time steps are irregular). 

This insight forms the basis for Kalman filters, where we will set g and h dynamically on each time step. 