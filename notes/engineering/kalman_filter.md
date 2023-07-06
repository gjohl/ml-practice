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
### Bayesian vs Frequentist
Bayesian statistics treats probability as a belief about a single event. 
Frequentist statistics describes past events based on their frequency. It has no bearing on future events.

If I flip a coin 100 times and get 50 heads and 50 tails, frequentist statistics states the probability of heads _was_ 50%
for those cases.
On the next coin flip, frequentist statistics has nothing to say about the probability. The state is simply unknown.
Bayesian statistics incorporates these past events as a prior belief, so that we can say the next coin flip has a 50% chance
of landing heads. "Belief" is a measure of the strength of our knowledge.

When talking about the probability of something, we are implicitly saying "the probability that this event is true given past events".
This is a Bayesian approach. In practice, we may incorporate frequentist techniques too, as in the example above when the 100 previous 
coin tosses were used to inform our prior.

- Prior is the probability distribution before including the measurement's information. This corresponds to the prediction in the Kalman filter.
- Posterior is a probability distribution after incorporating the measurement's information. This corresponds to the estimated state in the Kalman filter.
- Likelihood is the joint probability of the observed data - how likely is each position given the measurement. This is not a probability distribution as it does not sum to 1.

The filter will use Bayes theorem:
```
posterior = likelihood * prior / normalization
```
In even simpler filter terms:
```
udpated knowledge = || likelihood of new knowledge * prior knowledge ||
```

If we have a prior distribution of positions and system model of the subsequent movement, 
we can convolve the two to calculate the posterior.

### Discrete Bayes filter
The discrete Bayes filter is a form of g-h filter.
It is useful for multimodal, discrete problems.

The equations are:
```
Predict step:
x_bar = x * f_x(.)
where:
- x is the current state
- f_x(.) is the state propagation function for x, i.e. the system model.


Update step:
x = ||L . x_bar||
where:
- L is the likely function
```

In pseudocode this is:
```
Initialisation:
1. Initialise our belief in the state.

Predict:
1. Predict state for the next time step using the system model.
2. Adjust belief to account for uncertainty in prediction.

Update:
1. Get a measurement and belief about its accuracy (noise estimate).
2. Compute likelihood of measurement matching each state.
3. Update posterior state belief based on likelihood.
```

Algorithms of this form a called "predictor correctors"; we make a prediction then correct it.
The predict step will always degrade our knowledge due to the uncertainty in the second step of the predict stage.
But adding another measurement, even if noisy, improves our knowledge again.
So we can converge on the most likely result.

### Evaluation of discrete Bayes filter
The algorithm is trivial to implement, debug and understand. 

Limitations:
- Scaling - Tracking i state variables results in O(n^i) runtime complexity.
- Discrete - Most real-world examples are continuous. We can increase the granularity to get a discrete approximation of 
             continuous measurements, but this increases the scale again.
- Multimodal - Sometimes you require a single output value.
- Needs a state change measurement.

The Kalman filter is based on the same idea that we can use Bayesian reasoning to combine measurements and system models.
The fundamental insight in this chapter is that we multiply (convolve) probabilities when we measure 
and shift probabilities when we update, which leads to a converging solution.


## References
- "Artificial Intelligence for Robotics". https://www.udacity.com/course/cs373
