---
layout: sidepage
permalink: /line-example/
---

An interactive version of this notebook is available at [this Google Colab link](https://colab.research.google.com/drive/1iTPRObgjIBqbWDLrcotOrIIOvBIJySdi?usp=sharing).

# Usage example

In this notebook we will go through a basic example that will show `quickemcee` usage.

# Setting up the problem

Suppose we want to fit a linear model to a set of $(x,y)$ data points. Let's fabricate some data

```
# make x data
import numpy as np
x_data = np.linspace(0, 1, 101) + np.random.uniform(-2e-3, 2e-3, 101)
# make y data
def f(x, m, h):
  """Compute the linear function."""
  return m * x + h
np.random.seed(2601)
y_data = np.array([f(x, -.5, 2.0) + np.random.normal(0.0, .1) for x in x_data])
# plot the data
import matplotlib.pyplot as plt
plt.scatter(x_data, y_data, c='tab:red', s=1.5)
plt.show()
```
![index](https://user-images.githubusercontent.com/94293518/179607405-b171565c-ea4a-4a90-b58e-754ae34bd9e9.png)

# quickemcee fitting

## Prediction function

The first thing we need for our model is a "prediction" function. It takes as argument a guess for the parameters and returns the observations that those values for the parameters would have produced with our model.

In this case we have a set of points that we want to represent with a linear function, so our prediction function would be a function that takes as argument a pair of values for $m$ and $h$, and computes $y = mx  + h$ for every value of $x$ in our `x_data` variable.  

```
# define the prediction function
# it must take as argument a 1D array with length the n of params in the model ##
# as a result it must output an array with the same shape as y_data
def predict(coords):
  # unpack the model parameters
  m, h = coords
  # compute the prediction of our model for the values given by 'coords'
  prediction = m * x_data + h # notice that x_data is a numpy array, so this
                              # operation gives as result an array
  return prediction
```

## Priors

The next thing we need is a set of prior PDFs for each parameter in the model. 

In general what we do is to take a look at the raw data and try to infer from it and our previous knowledge what the parameter values should be. Some examples of prior knowledge are

1.   The value of the mass of an object is a finite positive number, therefore $0<m<M$ for some upper bound $M$.
2.   A previous experiment measured some variable and got as result $x = 5\pm 0.5$.

And then the normalized(the integral over all the posible values must return 1) priors that express this previous knowledge are

1.   $p(m) = \begin{cases} 1/M \quad & \text{if } 0<m<M \\ 0 & \text{otherwise} \end{cases}$
2.   $p(x) = \frac{1}{0.5\sqrt{2\pi}}\exp{\left[-\frac{1}{2}\left(\frac{x-5}{0.5}\right)^2\right]}$

Specifically for our model we need to define priors for $m$ and $h$. From the data points plot we see

1.   $m$ must be negative from the declining behaviour of the points, and from the range where the values oscillate, it's absolute value shouldn't be larger than 1. So we will use $p(m) = \begin{cases} 1 \quad & \text{if } -1<m<0 \\ 0 & \text{otherwise} \end{cases}$
2.   $h$ can be obtained from $y(x=0)$, so $h \approx y(x\approx 0)$. The first values of $y$ range from $1.8$ to $2.3$, so we set $p(h) = \begin{cases} 2 \quad & \text{if } 1.8<h<2.3 \\ 0 & \text{otherwise} \end{cases}$

In the code we would define this priors as callables that pick a float and return a float.

```
# define priors
def m_prior(m):
  if -1.0 < m < .0:
    return 1.0
  else:
    return .0

def h_prior(h):
  if 1.8 < h < 2.3:
    return 2.0
  else:
    return .0
```

Note: these can also be defined with lambda functions, or some script that uses local functions such as

```
# define uniform distribution "constructor"
def uniform_distribution(xmin, xmax):
  # define local function
  def f(x):
    if xmin < x < xmax:
      return 1.0 / (xmax - xmin)
    else:
      return .0
  # return local function as result
  return f

m_prior = uniform_distribution(-1.0, .0)

# or with lambda functions

m_prior = lambda m : 1 if -1.0 < m < .0 else .0
```

However, when defined in this way, our m_prior callable is not gonna be able to be pickled by Python, which will cause problems down the road, specially when trying to implemented parallelization to our code. 

To spare us code writing and debugging, `quickemcee.utils` includes some constructors to set up our PDF priors in one line without the problems mentioned earlier.

The previous code can be shortened using these methods with

```
# define priors
m_prior = qmc.utils.uniform_prior(-1.0, .0)
h_prior = qmc.utils.uniform_prior(1.8, 2.3)
```

When building the model, `quickemcee` needs the priors to be provided on a single list containing all of them. Note that the priors list must be ordered such that, each prior has the same index on the list as the index of its parameter on the predict function argument. So we pack the priors with

```
priors = [m_prior, h_prior]
```

## Building the model

Finally, we need for our model an uncertainty for the y_data values. This can be either a single value for all measurements, or an array of the same shape as y_data, with a value for each element of y_data. In this case we will assume that all values of y_data vary with $\pm0.1$, so we set

```
y_sigma = .1
```

Now we can build a `quickemcee` Model instance, which will interally define everything needed to start working with MCMC. We do it as 

```
# note: the first argument ndim is the number of parameters to fit in the model
# which in this case is 2
mymodel = qmc.core.Model(ndim=2, predict=predict, priors=priors,
                         y_data=y_data, y_sigma=y_sigma)
```

## Running chains and analyzing results

Now we can start running some MCMC chains. We will start by running a chain with 50 walkers, 50 burn in steps, and 100 production steps. 

In this case, we also should define an initial guess within the limits of the uniform prior distributions(this wouldn't be a problem with normal priors), otherwise the first values will overflow because the priors would be returning $0$, and we have to compute their logarithms. If we don't do this the script may still work, but we want to prevent overflows.

```
nwalkers, burn_iter, main_iter, init_x = 50, 50, 100, np.array([-.1, 2.0])
```

Now we run the chain with

```
sampler = mymodel.run_chain(nwalkers=nwalkers, burn_iter=burn_iter,
                            main_iter=main_iter, init_x=init_x)

```

```
>>>Running burn-in...
>>>100%|██████████| 50/50 [00:00<00:00, 330.99it/s]
>>>Running production...
>>>100%|██████████| 100/100 [00:00<00:00, 400.52it/s]
```

Then extract the results, and make a list of strings for naming the parameters.

```
samples, flat_samples = sampler.get_chain(), sampler.get_chain(flat=True)
labels = ['m', 'h']
```

To analyze the results we use the following plotting tools from `quickemcee.utils`

```
# cornerplots
qmc.utils.cornerplots(flat_samples=flat_samples, labels_list=labels)
```
