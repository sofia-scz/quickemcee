---
layout: sidepage
permalink: /example-usage/
---

An interactive version of this notebook is available at [this Google Colab link](https://colab.research.google.com/drive/1iTPRObgjIBqbWDLrcotOrIIOvBIJySdi?usp=sharing).

# Usage example

In this notebook we will go through a basic example that will show `quickemcee` usage.

## Setting up the problem

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

## quickemcee fitting

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
