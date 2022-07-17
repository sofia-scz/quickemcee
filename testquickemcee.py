import quickemcee
import numpy as np
from time import sleep

ndata = 201

np.random.seed(335)
x_data = np.linspace(0, 1, ndata) + np.random.uniform(
                                            -.1/ndata, .1/ndata, ndata)


def f(x, m, h):
    """Compute y(x)=m*x+h."""
    return m * x + h


y_data = np.array([f(x, .5, 1) + np.random.normal(0, .1) for x in x_data])


def predict(coords):
    """Compute model prediction for a vector in params spaces."""
    m, h = coords
    sleep(0.0005)
    return np.array([f(x, m, h) for x in x_data])


priors = [quickemcee.utils.uniform_prior(0, 5),
          quickemcee.utils.uniform_prior(0, 5)
          ]

model = quickemcee.core.qmcModel(2, predict, priors, y_data, .1)

sampler = model.qmc_run_chain(nwalkers=100, burn_iter=200, main_iter=500,
                              init_vals=[.1, .1], workers=10)

samples, flat_samples = sampler.get_chain(), sampler.get_chain(flat=True)

labels = ['m', 'h']

quickemcee.utils.traceplots(samples, labels)
quickemcee.utils.cornerplots(flat_samples, labels)
quickemcee.utils.autocplots(flat_samples, labels)
quickemcee.utils.resultplot(flat_samples, y_data, x_data, predict,
                            plotmeans=False, plotsamples=100,
                            dotsize=2.0, linewidth=1.7,
                            figsize=(7,5))
