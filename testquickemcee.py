import quickemcee
import numpy as np

np.random.seed(335)
x_data = np.linspace(0, 1, 51) + np.random.uniform(-.01, .01, 51)


def f(x, m, h):
    """Compute y(x)=m*x+h."""
    return m * x + h


y_data = np.array([f(x, .5, 1) + np.random.normal(0, .05) for x in x_data])


def predict(coords):
    """Compute model prediction for a vector in params spaces."""
    m, h = coords
    return np.array([f(x, m, h) for x in x_data])


priors = [quickemcee.utils.uniform_prior(0, 5),
          quickemcee.utils.uniform_prior(0, 5)
          ]

model = quickemcee.core.qmcModel(2, predict, priors, y_data, .1)

sampler = model.setup_emcee_sampler(nwalkers=50)

quickemcee.core.run_mcmc_chain(sampler, 300, 1000)

samples, flat_samples = sampler.get_chain(), sampler.get_chain(flat=True)

labels = ['m', 'h']

quickemcee.utils.traceplots(samples, labels)
quickemcee.utils.cornerplots(flat_samples, labels)
quickemcee.utils.autocplots(flat_samples, labels)
