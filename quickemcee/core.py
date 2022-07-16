"""Main file with the core scripts."""
import numpy as np
import emcee
from multiprocessing import Pool


class Model:
    """Build a model object."""

    def __init__(self, ndim, predict, priors, y_data, y_sigma):
        """
        Init an instance of Model.

        Parameters
        ----------
        ndim : int
            number of parameters to fit in the model.
        predict : callable
            predict is a callable that takes as argument a 1D array of length
            ndim and returns an array of the same shape as y_data. This object
            should compute the prediction of the model for a vector in the
            parameter space.
        priors : list
            A list of of length ndim, whose elements are callables that take a
            float and compute the prior probability of a parameter for that
            number. The list must be ordered in the same way that predict
            arguments are.
        y_data : array
            Array with the target data used to fit the model.
        y_sigma : float or array
            float with a single value for all elements in y_data or array of
            the same shape of y_data with a unique value for each element in
            y_data.

        """
        self.predict, self.ndim, self.priors = (predict, ndim, priors)
        self.y_data, self.y_sigma = (y_data, y_sigma)

    def _log_prior(self, coords):
        """
        Compute log prior.

        Parameters
        ----------
        coords : ndarray
            A one dimensional array of length ndim with the vector in the
            parameters space for which log prior is computed.

        Returns
        -------
        float

        """
        lp = 1
        for i in range(self.ndim):
            iprob = self.priors[i](coords[i])
            if iprob <= 0:
                return -np.inf
            else:
                lp = lp * iprob
        return np.log(lp)

    def _log_likelihood(self, coords):
        """
        Compute log likelihood.

        Parameters
        ----------
        coords : ndarray
            A one dimensional array of length ndim with the vector in the
            parameters space for which log likelihood is computed.

        Returns
        -------
        float

        """
        # discard diverging simulations
        prediction = self.predict(coords)
        if not np.isfinite(prediction).all():
            return -np.inf

        return -0.5 * np.sum(((prediction - self.y_data) / self.y_sigma) ** 2)

    def _log_probability(self, coords):
        """
        Compute log probability.

        Parameters
        ----------
        coords : ndarray
            A one dimensional array of length ndim with the vector in the
            parameters space for which log probability is computed.

        Returns
        -------
        float

        """
        lp = self._log_prior(coords)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self._log_likelihood(coords)

    def setup_emcee_sampler(self, nwalkers, cpu_cores=1, emcee_moves=None):
        """
        Set up the `emcee` Sampler object.

        See `emcee` docs for more details.

        Parameters
        ----------
        nwalkers : int
            number of walkers.
        cpu_cores : int, optional
            number of CPU cores to be used by the sampler. The default is 1.
        emcee_moves : TYPE, optional
            `emcee` moves object. The default is None.

        Returns
        -------
        sampler : `emcee` Ensemble Sampler object

        """
        with Pool(processes=cpu_cores) as pool:
            sampler = emcee.EnsembleSampler(nwalkers,
                                            self.ndim,
                                            self._log_probability,
                                            moves=emcee_moves,
                                            pool=pool)
        return sampler


def run_mcmc_chain(sampler, burn_iter, main_iter, init_vals=None):
    """
    Do.

    Parameters
    ----------
    sampler : `emcee` Ensemble Sampler object
        The `emcee` sampler for which the chain is run.
    burn_iter : int
        the number of steps that the chain will do during the burn in
        phase. The samples produced during burn in phase are discarded.
    main_iter : int
        the number of steps that the chain will do during the production
        phase. The samples produced during production phase are saved in
        the sampler and can be extracted for later analysis.
    init_vals : array, optional
        1D array of length ndim with initial values for the coordinates vector.
        When set as None uses all zeroes. The default is None.

    """
    ndim, nwalkers = sampler.ndim, sampler.nwalkers
    if init_vals is None:
        init_vals = np.zeros(ndim)

    p0 = [init_vals + 1e-7 * np.random.randn(ndim)
          for i in range(nwalkers)]

    print("")
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, burn_iter, progress=True)
    sampler.reset()

    print("")
    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, main_iter, progress=True)

    # function ends here
