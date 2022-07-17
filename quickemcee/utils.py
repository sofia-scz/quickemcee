"""Submodule with utilities/auxiliary scripts."""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import corner
from emcee.autocorr import function_1d
from scipy.stats import gaussian_kde


#                           Aux functions

def mode(data, xmin=None, xmax=None, bw=None, xr_fr=.1):
    """
    Compute the mode of a 1D array of numerical values.

    Compute the mode of an array of values using a kernel density estimation
    method. The KDE method used is `scipy.stats.gaussian_kde`.

    In multimodal distributions it should return the mode of the highest peak
    within the cutoff values.

    Parameters
    ----------
    data : array
        Array containing the values.
    xmin : float, optional
        Lower bound for cutting off the values. The default is None (uses the
        minimum of all values).
    xmax : float, optional
        Upper bound for cutting off the values. The default is None (uses the
        maximum of all values).
    bw : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth when calling
        `scipy.stats.gaussian_kde`. The default is None (uses the default
        method). See `scipy` docs for more info.
    xr_fr : float, optional
        Parameter used to define the KDE grid. The default is .1.

    Returns
    -------
    float
        Value of the mode.
    """
    if xmin is None:
        xmin = np.min(data)
    if xmax is None:
        xmax = np.max(data)
    # Define KDE limits.
    x_rang = xr_fr * (xmax - xmin)
    kde_x = np.mgrid[xmin - x_rang:xmax + x_rang:1000j]
    try:
        kernel_cl = gaussian_kde(data, bw_method=bw)
        kde = np.reshape(kernel_cl(kde_x).T, kde_x.shape)
    except np.linalg.LinAlgError:
        kde = np.array([])

    return kde_x[np.argmax(kde)]


#                                   PRIOR CLASSES


class uniform_prior:
    r"""
    Define uniform PDF callable.

    Defines a probability distribution object. An instance of this class can be
    called to compute the probability density of a float, and it only takes the
    float as argument.

    The uniform prior distribution is defined as

    $$p(x) = \frac{1}{m_{max}-m_{min}} \quad\text{if}\quad m_{min}<x<m_{max} \\
        0 \quad\text{otherwise}$$

    and is given normalized.
    """

    def __init__(self, xmin, xmax):
        """
        Initialice an instance.

        Parameters
        ----------
        xmin : float
            lower bound of the uniform distribution.
        xmax : float
            upper bound of the uniform distribution.
        """
        self.xmin = xmin
        self.xmax = xmax

    def __call__(self, x):
        """
        Compute the probability of a real valued variable.

        Parameters
        ----------
        x : float

        Returns
        -------
        float
            p(x) as per the mathematical definition of the PDF.
        """
        p = 1 if x < self.xmax and x > self.xmin else 0
        return p / (self.xmax - self.xmin)


class normal_prior:
    r"""
    Define normal PDF callable.

    Define a probability distribution object. An instance of this class can be
    called to compute the probability density of a float, and it only takes the
    float as argument.

    The normal (aka Gaussian) prior distribution is defined as

    $$p(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp{
        \left\{ -\frac{1}{2} \left( \frac{x-x_0}{\sigma} \right)^2
            \right\}
        }$$

    and is given normalized.
    """

    def __init__(self, x0, sigma):
        """
        Initialice an instance.

        Parameters
        ----------
        x0 : float
            central value of the normal distribution.
        sigma : float
            standard deviation of the normal distribution.
        """
        self.x0 = x0
        self.sigma = sigma

    def __call__(self, x):
        """
        Compute the probability of a real valued variable.

        Parameters
        ----------
        x : float

        Returns
        -------
        float
            p(x) as per the mathematical definition of the PDF.
        """
        p = (np.exp(-0.5 * ((x - self.x0) / self.sigma) ** 2)) / sqrt(
            2 * np.pi) / self.sigma
        return p


#                                   PLOTTING


def cornerplots(flat_samples, labels_list):
    """
    Make cornerplots given a sample and a list of labels for each parameter.

    The middle red line on each PDF marks the median, which is also the central
    value reported above the plot. The dashed grey lines mark the 16/84 and
    84/16 quantiles, which indicate the SD in a normal distribution.

    Parameters
    ----------
    flat_samples : array
        flattened array of samples.
    labels_list : list
        list with a label(str) for each parameter of the samples.
    """
    dim = len(labels_list)
    sample_medians = [np.median(flat_samples[:, _]) for _ in range(dim)]

    corner.corner(
        flat_samples,
        labels=labels_list,
        quantiles=(0.16, 0.84),
        show_titles=True,
        title_fmt=".3g",
        truths=sample_medians,
        truth_color="tab:red",
    )
    plt.show()


def traceplots(samples, labels_list, figsize=None, dpi=100):
    """
    Make traceplots for each parameter of a given sample.

    A trace plot shows the evolution of each walker for a parameter during an
    MCMC run. This is used for analyzing the convergence of an MCMC chain, or
    to diagnose problems in a not converging chain.

    Notice that for this plot the samples array is not given flattened.

    Parameters
    ----------
    samples : array
        non flattened array of samples.
    labels_list : list
        list with a label(str) for each parameter of the samples.
    figsize : touple, optional
        A touple of integers with the size of the figure. The default is
        None(uses pyplot default).
    dpi : int, optional
        Figure dots per inch. The default is 100.
    """
    dim = len(labels_list)
    fig, axes = plt.subplots(dim, figsize=figsize, dpi=dpi, sharex=True)
    plt.suptitle("parameter traces")
    for i in range(dim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels_list[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.show()


def autocplots(flat_samples, labels_list, figsize=None, dpi=100):
    """
    Plot autocorrelation function for each parameter.

    Plot the autocorrelation function for each parameter of a given sample. The
    function is computed with `emcee.autocorr.function_1d`. It is used to asses
    the convergence of an MCMC chain.

    An autocorrelation function that quickly drops from 1 to 0, and keeps
    oscillating around 0 afterwards suggests that the samples might come from a
    converged chain. It is not a final answer and running other tests at the
    same time is advised.

    Parameters
    ----------
    flat_samples : array
        flattened array of samples.
    labels_list : list
        list with a label(str) for each parameter of the samples.
    figsize : touple, optional
        A touple of integers with the size of the figure. The default is
        None(uses pyplot default).
    dpi : int, optional
        Figure dots per inch. The default is 100.
    """
    dim, clen = len(labels_list), len(flat_samples)
    step_slice = clen // 100
    aux_dom = range(0, clen, step_slice)
    aux_fl = flat_samples[::step_slice]
    autocfs = np.array([function_1d(aux_fl[:, _]) for _ in range(dim)])
    fig, axes = plt.subplots(dim, figsize=figsize, dpi=dpi, sharex=True)
    plt.suptitle("autocorrelation functions")
    for i in range(dim):
        ax = axes[i]
        ax.stem(aux_dom, autocfs[i, :], markerfmt="")
        ax.set_ylabel(labels_list[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("sample number")
    plt.show()


def resultplot(flat_samples, y_data, x_data, predf,
               plotmeans=True, plotmodes=False, plotsamples=0,
               figsize=None, dpi=100, dotsize=None, linewidth=None):
    """
    Plot different simulated results and the data.

    Makes a scatter plot for the data, and makes extra plots on top of it to
    compare the results with the data.

    It's possible to plot simulations of the results with the means and the
    modes of the results, and to make shaded plots of simulations using samples
    from the chain, showing the most likely results as more shaded areas.

    Parameters
    ----------
    flat_samples : array
        flattened array of samples.
    y_data : array
        Array with the y target data used to fit the model.
    x_data : array
        Array with the x values corresponding to the y_data values.
    predf : callable
        The prediction function used in sampling. Should take a 1D array of
        length ndim and return an array of the same shape as y_data.
    plotmeans : boolean, optional
        Plots the predicted results using the mean of the samples. The default
        is True.
    plotmodes : boolean, optional
        Plots the predicted results using the mode(most likely value) of the
        samples. The default is True.
    plotsamples : int, optional
        Number of samples to draw for making the shaded plots. The default is
        0.
    figsize : touple, optional
        A touple of integers with the size of the figure. The default is
        None(uses pyplot default).
    dpi : int, optional
        Figure dots per inch. The default is 100.
    dotsize : float, optional
        Marker size for scatter plots. The default is None(uses pyplot
        default).
    linewidth : float, optional
        Line width for regular plots. The default is None(uses pyplot default).
    """
    ndim, ndata = len(flat_samples[0]), len(y_data)
    x_aux = np.linspace(x_data.min(), x_data.max(), ndata)
    plt.figure(figsize=figsize, dpi=dpi)

    if plotsamples:
        permutation = np.random.permutation(plotsamples)
        auxsamples = [flat_samples[i] for i in permutation]
        for sample in auxsamples:
            plt.plot(x_aux, predf(sample),
                     lw=0.7*linewidth, c='black', alpha=0.1)

    plt.scatter(x_data, y_data, s=dotsize, label='data')

    if plotmeans:
        means = np.array([np.mean(flat_samples[:, i]) for i in range(ndim)])
        plt.plot(x_aux, predf(means), lw=linewidth, c='tab:red', label='mean')

    if plotmodes:
        modes = np.array([mode(flat_samples[:, i]) for i in range(ndim)])
        plt.plot(x_aux, predf(modes), lw=linewidth, c='tab:purple',
                 label='mode')
    plt.legend()
    plt.show()
