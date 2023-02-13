import numpy as np


def entropy(pmf):
    """
    Compute entropy of a discrete random variable given it's probability mass function
    """

    pmf = pmf.flatten()

    # Handle log_2(0) error by replacing 0's with 1's. log_2(1) makes the term 0
    pmf = np.where(pmf == 0, 1, pmf)
    return -np.sum(pmf * np.log2(pmf))


def entropy_conditional(pmf_marginal, pmf_conditional):
    """
    Compute entropy of a discrete random variable given it's probability mass function
    """

    # Handle log_2(0) error by replacing 0's with 1's. log_2(1) makes the term 0
    pmf_conditional = np.where(pmf_conditional == 0, 1, pmf_conditional)
    pmf_marginal = np.where(pmf_marginal == 0, 1, pmf_marginal)

    return np.sum(pmf_marginal * -np.sum(pmf_conditional * np.log2(pmf_conditional), axis=0))


def marginal_1(pmf, i):
    """
    Marginalize a joint pmf with respect to a given random variable (by index i)
    """

    [x, y, z] = np.setdiff1d([0, 1, 2, 3], [i])

    return (pmf
            .reshape((10, 10, 10, 10))
            .sum(axis=3-x)
            .sum(axis=3-y)
            .sum(axis=3-z))


def marginal_2(pmf, i, j):
    """
    Marginalize a joint pmf with respect to two given random variables with indices i and j
    """

    [x, y] = np.setdiff1d([0, 1, 2, 3], [i, j])

    pmf = (pmf
           .reshape((10, 10, 10, 10))
           .sum(axis=3-x)
           .sum(axis=3-y))

    if i > j:
        pmf = pmf.T

    return pmf


def mutual_information(pmf, i, j):
    """
    Compute mutual information between two random variables indexed by i and j
    given a joint pmf of 4 variables
    """

    assert(i != j)

    p_i = marginal_1(pmf, i)
    H_i = entropy(p_i)

    p_j = marginal_1(pmf, j)
    p_ij = marginal_2(pmf, i, j)
    p_i_given_j = p_ij / p_j

    H_i_given_j = entropy_conditional(p_j, p_i_given_j)

    I_ij = H_i - H_i_given_j

    return I_ij