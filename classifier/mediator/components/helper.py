import numpy as np


def get_feature_subset(x, index):
    """Get the feature subset A_{s(index)} for a given example and a
    permutation s.

    Parameters
    -------
    x : array-like of shape (1, n_features)
        Example, where n_features is the number of features.

    index : int
        Index for the feature subset.

    Returns
    -------
    result : frozenset
        Feature subset.
    """

    result = set()

    m = len(x)
    sorted_x = np.sort(x)

    permutation = _get_permutation_for_x(x, sorted_x)

    # compute feature subset {\sigma(i),...,\sigma(m)}
    for i in range(index, m + 1):
        result.add(permutation[i])

    return frozenset(result)


def _get_permutation_for_x(x, sorted_x):
    """Get permutation for an example.

    Get a dictionary describing a permutation s of x with
    x_s(1) <= ... <= x_s(m). The dictionary maps the values
    1,...,m to the corresponding position of the permutation,
    where m is the number of features.

    Parameters
    -------
    x : array-like of shape (1, n_features)
        Example, where n_features is the number of features.

    sorted_x : array-like of shape (1, n_features)
        Sorted example, where n_features is the number of features.

    Returns
    -------
    permutation : dict
        Dictionary describing the permutation.
    """

    permutation = dict()

    x_copy = x.copy()

    for i in range(len(x)):
        value = sorted_x[i]

        # TODO which index to choose if there is a tie between values
        index = np.where(x_copy == value)[0][0]

        permutation[i + 1] = index + 1
        x_copy[index] = -9999  # TODO choose dummy value

    return permutation
