import numpy as np

from . import helper as h


class SugenoIntegral:
    """Sugeno integral

    Class to store a capacity function and compute the aggregation value.

    Parameters
    -------
    capacity : dict
            Dictionary describing the capacity function.
    """

    def __init__(self, capacity):
        self.capacity = capacity

    def compute_aggregation_value(self, x, method='definition'):
        """Compute the aggregation value for a given example.

        Parameters
        -------
        x : array-like of shape (1, n_features)
                Example, where n_features is the number of features.

        method : str
                Specify the method used for the Sugeno integral. Default is
                the method according to the definition. Use 'median' to use
                the median representation of the Sugeno integral.

        Returns
        -------
        integral_value : float
                Integral value computed by the specified method.

        """

        if method == 'definition':
            integral_value = self._get_integral_value(x, self.capacity)

            return integral_value

        if method == 'median':
            integral_value = self._get_median_value(x, self.capacity)

            return integral_value

    def _get_integral_value(self, x, capacity):
        """Get integral value for a given example according the definition
        of the Sugeno integral.

        Parameters
        -------
        x : array-like of (1, n_features)
                Example, where n_features is the number of features.

        capacity : dict
                Dictionary describing the capacity function.

        Returns
        -------
        result : float
                Integral value.
        """

        result = 0
        number_of_features = len(x)
        sorted_x = np.sort(x)

        for j in range(number_of_features):
            x_value = sorted_x[j]

            feature_subset = h.get_feature_subset(x, j + 1)
            subset_value = capacity[feature_subset]

            minimum_for_index = min(x_value, subset_value)

            # update maximum
            if minimum_for_index > result:
                result = minimum_for_index

        return result

    def _get_median_value(self, x, capacity):
        """Get integral value for a given example according the median
        representation of the Sugeno integral.

        Parameters
        -------
        x : array-like of (1, n_features)
                Example, where n_features is the number of features.

        capacity : dict
                Dictionary describing the capacity function.

        Returns
        -------
        result : float
                Integral value.
        """

        sorted_x = np.sort(x)

        median_representation = list(x)

        for i in range(2, len(x) + 1):
            subset_for_index = h.get_feature_subset(x, i)

            subset_value = capacity[subset_for_index]

            median_representation.append(subset_value)

        integral_value = np.median(median_representation)

        return integral_value
