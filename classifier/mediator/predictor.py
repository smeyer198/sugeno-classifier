import numpy as np

from .components.sugeno_integral import SugenoIntegral


class Predictor:
    """Predictor

    This class implements the methods for the classification.
    """

    def __init__(self):
        pass

    def get_classes(self, X, capacity, threshold):
        """Get classes for input data.

        Get classes for input data. Use the capacity for the Sugeno integral
        and the threshold for the classification.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        capacity : dict
            Dictionary describing the capacity function.

        threshold : float
            Threshold.

        Returns
        -------
        result : ndarray of shape (1, n_samples)
            Array containing the classes for the corresponding examples.
        """

        sugeno_integral = SugenoIntegral(capacity)

        result = self._get_classes_for_X(X, sugeno_integral, threshold)

        return result

    def _get_classes_for_X(self, X, integral, threshold):
        """Get classes for input data.

        Get classes for input data. Use the integral and the threshold
        for the classification.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        integral : SugenoIntegral
            Sugeno integral.

        threshold : float
            Threshold.

        Returns
        -------
        result : ndarray of shape (1, n_samples)
            Array containing the classes for the corresponding examples.
        """

        result = list()

        for x in X:
            # aggregation METHOD , method='median'
            aggregation_value = integral.compute_aggregation_value(x)

            cl = self._get_decision(aggregation_value, threshold)

            result.append(cl)

        return np.array(result)

    def _get_decision(self, aggregation_value, threshold):
        """Get class for aggregation value.

        Parameters
        -------
        aggregation_value : float
            Aggretion value.

        threshold : float
            Threshold.
        """

        if aggregation_value >= threshold:
            return np.array([1])
        else:
            return np.array([0])
