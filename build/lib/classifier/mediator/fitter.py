import numpy as np

from .components.feature_transformation import FeatureTransformation
from .components.capacity import Capacity
from .components.threshold import Threshold


class Fitter:
    """Fitter

    This class implements the funtions to compute the parameters of the
    Sugeno classifier.
    """

    def __init__(self):
        pass

    def fit_feature_transformation(self, X):
        """Fit the Feature Transformation for given input data.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        feature_transformation : FeatureTransformation
            Instance of the fitted Feature Transformation.
        """

        feature_transformation = FeatureTransformation(X)

        return feature_transformation

    def fit_capacity(self, X, y, threshold, maxitivity, margin):
        """Fit the capacity function for given input data.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : arra-like of shape (1, n_samples)
            Target labels to X.

        threshold : float
            Threshold.

        maxitivity : int or None
            Maxitivity (First hyperparameter of the Sugeno classifier). If
            the maxitivity is None, the number of features will be used to
            compute the capacity function.

        margin : float
            Margin (Second hyperparameter of the Sugeno classifier).

        Returns
        -------
        function : dict
            Dictionary describing the capacity function.
        """

        number_of_features = np.shape(X)[1]

        if maxitivity is None:
            maxitivity = number_of_features

        capacity = Capacity(X, y)

        function = capacity.compute_capacity(threshold, maxitivity, margin)

        return function

    def fit_threshold(self, X, y):
        """Fit the threshold for given input data.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : arra-like of shape (1, n_samples)
            Target labels to X.

        Returns
        -------
        threshold : float
            Threshold.
        """

        threshold = Threshold(X, y)

        result = threshold.compute_threshold()

        return result
