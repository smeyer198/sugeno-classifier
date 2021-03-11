import numpy as np
import pandas as pd


class FeatureTransformation:
    """Feature Transformation

    Funtion f(x)=(f_0(x_0),...,f_{m-1}(x_{m-1})) to normalize the train
    data, where m is the number of features. Use the item and call
    operator to get the value for a specific x_i.
    Example: f[1](5) is equivalent to f_1(5) for x_i=5

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data, where n_samples is the number of samples and
        n_features is the number of features.
    """

    def __init__(self, X):
        self.function = self._initialize(X)

    def _initialize(self, X):
        """Initialize the funtion.

        Create an array of shape (1, n_features), where n_features is
        the number of features. Each index stores an instance of the
        class _FeatureTransformationComponent describing a component f_i

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns:
        ----------
        array : ndarray of shape (, n_features)
            Array containing a _FeatureTransformationCompoent f_i for
            each feature.
        """

        result = list()

        number_of_columns = np.shape(X)[1]

        for i in range(number_of_columns):
            # create a feature transformation component for eacht feature
            column = X[:, i]

            function_component = _FeatureTransformationComponent(column)

            result.append(function_component)

        return np.array(result)

    def __getitem__(self, i):
        # use the item operator to access the feature transformation
        # compontent f_i
        return self.function[i]


class _FeatureTransformationComponent:
    """Feature Transformation Component

    A feature transformation component f_i. Store all values of the i-th
    feature. Use the call operator to compute the value for a specific x_i.

    Parameters:
    ----------
    x_values : array-like of shape (n_samples, 1)
        Input data for the i-th feature, where n_samples is the number
        of samples
    """

    def __init__(self, x_values):
        self.x_values = np.sort(x_values)

        # corresponding y values according to a_j
        self.y_values = np.array(
            [1/2 * (
                self.x_values[self.x_values < value].size
                + self.x_values[self.x_values <= value].size
            ) for value in self.x_values]
        )

        # normalize y-values
        self.y_values /= self.x_values.size

        # make all values unique
        self.x_values = np.unique(self.x_values)
        self.y_values = np.unique(self.y_values)

        assert(len(self.x_values) == len(self.y_values))

    def __call__(self, x):
        # use the call operator to compute a normalized value for x
        return np.interp(x, self.x_values, self.y_values, left=0, right=1)
