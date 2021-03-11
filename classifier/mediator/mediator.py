import numpy as np

from .fitter import Fitter
from .predictor import Predictor

from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array


class Mediator:
    """Mediator

    Class to check the input data, compute the parameters of the
    Sugeno classifier and do the classification.
    """

    def __init__(self):
        self.number_of_features = 0

    def check_train_data(self, X, y):
        """Check the input data for training.

        See 'check_X_y' in the documentation of scikit-learn for more
        information regarding the checks.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : arra-like of shape (1, n_samples)
            Target labels to X.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            Formated input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : arra-like of shape (1, n_samples)
            Formated target labels to X.

        Raises
        -------
        ValueError, if a check should fail.
        """

        X, y = check_X_y(X, y)

        if not self._check_for_regression_targets(y):
            raise ValueError('Unknown label type: ', y)

        return X, y

    def check_test_data(self, X):
        """Check the input data for classification.

        See 'check_array' in the documentation of scikit-learn for more
        information regarding the checks.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            Formated input data, where n_samples is the number of samples and
            n_features is the number of features.

        Raises
        -------
        ValueError, if a check should fail.
        """

        X = check_array(X)

        if np.shape(X)[1] != self.number_of_features:
            raise ValueError('Input data does not match number of features')

        return X

    def fit_components(self, X, y, maxitivity, margin, threshold):
        """Fit the Feature Transformation, the capacity function and
        the threshold for given input data.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target labels to X.

        maxitivity : int or None
            Maxitivity (First hyperparameter of the Sugeno classifier). If
            the maxitivity is None, the number of features will be used to
            compute the capacity function.

        margin : float
            Margin (Second hyperparameter of the Sugeno classifier).

        threshold : float or None
            Threshold. If None the linear program will be used to compute
            the threshold. Otherwise the linear program is skipped and the
            given threshold will be used for computing the capacity and
            the classification.

        Returns
        -------
        True, indicating that all parameters have been computed.
        """

        self.number_of_features = np.shape(X)[1]

        if maxitivity is not None and self.number_of_features < maxitivity:
            raise ValueError('Maxitivity is greater than n_features')

        fitter = Fitter()

        self.feature_transformation = fitter.fit_feature_transformation(X)

        normalized_X = self._get_normalized_X(X, self.feature_transformation)
        # print(normalized_X)

        if threshold is None:
            self.threshold = fitter.fit_threshold(normalized_X, y)
        else:
            self.threshold = threshold
        # print(self.threshold)

        self.capacity = fitter.fit_capacity(
            normalized_X, y, self.threshold, maxitivity, margin)
        # print(self.capacity)
        # self._print_capacity()

        return True

    def predict_classes(self, X):
        """Predict classes for input data.

        Predict classes for input data. The function fit_components
        had to be called before to use the Feature Transformation, the
        capacity function and the threshold.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        result : ndarray of shape (1, n_samples)
            Array containing the classes for the corresponding examples.
        """

        predictor = Predictor()

        normalized_X = self._get_normalized_X(X, self.feature_transformation)

        result = predictor.get_classes(
            normalized_X, self.capacity, self.threshold)

        return result.ravel()

    def _check_for_regression_targets(self, y):
        """Check target data for regression targets.

        Regression targets are considered to be real non integer numbers.
        This check is necessary to be compatible with scikit-learn.

        Parameters
        -------
        y : array-like of shape (1,)
            Target labels.

        Returns
        -------
        False, if there is a regression target. Otherwise True.
        """

        for value in y:
            # check for numeric value
            if not (isinstance(value, (int, float))
               and not isinstance(value, bool)):
                continue

            if not float(value).is_integer():
                return False

        return True

    def _get_normalized_X(self, X, f):
        """Normalize the input data using a Feature Transformation.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        f : FeatureTransformation
            Feature Transformation.

        Returns
        -------
        result : ndarray of shape (n_samples, n_features)
            Normalized input data, where n_samples is the number of samples
            and n_features the number of features.
        """

        result = list()

        for x in X:
            normalized_x = self._get_normalized_x(x, f)

            result.append(normalized_x)

        return np.array(result)

    def _get_normalized_x(self, x, f):
        """Normalize example using a Feature Transformation.

        Parameters
        -------
        x : array-like of shape (1, n_features)
            Example, where n_features is the number of features.

        f : FeatureTransformation
            Feature Transformation.

        Returns
        -------
        normalized_x : ndarray
            Normalized example.
        """

        normalized_x = list()
        number_of_features = len(x)

        for i in range(number_of_features):
            feature = x[i]
            normalized_feature = f[i](feature)

            normalized_x.append(normalized_feature)

        return np.array(normalized_x)

    def _print_capacity(self):
        for key, value in self.capacity.items():
            print(key, '->', value)
