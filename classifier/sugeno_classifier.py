import numpy as np

from .mediator.mediator import Mediator

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array


class SugenoClassifier(BaseEstimator, ClassifierMixin):
    """The Sugeno classifier.

    Implementation of the Sugeno classifier, which were invented in
    the paper "Machine Learning with the Sugeno Integral: The Case of
    Binary Classification". The classifier is compatible to scikit-learn,
    i.e. it can be used together with other algorithms from this library.

    Parameters
    -------
    maxitivity : int, default=None
        The maxitivity of the capacity function. Setting this parameter
        might increase the performance. The value is expected to be
        between 1 and the number of features in a dataset. The default
        value will set the maxitivity to the number of features in a
        given dataset.

    margin : float, default=0
        The margin which influences the values of the capacity function.
        Tests have shown that a better performance can be achieved by
        choosing a margin greater than 0. Reasonable values can be found
        in the intervals [0, 0.1] or [0, 0.2].

    threshold : float, default=None
        The threshold which is used to compute the capacity and classify
        data. Setting this parameter skips the computation of the liear
        program and uses the value instead. This parameter was used in
        an evaluation part of a Bachelor thesis and should not be changed.
    """

    def __init__(self, maxitivity=None, margin=0, threshold=None):
        self.maxitivity = maxitivity
        self.margin = margin
        self.threshold = threshold

    def fit(self, X, y):
        """Initialize the parameters of the Sugeno classifier.

        Initialize the Feature Transformation, the capacity and the
        threshold with the initializes hyperparameter for a given dataset.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features. The numger of features
            have to less or equal to the maxitivity.

        y : array-like of shape (n_samples,)
            Target labels to X.

        Returns
        -------
        self : SugenoClassifier
            Fitted estimator.
        """

        self.mediator_ = Mediator()

        X, y = self.mediator_.check_train_data(X, y)

        # replace class labels with the values 0 and 1
        self.classes_, y = np.unique(y, return_inverse=True)

        self.mediator_.fit_components(
            X, y, self.maxitivity, self.margin, self.threshold)

        # attribute which is used to be compatible with scikit-learn
        self.n_features_in_ = self.mediator_.number_of_features

        return self

    def predict(self, X):
        """Predict class for X.

        Predict the class labels for all samples in X, which were
        specified in fit. The number of features have to match the
        the number of features from the train data.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """

        check_is_fitted(self)

        X = self.mediator_.check_test_data(X)

        result = self.mediator_.predict_classes(X)

        return self.classes_[result]

    # ===============================================================
    # Functions for compatibility with scikit-learn. They are not
    # supposed to be used.
    # ===============================================================

    def get_params(self, deep=True):
        return {'maxitivity': self.maxitivity,
                'margin': self.margin,
                'threshold': self.threshold}

    def _more_tags(self):
        return {'binary_only': True, 'poor_score': True}

    def _get_threshold(self):
        return self.mediator_.threshold
