import numpy as np
import pulp as pl

from pulp import LpProblem
from pulp import LpMinimize
from pulp import LpVariable
from pulp import lpSum


class Threshold:
    """Threshold class to compute the value for the threshold

    Use the constructor to store the train data and the function
    compute_threshold to compute the threshold. The train data are
    supposed to be normalized.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data, where n_samples is the number of samples and
        n_features is the number of features. The data are supposed
        to be normalized. Otherwise the linear programm might not
        have a solution.

    y : array-like (1, n_samples)
        Target labels to X.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def compute_threshold(self):
        """Compute the threshold according to the definition of the
        linear program.

        Returns
        -------
        threshold : float
            Value for the threshold.
        """
		
        threshold = self._solve_linear_program()

        return threshold

    def _solve_linear_program(self):
        """Solve linear program defined for the threshold.

        Returns
        -------
        threshold : float
            Value for the threshold.
        """

        # for each instance in the train data a contraint is derived
        number_of_constraints = np.shape(self.X)[0]

        # --------------------- Initialize --------------------- #
        # interior point: , options=['barrier']
        solver = pl.PULP_CBC_CMD(msg=0)  # , options=['barr'])

        linprob = LpProblem(name='Threshold', sense=LpMinimize)

        beta = LpVariable(name='beta', lowBound=0, upBound=1, cat='Continuous')

        slack_items = np.arange(number_of_constraints)
        slack_variables = LpVariable.dicts(
            's', slack_items, lowBound=0, upBound=1, cat='Continuous')

        linprob += lpSum([1 * slack_variables[i]
                          for i in range(number_of_constraints)])

        # --------------------- Initialize --------------------- #

        # --------------------- Constraints --------------------- #

        for i in range(number_of_constraints):
            x = self.X[i]
            y = self.y[i]

            median = np.median(x)

            if y == 1:
                linprob += (median + slack_variables[i] >= beta)
            else:
                linprob += (median - slack_variables[i] <= beta)

        # --------------------- Constraints --------------------- #
        linprob.solve(solver)

        # search for correct variable
        for variable in linprob.variables():
            if str(variable).startswith('beta'):
                return variable.varValue

        raise ValueError('Unable to solve linear program for threshold')
