import numpy as np
import pulp as pl
import itertools

from . import helper as h

from collections import defaultdict
from pulp import LpProblem
from pulp import LpMinimize
from pulp import LpVariable
from pulp import lpSum


class Capacity:
    """Capacity function for the sugeno integral

    Normalized Capacity funtion for the sugeno integral. Use the function
    compute_capacity to compute the capacity. The train data are supposed
    to be normalized.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data, where n_samples is the number of samples and
        n_features is the number of features. The data are supposed
        to be normalized. Otherwise thelinear programm might not have
        a solution.

    y : array-like (1, n_samples)
        Target labels to X.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.number_of_features = np.shape(self.X)[1]

        # store the values 1...m, i.e. [m]
        self.base_set = frozenset(
            [i for i in range(1, self.number_of_features + 1)])

        # data structures, remove empty set because it is set to 0
        self.powerset = self._get_powerset(self.base_set)[1:]
        self.subset_ids = self._get_subset_ids(self.powerset)
        self.subsets_of_size = self._get_subsets_of_size(self.powerset)

    def compute_capacity(self, threshold, maxitivity, margin):
        """Get the capacity function for the saved input data.

        Parameters
        ----------
        threshold : float
            Threshold.

        maxitivity : int
            Maxitivity (First hyperparameter of the sugeno classifier).

        margin : float
            Margin (Second hyperparameter of the sugeno classifier).

        Returns
        ----------
        function : dict
            Capacity function: the keys are the subsets of [m], the
            values are the computed values.
        """

        linprob, number_of_c_variables = self._initialize_linear_program(
            threshold, maxitivity, margin)

        result = self._solve_linear_program(linprob, number_of_c_variables)

        function = self._get_capacity_from_result(result, maxitivity)

        function = self._update_capacity_for_maxitivity(maxitivity, function)

        return function

    def _get_powerset(self, s):
        """Get the powerset of s.

        Parameters
        ----------
        s : array-like of shape (1, n_samples)
            Input data, where n_samples is the number of elements in the
            set.

        Returns
        ----------
        powerset : ndarray of shape (1, n_samples)
            Array containing all subsets of given set.
        """

        result = list()
        items = list(s)

        # copied from
        # https://docs.python.org/3/library/itertools.html#itertools-recipes
        powerset = itertools.chain.from_iterable(
            itertools.combinations(items, r) for r in range(len(items) + 1))

        # parse all elements to frozensets
        for item in powerset:
            result.append(frozenset(item))

        return np.array(result)

    # map: subset -> id
    def _get_subset_ids(self, s):
        """Get a map: subset -> id

        Compute a dictionary describing a map. The keys are subsets of s,
        which are mapped to an unique id ranging from 0 to |s|-1

        Parameter
        ----------
        s : array-like of shape (1,)
            Sets, which are supposed to be mapped.

        Returns
        ----------
        map : dict
            Dictionary, which maps the given subsets to an unique id
        """

        result = dict()
        counter = 0

        for subset in s:
            result[subset] = counter
            counter += 1

        return result

    # map: n -> list with subsets of size n
    def _get_subsets_of_size(self, s):
        """Get a map: n -> list with subsets of size n

        Compute a dictionaty describing a map. The keys are numerical
        values ranging from the smallest element to the biggest element
        in s. These are mapped to a list containing all elements of the
        same size as the key.

        Parameter
        ----------
        s : array-like of shape (1, n_sets)
            Input. Each element is a set of type frozenset.

        Returns
        ----------
        map : dict
            Dictionary, which maps the numerical values to a list
            containing all sets with the corresponding value size.
        """

        result = defaultdict(list)

        for subset in s:
            result[len(subset)].append(subset)

        return result

    def _get_capacity_from_result(self, result, maxitivity):
        """Get a dictionary containing all computed values for given variables

        Compute first part of the capacity funtion. This method sets all
        values for the sets, which corresponding variables were part of
        the linear program.

        Parameter
        ----------
        result : list
            List with values for the subsets. The position in the list
            has to match the index of the corresponding set.

        maxitivity : int
            The maxitivity used by solving the linear program.

        Returns
        ----------
        function : dict
            Dictionary containing the sets as keys and their
            corresponding computed values as values.

        """

        function = dict()

        subsets_of_size_k = self.subsets_of_size[maxitivity]

        for s in self.powerset:
            if len(s) <= maxitivity:
                subset_id = self.subset_ids[s]

                # set values from variables of LP
                result_value = result[subset_id]

                function[s] = result_value

        # base cases
        function[frozenset()] = 0.0
        function[self.base_set] = 1.0

        return function

    def _update_capacity_for_maxitivity(self, maxitivity, capacity):
        """Update a given capacity function for maxitivity k.

        If the maxitivity k is smaller than the number of features,
        the values for sets of size s with s > k are not part of the
        linear program. Hence they have to be added manually.

        Parameters
        ----------
        maxitivity : int
            Maxitivity of the capacity function.

        capacity : dict
            Dictionary describing the capacity funtion. All values for
            sets smaller or equal to the maxitivity have to be set.

        Returns
        ----------
        capacity : dict
            Updated dictionary containing all subsets of the base set.
        """

        subsets_of_size_k = self.subsets_of_size[maxitivity]

        # set values for all sets with size greater than the maxitivity
        for s in range(maxitivity + 1, self.number_of_features):
            subsets_of_size_s = self.subsets_of_size[s]

            for subset in subsets_of_size_s:
                # get all subsets of size k for the iterated set
                k_subsets = self._get_k_subsets_for_set(maxitivity, subset)

                # get maximum value computed for the capacity
                maximum = self._get_maximum_of_subsets(k_subsets, capacity)

                capacity[subset] = maximum

        return capacity

    def _get_k_subsets_for_set(self, k, subset):
        """Get all subsets of size k for a given set.

        Parameters
        ----------
        k : int
            Size of the sets to be determined.

        subset : frozenset
            Set whose subsets of size k to be determined.

        Returns
        ----------
        result : ndarray
            Array containing all subsets of size k for subset.
        """

        result = list()

        k_subsets = self.subsets_of_size[k]

        # find all subsets of size k
        for s in k_subsets:
            if s.issubset(subset):
                result.append(s)

        return np.array(result)

    def _get_maximum_of_subsets(self, subsets, function):
        """Find the maximum value for the subsets in the given function.

        Parameters
        ----------
        subsets : array-like of shape (1,)
            Subsets whose maximum to be determined.

        function : dict
            Dictionary mapping the susbets to a value.

        Returns
        ----------
        result : float
            Maximum of function values for the given subsets.
        """

        result = 0

        # find maximum of subsets
        for s in subsets:
            if function[s] > result:
                result = function[s]

        return result

    def _solve_linear_program(self, linprob, number_of_c_variables):
        """Solve the linear program.

        Parameters
        ----------
        linprob : LpProblem
            Instance of a linear program to be solved.

        number_of_c_variables : int
            Number of variables for the function values. This is used because
            not all c_variables might be part of the linear program.

        Returns
        ----------
        solution : ndarray
            Array containing the solution of the x-variables from the
            linear program. The positions in the array match the ids
            from subset_ids.
        """

        # interior point: , options=['barr']
        solver = pl.PULP_CBC_CMD(msg=0)

        linprob.solve(solver)

        solution = np.zeros(number_of_c_variables)

        # find x-variables and set their values
        for variable in linprob.variables():
            if str(variable).startswith('x'):
                index = str(variable)[len('x') + 1:]

                solution[int(index)] = variable.varValue

        return solution

    def _initialize_linear_program(self, threshold, maxitivity, margin):
        """Initialize the linear program.

        Intialize the linear program, i.e. the variables, the objective
        function and the contraints. The variables are named x_0,...,x_s
        depending on the number of features and the maxitivity.

        Parameters
        ----------
        threshold : float
            Threshold.

        maxitivity : int
            Maxitivity (First hyperparameter of the Sugeno classifier).

        margin : float
            Margin (Second hyperparameter of the Sugeno classifier).

        Returns
        ----------
        linprob : LpProblem
            Instance of the linear problem.

        number_of_c_variables : int
            Number of variables to compute for the capacity function values.
        """

        number_of_c_variables = self._get_number_of_values_to_compute(
            1, maxitivity)
        number_of_instances = np.shape(self.X)[0]

        # --------------------- Initialize --------------------- #

        linprob = LpProblem('Capacity', sense=LpMinimize)

        c_items = np.arange(number_of_c_variables)
        c_variables = LpVariable.dicts(
            'x', c_items, lowBound=0, upBound=1, cat='Continuous')

        # --------------------- Initialize --------------------- #

        # --------------------- monotony constraints --------------------- #

        monotony_constraints = self._get_monotony_constraints(1, maxitivity)

        for A, B in monotony_constraints:
            A_id = self.subset_ids[A]
            B_id = self.subset_ids[B]

            linprob += (c_variables[A_id] <= c_variables[B_id])

        # --------------------- monotony constraints --------------------- #

        # --------------------- median constraints --------------------- #

        slack_item = 0
        slack_variables = dict()

        for i in range(number_of_instances):
            x = self.X[i]
            y = self.y[i]

            # get all sets which have to be constrained
            feature_subsets = self._get_feature_subsets_for_x_y(
                x, y, threshold, maxitivity, margin)

            for feature_subset in feature_subsets:
                assert len(feature_subset) <= maxitivity

                # create new slack variable
                s_variable = LpVariable(
                    's_' + str(slack_item),
                    lowBound=0,
                    upBound=1,
                    cat='Continuous'
                )
                slack_variables[slack_item] = s_variable

                feature_subset_id = self.subset_ids[feature_subset]

                if y == 1:
                    linprob += (c_variables[feature_subset_id]
                                + slack_variables[slack_item]
                                >= threshold + margin)
                else:
                    linprob += (c_variables[feature_subset_id]
                                - slack_variables[slack_item]
                                <= threshold - margin)

                slack_item += 1

        number_of_s_variables = slack_item

        # --------------------- median constraints --------------------- #

        # objective function
        linprob += lpSum([1 * slack_variables[i]
                          for i in range(number_of_s_variables)])

        # return linprob
        return linprob, number_of_c_variables

    def _get_number_of_values_to_compute(self, lower, upper):
        """Get number of values to compute.

        Get the number of variables which correspond to the C variables.
        This number is equivalent to the sum of the binomial coefficients
        <m over lower> to <m over upper>, where m is the nubmer of features.

        Parameter
        ----------
        lower : int
             Lower bound.

        upper : int
            Upper bound.

        Returns
        ----------
        result : int
            Number of needed C variables.

        """

        result = 0

        for i in range(lower, upper + 1):
            subsets_of_size_i = self.subsets_of_size[i]

            result += len(subsets_of_size_i)

        return result

    # create monotony constraints (equation (7))
    def _get_monotony_constraints(self, lower, upper):
        """Get monotony contraints for the linear program.

        Get all contraints for f(A) <= f(B) for the capacity funtion f
        two sets A and B with |B|=|A| + 1 and A is a subet of B.

        Parameters
        ----------
        lower : int
            Lower bound describing the minimum size of sets.

        upper : int
            Upper bound describung the maximum size of sets

        Returns
        ----------
        constraints : ndarray of shape (1,)
            Array containing tuples (A,B) corresponding to sets A and B
            according to the description above.

        """

        monotony_constraints = list()

        for s in range(lower, upper):
            # get contraints for f(A) <= f(B) with |A| = s
            constraints = self._get_constraints_for_subsets_of_size(s)

            for item in constraints:
                monotony_constraints.append(item)

        return np.array(monotony_constraints)

    # contraints: subset A of size i <= subset B of size i+1
    def _get_constraints_for_subsets_of_size(self, size):
        """Get constraints for all subsets of size.

        Get all contraints f(A) <= f(B) with |A| = size and
        |B| = size + 1.

        Parameters
        ----------
        size : int
            Size of subsets.

        Returns
        ----------
        contraints : ndarray of shape (1,)
            Array containing tuples (A,B) corresponding to sets A and B
            according to the description above.
        """

        constraints = list()

        # check pairwise subset relation of sets of size and sets of size + 1
        for subset in self.subsets_of_size[size]:
            for other_subset in self.subsets_of_size[size + 1]:

                if subset.issubset(other_subset):
                    constraints.append((subset, other_subset))

        return np.array(constraints)

    def _get_feature_subsets_for_x_y(self, x, y, threshold, maxitivity,
                                     margin):
        """Get sets which have to be constraint for a given example.

        Parameters
        ----------
        x : array-like of shape (1, n_features)
            Example, where n_feature is the number of features.

        y : int
            Target label to x.

        threshold : float
            Treshold.

        maxitivity : int
            Maxitivity (First hyperparameter of the Sugeno classifier).

        margin : float
            Margin (Second hyperparameter of the Sugeno classifier).

        Returns
        ----------
        result : ndarray of shape (1,)
            Array containing all sets, which have to be constraint.
        """

        result = list()

        sorted_x = np.sort(x)

        if y == 1:
            p = self._get_p_for_positive_x(
                sorted_x, threshold, maxitivity, margin)
        else:
            p = self._get_p_for_negative_x(
                sorted_x, threshold, maxitivity, margin)

        # no contraint has to be added
        if p is None:
            return np.array([])

        if y == 1:
            if p > maxitivity:
                # to be consistent always choose c_A with size k
                feature_subset = h.get_feature_subset(
                    x, self.number_of_features - maxitivity + 1)
            else:
                feature_subset = h.get_feature_subset(
                    x, self.number_of_features - p + 1)

            result.append(feature_subset)
        else:
            if p < self.number_of_features - maxitivity:
                # add all subsets of B with size k
                B = h.get_feature_subset(x, p + 1)

                feature_subsets = self._get_feature_subsets_of_size(
                    maxitivity, B)

                for feature_subset in feature_subsets:
                    result.append(feature_subset)
            else:
                feature_subset = h.get_feature_subset(x, p + 1)

                result.append(feature_subset)

        return np.array(result)

    def _get_p_for_positive_x(self, x, threshold, maxitivity, margin):
        """Get the p value for a positive example (y = 1).

        Parameters
        ----------
        x : array-like of shape (1, n_features)
            Example, where n_features is the number of features.

        threshold : float
            Threshold.

        maxitivity : int
            Maxitivity (First hyperparameter of the Sugeno classifier).

        Margin : float
            Margin (Second hyperparameter of the Sugeno classifier).

        Returns
        ----------
        p : int or None
            p value describing the set which has to be contraint or None
            if no constraint can be derived.
        """

        # number of elements >= threshold + margin
        p = len(np.where(x >= threshold + margin)[0])

        if p == self.number_of_features or p == 0:
            return None

        return p

    def _get_p_for_negative_x(self, x, threshold, maxitivity, margin):
        """Get the p value for a negative example (y = 0).

        Parameters
        ----------
        x : array-like of shape (1, n_features)
            Example, where n_features is the number of features.

        threshold : float
            Threshold.

        maxitivity : int
            Maxitivity (First hyperparameter of the Sugeno classifier).

        Margin : float
            Margin (Second hyperparameter of the Sugeno classifier).

        Returns
        ----------
        p : int or None
            p value describing the set which has to be contraint or None
            if no constraint can be derived.
        """

        # TODO less or less equal ?
        # number of elments < threshold - margin
        p = len(np.where(x < threshold - margin)[0])

        if p == self.number_of_features or p == 0:
            return None

        return p

    # get all subsets A of size k with A<B={p,...,m}, s.t. c(A) < threshold
    def _get_feature_subsets_of_size(self, k, B):
        """Get feature subsets of size k.

        Get all feature subsets of size k for the set B.

        Parameters
        ----------
        k : int
            Size of feature subsets to be determined.

        B : frozenset
            Set whose feature subsets to be determined.

        Returns
        ----------
        result : ndarray of shape (1,)
            Array containting all feature subsets which have to be constraint.
        """

        result = list()

        # there are only subsets with less size
        if len(B) < k:
            return np.array(result)

        for A in self.subsets_of_size[k]:
            if A.issubset(B):
                result.append(A)

        return np.array(result)

    def print_capacity(self, function):
        """Print capacity funtion.

        Parameters
        ----------
        funtion : dict
            Dictionary describing the capacity funtion.

        Returns
        ----------
        result : str
            String describing the capacity function in the format
            'set -> value'.
        """

        result = ''

        for key, value in function.items():
            result += str(key) + ' -> ' + str(value) + '\n'

        return result
