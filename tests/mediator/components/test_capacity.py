import unittest
import numpy as np

from itertools import permutations
from classifier.mediator.components.capacity import Capacity


class TestCapacity(unittest.TestCase):

	def test_powerset(self):
		s = set([1,3,5,7])
		x_data = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]

		c = Capacity(x_data, [])

		powerset = c._get_powerset(s)
		expected_powerset = [frozenset(), frozenset([1]), frozenset([3]), frozenset([5]), frozenset([7]), 
						frozenset([1, 3]), frozenset([1, 5]), frozenset([1, 7]), frozenset([3, 5]), frozenset([3, 7]), frozenset([5,7]), 
						frozenset([1, 3, 5]), frozenset([1, 3, 7]), frozenset([1, 5, 7]), frozenset([3, 5, 7]), frozenset([1, 3, 5, 7])]
		
		self.assertListEqual(list(powerset), expected_powerset)

	def test_subset_ids(self):
		c = Capacity([[]], [])

		set_1 = frozenset([1, 2])
		set_2 = frozenset([1, 5, 7])
		set_3 = frozenset([1, 1, 1, 4, 4, 5 ,7])
		set_4 = frozenset()

		s = list([set_1, set_2, set_3, set_4])
		subset_ids = c._get_subset_ids(s)
		
		self.assertEqual(subset_ids[set_1], 0)
		self.assertEqual(subset_ids[set_2], 1)
		self.assertEqual(subset_ids[frozenset([1, 4, 5, 7])], 2)
		self.assertEqual(subset_ids[set_4], 3)

	def test_subsets_of_size(self):
		c = Capacity([[]], [])

		set_1 = frozenset([1, 2, 3])
		set_2 = frozenset([1])
		set_3 = frozenset()
		set_4 = frozenset([1, 2, 4])
		set_5 = frozenset([1, 1, 1, 3, 3, 3, 5, 5])

		s = list([set_1, set_2, set_3, set_4, set_5])
		subsets_of_size = c._get_subsets_of_size(s)

		self.assertIn(set_1, subsets_of_size[3])
		self.assertIn(set_2, subsets_of_size[1])
		self.assertIn(set_3, subsets_of_size[0])
		self.assertIn(set_4, subsets_of_size[3])
		self.assertIn(set_5, subsets_of_size[3])

		self.assertEqual(len(subsets_of_size[3]), 3)
		self.assertEqual(len(subsets_of_size[2]), 0)

	def test_capacity_components(self):
		X = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]

		c = Capacity(X, [])

		expected_base_set = set([1, 2, 3])
		self.assertSetEqual(c.base_set, expected_base_set)

		expected_powerset = [frozenset([1]), frozenset([2]), frozenset([3]),
							frozenset([1, 2]), frozenset([1, 3]), frozenset([2, 3]),
							frozenset([1, 2, 3])]
		self.assertListEqual(list(c.powerset), expected_powerset)

		expected_subset_ids = {frozenset([1]) : 0, frozenset([2]) : 1, frozenset([3]) : 2, frozenset([1, 2]) : 3,
							frozenset([1, 3]) : 4, frozenset([2, 3]) : 5, frozenset([1, 2, 3]) : 6}
		self.assertDictEqual(c.subset_ids, expected_subset_ids)

		expected_subsets_of_size = {1 : [frozenset([1]), frozenset([2]), frozenset([3])], 
									2 : [frozenset([1, 2]), frozenset([1, 3]), frozenset([2, 3])], 3: [frozenset([1, 2, 3])]}
		self.assertDictEqual(dict(c.subsets_of_size), expected_subsets_of_size)

	def test_capacity_from_result(self):
		X = [[1, 2, 3]]

		# ids 0: {1}, 1: {2}, 2: {3}, 3: {1,2}, 4: {1,3}, 5: {2,3}, 6: {1,2,3}
		result = np.array([1, 2, -3, 4, -5, 6, 7])

		c = Capacity(X, [])
		function = c._get_capacity_from_result(result, 3)

		self.assertEqual(function[frozenset()], 0)
		self.assertEqual(function[frozenset([1])], 1)
		self.assertEqual(function[frozenset([2])], 2)
		self.assertEqual(function[frozenset([3])], -3)
		self.assertEqual(function[frozenset([1, 2])], 4)
		self.assertEqual(function[frozenset([1, 3])], -5)
		self.assertEqual(function[frozenset([2, 3])], 6)
		self.assertEqual(function[frozenset([1, 2, 3])], 1)

		# maxitivity: 1
		result = np.array([1, -2, -3])
		function = c._get_capacity_from_result(result, 1)

		self.assertEqual(function[frozenset([])], 0)
		self.assertEqual(function[frozenset([1])], 1)
		self.assertEqual(function[frozenset([2])], -2)
		self.assertEqual(function[frozenset([3])], -3)
		self.assertEqual(function[frozenset([1, 2, 3])], 1)

		with self.assertRaises(KeyError):
			a = function[frozenset([1, 2])]

		with self.assertRaises(KeyError):
			b = function[frozenset([1, 3])]

		with self.assertRaises(KeyError):
			c = function[frozenset([2, 3])]

	def test_update_capacity_for_maxitivity(self):
		function = dict()

		X = [[1, 2, 3]]

		c = Capacity(X, [])

		function[frozenset([1])] = 100
		function[frozenset([2])] = 0
		function[frozenset([3])] = 0

		result = c._update_capacity_for_maxitivity(1, function)

		self.assertEqual(result[frozenset([1, 2])], 100)
		self.assertEqual(result[frozenset([1, 3])], 100)
		self.assertEqual(result[frozenset([2, 3])], 0)

		X = [[1, 2, 3, 4]]

		c = Capacity(X, [])

		function[frozenset([1, 2])] = 0
		function[frozenset([1, 3])] = 10
		function[frozenset([1, 4])] = 20
		function[frozenset([2, 3])] = 10
		function[frozenset([2, 4])] = 20
		function[frozenset([3, 4])] = 0

		result = c._update_capacity_for_maxitivity(2, function)

		self.assertEqual(result[frozenset([1, 2, 3])], 10)
		self.assertEqual(result[frozenset([1, 2, 4])], 20)
		self.assertEqual(result[frozenset([1, 3, 4])], 20)
		self.assertEqual(result[frozenset([2, 3, 4])], 20)

	def test_k_subset_for_set(self):
		X = [[1, 2, 3]]

		c = Capacity(X, [])

		# get all subsets of {1,2} with size 1
		subsets_for_1_2 = list(c._get_k_subsets_for_set(1, frozenset([1, 2])))

		self.assertListEqual(subsets_for_1_2, [frozenset([1]), frozenset([2])])

		# subsets of size 1
		subsets_for_1_2_3 = list(c._get_k_subsets_for_set(1, frozenset([1, 2, 3])))

		self.assertListEqual(subsets_for_1_2_3, [frozenset([1]), frozenset([2]), frozenset([3])])

		# subsets of size 2
		subsets_for_1_2_3 = list(c._get_k_subsets_for_set(2, frozenset([1, 2, 3])))

		self.assertListEqual(subsets_for_1_2_3, [frozenset([1, 2]), frozenset([1, 3]), frozenset([2, 3])])

	def test_maximum_of_subsets(self):
		X = [[1, 2, 3, 4]]

		c = Capacity(X, [])

		function = dict()
		function[frozenset([1, 2])] = 1
		function[frozenset([1, 3])] = 2
		function[frozenset([1, 4])] = 2
		function[frozenset([2, 3])] = 1
		function[frozenset([2, 4])] = 3
		function[frozenset([3, 4])] = 0

		subsets_of_size_2 = c.subsets_of_size[2]

		maximum = c._get_maximum_of_subsets(subsets_of_size_2, function)
		self.assertEqual(maximum, 3)

		function[frozenset([2, 4])] = 0

		maximum = c._get_maximum_of_subsets(subsets_of_size_2, function)
		self.assertEqual(maximum, 2)

	def test_monotony_constraints(self):
		# four attributes
		m = 4
		X = [np.arange(m)]

		c = Capacity(X, [])

		monotony_constraints = c._get_monotony_constraints(0, m)

		for A, B in monotony_constraints:
			self.assertEqual(len(A) + 1, len(B))

			self.assertTrue(A.issubset(B))

	def test_median_constraints(self):

		# five attributes
		x = np.array([6, 3, 7, 1, 5])

		# positive example
		y = np.array([1])

		c = Capacity([x], y)

		median_constraints = c._get_feature_subsets_for_x_y(x, y, 5, 5, 0)

		# one contraint has to be added, feature subset A_{\sigam (3)}
		self.assertEqual(len(median_constraints), 1)
		self.assertSetEqual(median_constraints[0], frozenset([1, 3, 5]))

		median_constraints = c._get_feature_subsets_for_x_y(x, y, 5, 2, 0)

		# still one constraint because of positive example, feature subset A_{\sigma (1)}
		self.assertEqual(len(median_constraints), 1)
		self.assertSetEqual(median_constraints[0], frozenset([1, 3]))

		# negative example
		y = np.array([0])

		c = Capacity([x], y)

		median_constraints = c._get_feature_subsets_for_x_y(x, y, 5, 5, 0)

		# one constaint has to be added, feature subset A_{\sigma (3)}
		self.assertEqual(len(median_constraints), 1)
		self.assertSetEqual(median_constraints[0], frozenset([1, 3, 5]))

		median_constraints = c._get_feature_subsets_for_x_y(x, y, 5, 2, 0)

		# constraints fpr all 2-subsets of {1, 3, 5} 
		self.assertEqual(len(median_constraints), 3)
		self.assertTrue(median_constraints[0] != median_constraints[1])
		self.assertTrue(median_constraints[0] != median_constraints[2])
		self.assertTrue(median_constraints[1] != median_constraints[2])

		expected_subsets = [frozenset([1, 3]), frozenset([1, 5]), frozenset([3, 5])]
		self.assertTrue(median_constraints[0] in expected_subsets)
		self.assertTrue(median_constraints[1] in expected_subsets)
		self.assertTrue(median_constraints[2] in expected_subsets)
		self.assertTrue(frozenset([1, 3, 5]) not in expected_subsets)

	def test_median_constraints_for_three_attributes(self):
		# test contraints for 3 attributes
		x = np.array([5, 1, 3])
		y = np.array([1])

		c = Capacity([x], y)

		# no constraints have to be added
		median_constraints = c._get_feature_subsets_for_x_y(x, y, 6, 3, 0)
		self.assertEqual(len(median_constraints), 0)

		median_constraints = c._get_feature_subsets_for_x_y(x, y, 0, 3, 0)
		self.assertEqual(len(median_constraints), 0)

		# maxitivity: 1
		median_constraints = c._get_feature_subsets_for_x_y(x, y, 2, 1, 0)
		self.assertEqual(len(median_constraints), 1)
		self.assertSetEqual(median_constraints[0], frozenset([1]))

		median_constraints = c._get_feature_subsets_for_x_y(x, y, 4, 1, 0)
		self.assertEqual(len(median_constraints), 1)
		self.assertSetEqual(median_constraints[0], frozenset([1]))

		# maxitivity: 2
		median_constraints = c._get_feature_subsets_for_x_y(x, y, 2, 2, 0)
		self.assertEqual(len(median_constraints), 1)
		self.assertSetEqual(median_constraints[0], frozenset([1, 3]))

		median_constraints = c._get_feature_subsets_for_x_y(x, y, 4, 1, 0)
		self.assertEqual(len(median_constraints), 1)
		self.assertSetEqual(median_constraints[0], frozenset([1]))

		# negative example
		y = np.array([0])

		# maxitivity: 1
		median_constraints = c._get_feature_subsets_for_x_y(x, y, 2, 1, 0)
		self.assertEqual(len(median_constraints), 2)
		self.assertSetEqual(set(median_constraints), set([frozenset([1]), frozenset([3])]))

		median_constraints = c._get_feature_subsets_for_x_y(x, y, 4, 1, 0)
		self.assertEqual(len(median_constraints), 1)
		self.assertSetEqual(median_constraints[0], frozenset([1]))

		# maxitivity: 2
		median_constraints = c._get_feature_subsets_for_x_y(x, y, 2, 2, 0)
		self.assertEqual(len(median_constraints), 1)
		self.assertSetEqual(median_constraints[0], frozenset([1, 3]))

		median_constraints = c._get_feature_subsets_for_x_y(x, y, 4, 2, 0)
		self.assertEqual(len(median_constraints), 1)
		self.assertSetEqual(median_constraints[0], frozenset([1]))


if __name__ == '__main__':
	unittest.main()