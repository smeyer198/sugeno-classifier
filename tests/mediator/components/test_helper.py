import unittest
import numpy as np

from classifier.mediator.components import helper as h

class TestHelper(unittest.TestCase):

	def test_feature_subset(self):
		instance_1 = np.array([6, 3, 7, 1, 5])
		sorted_instance_1 = np.sort(instance_1)

		permutation_1 = h._get_permutation_for_x(instance_1, sorted_instance_1)

		subset_1_index_1 = h.get_feature_subset(instance_1, 1)
		expected_subset_1_index_1 = set([4, 2, 5, 1, 3])
		self.assertSetEqual(subset_1_index_1, expected_subset_1_index_1)

		subset_1_index_2 = h.get_feature_subset(instance_1, 2)
		expected_subset_1_index_2 = set([2, 5, 1, 3])
		self.assertSetEqual(subset_1_index_2, expected_subset_1_index_2)

		subset_1_index_3 = h.get_feature_subset(instance_1, 3)
		expected_subset_1_index_3 = set([5, 1, 3])
		self.assertSetEqual(subset_1_index_3, expected_subset_1_index_3)

		subset_1_index_4 = h.get_feature_subset(instance_1, 4)
		expected_subset_1_index_4 = set([1, 3])
		self.assertSetEqual(subset_1_index_4, expected_subset_1_index_4)

		subset_1_index_5 = h.get_feature_subset(instance_1, 5)
		expected_subset_1_index_5 = set([3])
		self.assertSetEqual(subset_1_index_5, expected_subset_1_index_5)

	def test_permutation(self):

		# no duplicate elements
		instance_1 = np.array([5, 2, 7, 4, 6])
		sorted_instance_1 = np.sort(instance_1)

		permutation_1 = h._get_permutation_for_x(instance_1, sorted_instance_1)
		expected_permutation_1 = {1 : 2, 2 : 4, 3 : 1, 4 : 5, 5 : 3}

		self.assertDictEqual(permutation_1, expected_permutation_1)

		# duplicate elements
		instance_2 = np.array([1, 1, 1, 1])
		sorted_instance_2 = np.sort(instance_2)

		permutation_2 = h._get_permutation_for_x(instance_2, sorted_instance_2)
		expected_permutation_2 = {1 : 1, 2 : 2, 3 : 3, 4 : 4}

		self.assertDictEqual(permutation_2, expected_permutation_2)

		# only one element
		instance_3 = np.array([5])
		sorted_instance_3 = np.sort(instance_3)

		permutation_3 = h._get_permutation_for_x(instance_3, sorted_instance_3)
		expected_permutation_3 = {1 : 1}

		self.assertDictEqual(permutation_3, expected_permutation_3)


if __name__ == '__main__':
	unittest.main()