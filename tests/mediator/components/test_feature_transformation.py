import numpy as np
import unittest

from classifier.mediator.components.feature_transformation import FeatureTransformation


class TestFeatureTransformation(unittest.TestCase):

	def test_number_of_components(self):
		# test for f = (f_1,...,f_m)

		test_runs = 5

		for i in range(1, test_runs):
			data = np.array([[j for j in range(test_runs)], [j + 1 for j in range(test_runs)], 
					[j * 2 for j in range(test_runs)], [j - 10 for j in range(test_runs)], 
					[j * 2 / 3 for j in range(test_runs)]])

			f = FeatureTransformation(data)

			number_of_components = len(f.function)
			expected_number_of_components = np.shape(data)[1]

			self.assertEqual(number_of_components, expected_number_of_components)

	def test_no_redundant_data(self):
		data = np.array([[1], [2], [3], [4], [5]])

		f = FeatureTransformation(data)

		self.assertEqual(f[0](1), 1/10)
		self.assertEqual(f[0](2), 3/10)
		self.assertEqual(f[0](3), 5/10)
		self.assertEqual(f[0](4), 7/10)
		self.assertEqual(f[0](5), 9/10)

	def test_reduntant_data(self):
		data = np.array([[2], [4], [5], [5], [7], [8], [8], [8], [11], [11]])

		f = FeatureTransformation(data)

		self.assertEqual(f[0](2), 1/20)
		self.assertEqual(f[0](4), 3/20)
		self.assertEqual(f[0](5), 6/20)
		self.assertEqual(f[0](7), 9/20)
		self.assertEqual(f[0](8), 13/20)
		self.assertEqual(f[0](11), 18/20)

	def test_values_between_points(self):
		data = np.array([[11], [8], [5], [8], [7], [8], [5], [4], [2], [11]])

		f = FeatureTransformation(data)

		self.assertTrue(np.isclose(f[0](4.5), 0.225))
		#self.assertEqual(f(4.5), 9/40)
		self.assertEqual(f[0](12.7), 1)

	def test_edge_cases(self):
		data = np.array([[6], [8], [3], [1], [8], [3]])

		f = FeatureTransformation(data)

		self.assertEqual(f[0](0.342), 0)
		self.assertEqual(f[0](10.345), 1)


if __name__ == '__main__':
	unittest.main()
