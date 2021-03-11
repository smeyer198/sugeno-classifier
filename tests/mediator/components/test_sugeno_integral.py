import numpy as np
import unittest

from classifier.mediator.components.capacity import Capacity
from classifier.mediator.components.sugeno_integral import SugenoIntegral

class TestSugenoIntegral(unittest.TestCase):

	def test_integral_value(self):
		x_data = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])
		y_data = np.array([0, 0, 1, 1])
		threshold = 0.5

		c = self._get_dummy_capacity(x_data, y_data, 0.5)
		#print(c.capacity_function)
		s = SugenoIntegral(c)

		instance_1 = np.array([0.4, 0.2, 0.7])
		integral_value_1 = s._get_integral_value(instance_1, c)
		expected_integral_value_1 = 0.7

		self.assertEqual(integral_value_1, expected_integral_value_1)

		instance_2 = np.array([0.2, 0.1, 0.1])
		integral_value_2 = s._get_integral_value(instance_2, c)
		expected_integral_value_2 = 0.1

		self.assertEqual(integral_value_2, expected_integral_value_2)

	# Dummy capacity to compute integral values
	def _get_dummy_capacity(self, x_values, y_values, threshold):
		capacity = Capacity(x_values, y_values)

		maxitivity = np.shape(x_values)[1]

		function = capacity.compute_capacity(threshold, maxitivity, 0)

		return function

if __name__ == '__main__':
	unittest.main()