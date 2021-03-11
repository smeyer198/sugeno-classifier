import numpy as np
import unittest

from classifier.mediator.components.threshold import Threshold


class TestThreshold(unittest.TestCase):
	
	def test_functionality(self):
		x_data = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
		y_data = [0, 0, 0, 1]

		threshold = Threshold(x_data, y_data)

		result = threshold._solve_linear_program()

		self.assertAlmostEqual(result, 0.4)

if __name__ == '__main__':
	unittest.main()