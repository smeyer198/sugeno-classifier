import unittest
import numpy as np

from classifier.mediator.mediator import Mediator


class TestMediator(unittest.TestCase):

	def test_for_regression_targets(self):
		X = [[1, 2, 3], [2, 3, 4], [4, 5, 6]]

		mediator = Mediator()

		# no regression targets
		y = [2, 5, 2]

		result = mediator._check_for_regression_targets(y)
		
		self.assertTrue(result)

		# regression targets
		y = [0.1, 0.4, 0.1]

		result = mediator._check_for_regression_targets(y)

		self.assertFalse(result)

		# mixed targets
		y = [3, 6, 0.5]
		
		result = mediator._check_for_regression_targets(y)

		self.assertFalse(result)

		# ints are allowed to be a floating point
		y = [3.0, 6.0, 5.0]
		
		result = mediator._check_for_regression_targets(y)

		self.assertTrue(result)

		# strings are considered to be labels and no regression targets
		y = [3, '3.7', 1]

		result = mediator._check_for_regression_targets(y)

		self.assertTrue(result)

	def test_fit_components(self):
		# integration tests are done by the function check_estimator
		# in the test class for sugeno_classifier,
		# test for correct maxitivity
		X = [[3, 2, 1], [1, 2, 3]]
		y = [0, 1]

		mediator = Mediator()

		with self.assertRaises(ValueError):
			mediator.fit_components(X, y, 4, 0.01, 0.2)

	def test_normalized_x(self):
		f = DummyFeatureTransformation(5)

		mediator = Mediator()

		x = [7, 10, 3, 5, 12]

		result = mediator._get_normalized_x(x, f)
		expected_result = [0.5, 0.31, 0.25, 0.31, 0.75]

		self.assertListEqual(list(result), expected_result)

	def test_normalized_X(self):
		f = DummyFeatureTransformation(5)
		X = [[1, 7, 2, 2, 5],
			 [6, 2, 65, 2, 4],
			 [-7, 0.1, 55.3, 5, 10]]

		mediator = Mediator()

		result = mediator._get_normalized_X(X, f)
		expected_result = [[0.25, 0.5, 0.25, 0.25, 0.31],
						   [0.5, 0.25, 0.75, 0.25, 0.25],
						   [0.25, 0.25, 0.75, 0.31, 0.31]]

		self.assertListEqual(result.tolist(), expected_result)


class DummyFeatureTransformation():

	def __init__(self, m):
		self.function = list()

		for i in range(m):
			self.function.append(DummyFeatureTransformationComponent())

	def __getitem__(self, i):
		return self.function[i]

class DummyFeatureTransformationComponent():

	def __init__(self):
		pass

	def __call__(self, x):
		if x < 5:
			return 0.25

		if 5 < x and x < 10:
			return 0.5

		if x > 10:
			return 0.75
		
		return 0.31

if __name__ == '__main__':
	unittest.main()
