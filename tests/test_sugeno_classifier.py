import unittest
import numpy as np

from sklearn.utils.estimator_checks import check_estimator
from classifier.sugeno_classifier import SugenoClassifier


class TestSugenoClassifier(unittest.TestCase):

	def test_compatibility(self):
		check_estimator(SugenoClassifier())

	def test_documentation(self):
		X = [[1, 3, 2], [2, 1, 3]]
		y = [0, 1]
		Z = [[3, 2, 1], [1, 2, 3]]	# Test data

		# first example
		sc = SugenoClassifier()
		sc.fit(X, y)
		result = sc.predict(Z)

		self.assertListEqual([0, 1], list(result))

		# second example
		sc = SugenoClassifier(maxitivity=2, margin=0.01)
		sc.fit(X, y)
		result = sc.predict(Z)

		self.assertListEqual([1, 1], list(result))

		# third example
		y = [2, 4]
		sc = SugenoClassifier()
		sc.fit(X, y)
		result = sc.predict(Z)

		self.assertListEqual([2, 4], list(result))

		y = ['one', 'two']
		ssc = SugenoClassifier()
		sc.fit(X, y)
		result = sc.predict(Z)

		self.assertListEqual(['one', 'two'], list(result))

if __name__ == '__main__':
	unittest.main()