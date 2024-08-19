import unittest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from linear_regression import normalize


class TestNormalize(unittest.TestCase):
    
    def test_basic(self):
        input = np.reshape(np.array([1, 2, 3, 5]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(result, expected, decimal=6, verbose=True)

    def test_zero(self):
        input = np.reshape(np.array([0, 0, 0, 0]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(result, expected, decimal=6, verbose=True)

    def test_negative(self):
        input = np.reshape(np.array([-1, -2, -8, -3]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(result, expected, decimal=6, verbose=True)

    def test_extreme_small(self):
        input = np.reshape(np.array([0.0001, 0.0002, 0.0003, -0.0004]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(result, expected, decimal=6, verbose=True)

    def test_extreme_large(self):
        input = np.reshape(np.array([10000, 200000, 50000, 10000, 2000000]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(result, expected, decimal=6, verbose=True)

    def test_same(self):
        input = np.reshape(np.array([5, 5]), (-1, 1))
        scaler = MinMaxScaler()
        expected = scaler.fit_transform(input)
        result = normalize(input)
        np.testing.assert_array_almost_equal(result, expected, decimal=6, verbose=True)

unittest.main()
