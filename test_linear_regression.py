import unittest
import numpy as np

from linear_regression import normalize


class TestFunctions(unittest.TestCase):
    
    def test_normalize(self):
        input = np.array([1, 2, 3, 5])
        expected = np.array([0, 0.25, 0.5, 1])
        result = normalize(input)
        self.assertEqual(result.all(), expected.all(), f"Expected {expected} but got {result}")

        
        #print('passed') if(self.assertEqual(result, expected)) else print('failed')

    def test_zero(self):
        input = np.array([0, 0, 0, 0])
        expected = np.array([0, 0, 0, 0])
        result = normalize(input)
        self.assertEqual(result.all(), expected.all(), f"Expected {expected} but got {result}")

unittest.main()