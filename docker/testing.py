import unittest
import numpy as np
import pickle
#import from heart_model.py
from heart_model import model
from heart_model import scaler

class TestHeartPrediction(unittest.TestCase):
    def test_heart_prediction(self):
        # Test case 1
        input_data_1 = np.array([53, 0, 63, 1, 60, 0, 368000, 0.8, 135, 1, 0, 22])
        input_data_1 = input_data_1.reshape(1, -1)

        # Apply scaling transformation
        scaled_data_1 = scaler.transform(input_data_1)

        # Make prediction
        result_1 = model.predict(scaled_data_1)
        print(f"Test case 1 result is {result_1[0]}") 


         # test validation
        self.assertEqual(result_1[0], 0)

          # Check if the output is either 0 or 1 (DAVEs thoughts)
        #self.assertIn(prediction[0], [0, 1])

        # Test case 2
        input_data_2 = np.array([80, 0, 148, 1, 38, 0, 149000, 1.9, 144, 1, 1, 23])
        input_data_2 = input_data_2.reshape(1, -1)

        # Apply scaling transformation
        scaled_data_2 = scaler.transform(input_data_2)

        # Make prediction
        result_2 = model.predict(scaled_data_2)
        print(f"Test case 2 result is {result_2[0]}")

            # test validation
        self.assertEqual(result_2[0], 1)

# figure out why it's called this??
# wamt to run a python file directly
# run is a model and a package
if __name__ == '__main__':
    unittest.main()
