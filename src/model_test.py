"""Test the model module."""
import unittest
from fastapi.testclient import TestClient
import main  # import the main module


class TestModel(unittest.TestCase):
    """Test the model module."""

    def test_generate_automation(self):
        """Test the generate_automation endpoint."""

        # Mock the MachineLearning class
        class MockMachineLearning(main.MachineLearning):
            """Mock the MachineLearning class"""

            def load_model(self, model_path, tokenizer_path):
                pass

            def generate_automation(self, start_sequence):
                return "Mock automation"

        # Replace the MachineLearning class in the main module with the mock class
        main.MachineLearning = MockMachineLearning

        client = TestClient(main.app)  # use the app from the main module

        # Define the request body
        request_body = {
            "start_sequence": "{\"alias\": \"Example automation\", \"trigger\": {\"platform\": \"state\", \"entity_id\": \"sun.sun\", \"to\": \"below_horizon\"}, \"condition\": {\"condition\": \"state\", \"entity_id\": \"device_tracker.person1\", \"state\": \"home\"}, \"action\": {\"service\": \"light.turn_on\", \"target\": {\"entity_id\": \"light.living_room\"}}"  # pylint: disable=line-too-long

        }

        # Make a request to the API
        response = client.post("/generate_automation", json=request_body)

        # Check that the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Check that the response body contains the expected result
        self.assertEqual(response.json(), {"generated_automation": "Mock automation"})


if __name__ == '__main__':
    unittest.main()
