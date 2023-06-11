from fastapi.testclient import TestClient
import pytest
from main import app, MachineLearning
from src import main

# I've created a mock MachineLearning class that doesn't load any model or tokenizer and always
# returns "Mock automation" when the generate_automation method is called.
# I've replaced the MachineLearning class in the main module with this mock class for the duration of the test.
# I've defined the request body, made a request to the API, and checked that the response status code is 200
# and the response body contains the expected result.

client = TestClient(app)

def test_generate_automation():
    # Mock the MachineLearning class
    class MockMachineLearning(MachineLearning):
        def load_model(self, model_path, tokenizer_path):
            pass

        def generate_automation(self, start_sequence):
            return "Mock automation"

    # Replace the MachineLearning class in the main module with the mock class
    main.MachineLearning = MockMachineLearning

    # Define the request body
    request_body = {
        "start_sequence": "{\"alias\": \"Example automation\", \"trigger\": {\"platform\": \"state\", \"entity_id\": \"sun.sun\", \"to\": \"below_horizon\"}, \"condition\": {\"condition\": \"state\", \"entity_id\": \"device_tracker.person1\", \"state\": \"home\"}, \"action\": {\"service\": \"light.turn_on\", \"target\": {\"entity_id\": \"light.living_room\"}}"
    }

    # Make a request to the API
    response = client.post("/generate_automation", json=request_body)

    # Check that the response status code is 200 (OK)
    assert response.status_code == 200

    # Check that the response body contains the expected result
    assert response.json() == {"generated_automation": "Mock automation"}
