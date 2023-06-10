# This is the entry point for FastAPI application.
# It will create the FastAPI application instance and include the routers from the api module.

from routers import automations
from json import JSONDecodeError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from database import DatabaseConnector
import myclass
import models

# Create database connector for local DB
databaseConnector = DatabaseConnector("database.db")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


## API ENDPOINTS ##

@app.post("/process_automations")
def process_automations(automation: Automation):
    directory = automation.directory
    automations = []

    correct_yaml_files(directory)

    for filename in os.listdir(directory):
        if filename.endswith(".yaml"):
            file = os.path.join(directory, filename)
            file_automations = process_automations(file)
            automations.extend(file_automations)

    return {"message": "Automations processed successfully"}

@app.post("/preprocess_data")
def preprocess_data(automations: list):
    encoded_platforms, encoded_conditions, encoded_services = prepocess_data(automations)
    return {"encoded_platforms": encoded_platforms, "encoded_conditions": encoded_conditions, "encoded_services": encoded_services}

@app.post("/feature_engineering")
def feature_engineering(automations: list):
    num_triggers, num_conditions, num_actions, has_state_trigger, has_sunrise_trigger = feature_engineering(automations)
    return {"num_triggers": num_triggers, "num_conditions": num_conditions, "num_actions": num_actions, "has_state_trigger": has_state_trigger, "has_sunrise_trigger": has_sunrise_trigger}

@app.post("/data_modeling")
def data_modeling(automations: list):
    automation = data_modeling(automations)
    return {"automation": automation}
