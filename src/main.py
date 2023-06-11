"""Main file for the API. Contains all endpoints and the main function."""
import urllib.request
from json import JSONDecodeError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from data_preparation import DataPreparation
from machine_learning import MachineLearning
import models
from database_handler import DatabaseHandler

# Create database connector for local DB
db = DatabaseHandler("database.db")

# load default model
ml_model = MachineLearning()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


## API ENDPOINTS ##

@app.post("/automations/")
async def create_automation(automation: models.Automation):
    # Process the automation data
    data_prep = DataPreparation(directory="automations/testset")
    processed_data = data_prep.process_automations(automation.dict())

    # Save the processed data to the database
    db.save_to_database(processed_data)

    return {"message": "Automation created and data saved to database"}


@app.post("/datamodelling/")
async def train_model(automation: models.Automation):
    # Process the automation data
    data_prep = DataPreparation(directory="automations/testset")
    processed_data = data_prep.process_automations(automation.dict())

    # Train the machine learning model
    ml_model.train_model(processed_data)

    # Save the trained model
    ml_model.save_model("model.h5")

    return {"message": "Model trained and saved"}


@app.get("/automations/{automation_id}")
async def read_automation(automation_id: int):
    # Load the trained model
    ml_model.load_model("model.h5")

    # TODO: Get the automation from the database
    # request_automation = db.get_automation(automation_id)
    request_automation = None

    # Use the model to generate an automation sequence
    automation_sequence = ml_model.generate_automation(request_automation)

    return {"automation_sequence": automation_sequence}


@app.post("/generate_automation")
def generate_automation(sequence: models.Sequence):
    # Initialize the machine learning
    ml = MachineLearning()

    # Load the model and tokenizer
    ml.load_model('model.h5', 'tokenizer.pickle')

    # Generate an automation
    generated_automation = ml.generate_automation(sequence.start_sequence)

    return {"generated_automation": generated_automation}

