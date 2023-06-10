from fastapi import FastAPI
from data_preparation import DataPreparation
from machine_learning import MachineLearning
from models import Automation
from database_handler import DatabaseHandler

app = FastAPI()


@app.post("/automations/")
async def create_automation(automation: Automation):
    # Process the automation data
    data_prep = DataPreparation(directory="your_directory")
    processed_data = data_prep.process_automations(automation.dict())

    # Save the processed data to the database
    db = DatabaseHandler()
    db.save_to_database(processed_data)

    # Train the machine learning model
    ml_model = MachineLearning()
    ml_model.train_model(processed_data)

    # Save the trained model
    ml_model.save_model("model.h5")

    return {"message": "Automation created and model trained"}


@app.get("/automations/{automation_id}")
async def read_automation(automation_id: int):
    # Load the trained model
    ml_model = MachineLearning()
    ml_model.load_model("model.h5")

    # Use the model to generate an automation sequence
    automation_sequence = ml_model.generate_sequence(automation_id)

    return {"automation_sequence": automation_sequence}

#if __name__ == '__main__':
#    directory = 'D:/Temp/yaml'
    # location in the project where the automations are stored
    # src/automations
    # directory = 'automations'
#    print(f"Directory: {directory}")

#    data_prep = DataPreparation(directory)
#    data_prep.run_all_methods()

#    ml = MachineLearning(data_prep.automations)  # Pass the automations variable to MachineLearning
#    print(ml.data_modeling())
