# This file will contain the API routers for the endpoints.
# Each endpoint will be a function that uses FastAPI's decorators to handle HTTP requests.
from fastapi import APIRouter, UploadFile, File
from typing import List
from ..models import Automation
from ..database import get_automations, create_automation, train_model, process_yaml_file

router = APIRouter()

@router.post("/generate", response_model=Automation)
async def generate_automation(automation: Automation):
    return create_automation(automation)

@router.post("/train")
async def train():
    train_model()
    return {"message": "Model training started"}

@router.post("/prepare")
async def prepare(file: UploadFile = File(...)):
    process_yaml_file(file.file)
    return {"message": "YAML file processed and data stored in database"}
