# This file will define the data models for the application.
"""Models for the application."""

from enum import Enum
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class SensorType(Enum):  # pylint: disable=too-few-public-methods
    """Enum for sensor types."""
    TEMPERATURE = 1
    HUMIDITY = 2


class Sensor(BaseModel):  # pylint: disable=too-few-public-methods
    """Sensor model."""
    id: int
    friendlyName: str
    haSensorId: str
    type: SensorType
    createdOn: str
    updatedOn: str
    deleted: int


class NewSensor(BaseModel):  # pylint: disable=too-few-public-methods
    """New sensor model."""
    friendlyName: str
    haSensorId: str
    type: int


class HaSensor(BaseModel):  # pylint: disable=too-few-public-methods
    """Home assistant sensor model."""
    entityId: str
    friendlyName: str
    state: str


class Automation(BaseModel):
    triggers: Optional[List[Dict[str, Any]]] = None
    conditions: Optional[List[Dict[str, Any]]] = None
    actions: Optional[List[Dict[str, Any]]] = None


class NewAutomation(BaseModel):
    """New automation model."""
    value: str


class StartSequence(BaseModel):
    alias: str
    trigger: dict
    condition: dict
    action: dict


class Sequence(BaseModel):
    start_sequence: str
