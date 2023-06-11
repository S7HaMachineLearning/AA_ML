"""Models for the application."""

from enum import Enum
from typing import Optional, List, Dict, Any  # pylint: disable=no-name-in-module
from pydantic import BaseModel  # pylint: disable=no-name-in-module


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


class Automation(BaseModel):  # pylint: disable=no-name-in-module
    """Automation model."""
    triggers: Optional[List[Dict[str, Any]]] = None
    conditions: Optional[List[Dict[str, Any]]] = None
    actions: Optional[List[Dict[str, Any]]] = None


class NewAutomation(BaseModel):  # pylint: disable=no-name-in-module
    """New automation model."""
    value: str


class StartSequence(BaseModel):  # pylint: disable=no-name-in-module
    """Start sequence model."""
    alias: str
    trigger: dict
    condition: dict
    action: dict


class Sequence(BaseModel):  # pylint: disable=no-name-in-module
    """Sequence model."""
    start_sequence: str
