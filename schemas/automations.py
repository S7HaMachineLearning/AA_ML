from pydantic import BaseModel

class AutomationBase(BaseModel):
    value: str
    type: int

class AutomationCreate(AutomationBase):
    pass

class Automation(AutomationBase):
    id: int

    class Config:
        orm_mode = True
