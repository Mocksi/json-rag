from pydantic import BaseModel

class FlexibleModel(BaseModel):
    __root__: dict
