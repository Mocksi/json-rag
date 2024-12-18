from pydantic import BaseModel

class FlexibleModel(BaseModel):
    """
    A flexible Pydantic model that can validate any JSON structure.
    
    Attributes:
        __root__ (dict): The root JSON object to validate
        
    Usage:
        validated = FlexibleModel.parse_obj(json_data)
        json_dict = validated.__root__
        
    Note:
        Used for basic JSON validation without enforcing specific schema
    """
    __root__: dict
