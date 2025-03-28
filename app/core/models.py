from typing import Dict, Any
from pydantic import RootModel


class FlexibleModel(RootModel[Dict[str, Any]]):
    """
    A flexible Pydantic model that can validate any JSON structure.

    Usage:
        validated = FlexibleModel.model_validate(json_data)
        json_dict = validated.root

    Note:
        Used for basic JSON validation without enforcing specific schema
    """

    pass
