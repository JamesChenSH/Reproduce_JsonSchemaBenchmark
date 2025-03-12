from pydantic import BaseModel, Field, constr, StringConstraints
from typing_extensions import Annotated
import json

class ReturnedModel(BaseModel):
    reasoning: Annotated[
        str,
        constr(max_length=1000)
    ]
    answer: int = Field(pattern=r'[1-9][0-9]{0,9}')

print(json.dumps(ReturnedModel.model_json_schema()))