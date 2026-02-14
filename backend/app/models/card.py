from pydantic import BaseModel


class CardSchema(BaseModel):
    card_type: str
    display_name: str
    description: str
    category: str
    execution_mode: str
    config_schema: dict
    input_schema: dict
    output_schema: dict
    output_view_type: str
