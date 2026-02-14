from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.card import CardSchema


class BaseCard(ABC):
    card_type: str
    display_name: str
    description: str
    category: str
    execution_mode: str = "local"
    output_view_type: str

    config_schema: dict = {}
    input_schema: dict = {}
    output_schema: dict = {}

    @abstractmethod
    def execute(self, config: dict, inputs: dict, storage) -> dict:
        """Run the card logic. Returns dict of output key -> storage ref."""
        ...

    @abstractmethod
    def get_output_preview(self, outputs: dict, storage) -> dict:
        """Return a frontend-friendly preview of the card's output."""
        ...

    def to_schema(self) -> CardSchema:
        return CardSchema(
            card_type=self.card_type,
            display_name=self.display_name,
            description=self.description,
            category=self.category,
            execution_mode=self.execution_mode,
            config_schema=self.config_schema,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            output_view_type=self.output_view_type,
        )
