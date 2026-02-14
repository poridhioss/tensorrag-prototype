from __future__ import annotations

from app.models.card import CardSchema
from cards.base import BaseCard
from cards.data_load import DataLoadCard
from cards.data_split import DataSplitCard
from cards.evaluate import EvaluateCard
from cards.inference import InferenceCard
from cards.model_define import ModelDefineCard
from cards.model_define_gpu import ModelDefineGPUCard
from cards.train import TrainCard
from cards.train_gpu import TrainGPUCard
from cards.training_build_model import TrainingBuildModelCard
from cards.training_prepare_batch import TrainingPrepareBatchCard
from cards.training_init_optimizer import TrainingInitOptimizerCard
from cards.training_forward import TrainingForwardCard
from cards.training_loss import TrainingLossCard
from cards.training_zero_grad import TrainingZeroGradCard
from cards.training_backward import TrainingBackwardCard
from cards.training_optimizer_step import TrainingOptimizerStepCard

CARD_REGISTRY: dict[str, BaseCard] = {}


def _register(card: BaseCard) -> None:
    CARD_REGISTRY[card.card_type] = card


_register(DataLoadCard())
_register(DataSplitCard())
_register(ModelDefineCard())
_register(ModelDefineGPUCard())
_register(TrainCard())
_register(TrainGPUCard())
_register(EvaluateCard())
_register(InferenceCard())
_register(TrainingBuildModelCard())
_register(TrainingPrepareBatchCard())
_register(TrainingInitOptimizerCard())
_register(TrainingForwardCard())
_register(TrainingLossCard())
_register(TrainingZeroGradCard())
_register(TrainingBackwardCard())
_register(TrainingOptimizerStepCard())


def get_card(card_type: str) -> BaseCard:
    card = CARD_REGISTRY.get(card_type)
    if card is None:
        raise ValueError(
            f"Unknown card type: {card_type}. "
            f"Available: {list(CARD_REGISTRY.keys())}"
        )
    return card


def list_cards() -> list[CardSchema]:
    return [card.to_schema() for card in CARD_REGISTRY.values()]
