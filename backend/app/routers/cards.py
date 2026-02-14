from fastapi import APIRouter

from cards.registry import list_cards

router = APIRouter(tags=["cards"])


@router.get("/cards")
def get_cards():
    return list_cards()
