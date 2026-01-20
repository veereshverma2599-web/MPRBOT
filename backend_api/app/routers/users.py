from fastapi import APIRouter
from app.services.user_summary_service import UserSummaryService

router = APIRouter(prefix="/users", tags=["users"])

CSV_PATH = "data/cases_training.csv"


@router.get("/{username}/summary")
def get_user_summary(username: str):
    service = UserSummaryService(CSV_PATH)
    return service.compute_user_summary(username)

@router.get("/{username}/cases")
def get_user_cases(username: str, type: str):
    service = UserSummaryService(CSV_PATH)
    return service.get_user_cases(username, type)

