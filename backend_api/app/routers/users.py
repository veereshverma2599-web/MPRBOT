from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/{username}/summary")
def get_user_summary(username: str):
    return {
        "username": username,
        "total_cases": 42,
        "pending": 18,
        "overdue": 9,
        "critical": 3,
        "status_breakdown": {
            "Resolved": 20,
            "Pending": 18,
            "Invalid": 4
        }
    }
