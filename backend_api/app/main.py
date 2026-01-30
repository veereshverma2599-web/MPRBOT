from fastapi import FastAPI
from app.routers import health , users

app = FastAPI(
    title="Auto MPR Backend API",
    version="0.1.0"
)

app.include_router(health.router)
app.include_router(users.router)
