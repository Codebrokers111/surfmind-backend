"""
This module defines the primary application controller for the FastAPI backend.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.controller.core_controller import router as core_router
from src.controller.sync_controller import router as sync_router
from src.utility.logger import AppLogger
from src.utility.settings import settings

AppLogger.init(
    level=logging.DEBUG if settings.debug else logging.INFO,
    log_to_file=True,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(core_router)
app.include_router(sync_router)


@app.get("/", tags=["Health"])
def health_check():
    """Simple root health endpoint confirming setup."""
    return {"status": "ok", "message": "Setup Successfull"}


@app.get("/health", tags=["Health"])
def health_check():
    """Secondary health endpoint to monitor FastAPI server state."""
    return {"status": "ok", "message": "Surfmind FastAPI server running!"}
