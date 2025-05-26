import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import HTTPException
import logging
from app.routes import router as api_router
import asyncio
import signal
import shutil
import atexit
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag for shutdown
is_shutting_down = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    try:
        # Startup
        logger.info("Starting up the application...")
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        logger.info("Startup complete!")
        yield
    finally:
        # Shutdown
        global is_shutting_down
        if not is_shutting_down:
            is_shutting_down = True
            logger.info("Shutting down the application...")
            await cleanup()

async def cleanup():
    """Clean up temporary files and resources."""
    try:
        logger.info("Cleaning up resources...")
        # Clean up temporary files
        temp_dir = Path("temp")
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary directory")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {e}")
        
        # Clean up any other resources here
        logger.info("Cleanup complete")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "Server is running!"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc)}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )

def handle_exit(signum, frame):
    """Handle exit signals gracefully."""
    global is_shutting_down
    if not is_shutting_down:
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        is_shutting_down = True
        # Let Uvicorn handle the shutdown
        sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)