"""
Script to run the FastAPI server
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import uvicorn
import yaml

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("run_api")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run emotion classification API server")

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")

    return parser.parse_args()


def main() -> None:
    """Main function to run API server"""
    args = parse_args()

    # API inference should run from local artifacts only; avoid HF Hub network checks.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # Try to load API config, fall back to defaults
    try:
        api_config = load_config("configs/api_config.yaml")
        host = api_config["api"].get("host", args.host)
        port = api_config["api"].get("port", args.port)
        reload = api_config["api"].get("reload", args.reload)
        workers = api_config["api"].get("workers", args.workers)
    except (FileNotFoundError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        logger.warning(f"Failed to load configs/api_config.yaml ({exc}); using CLI/default values.")
        host = args.host
        port = args.port
        reload = args.reload
        workers = args.workers

    logger.info("Starting MoodNote AI - Emotion Classification API Server")
    logger.info(f"Host: {host} | Port: {port} | Reload: {reload} | Workers: {workers}")
    logger.info(f"API Documentation: http://{host}:{port}/docs")
    logger.info(f"Health Check: http://{host}:{port}/health")

    # Run server
    uvicorn.run(
        "src.inference.api:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Workers don't work with reload
    )


if __name__ == "__main__":
    main()
