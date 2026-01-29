"""
Script to run the FastAPI server
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
import argparse
from src.utils.config import load_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run emotion classification API server")

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )

    return parser.parse_args()


def main():
    """Main function to run API server"""
    args = parse_args()

    # Try to load API config, fall back to defaults
    try:
        api_config = load_config("configs/api_config.yaml")
        host = api_config['api'].get('host', args.host)
        port = api_config['api'].get('port', args.port)
        reload = api_config['api'].get('reload', args.reload)
        workers = api_config['api'].get('workers', args.workers)
    except:
        host = args.host
        port = args.port
        reload = args.reload
        workers = args.workers

    print("=" * 60)
    print("MoodNote AI - Emotion Classification API Server")
    print("=" * 60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print(f"Workers: {workers}")
    print()
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Health Check: http://{host}:{port}/health")
    print("=" * 60)

    # Run server
    uvicorn.run(
        "src.inference.api:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1  # Workers don't work with reload
    )


if __name__ == "__main__":
    main()
