"""
Downloads the Bangladeshi Currency Detection dataset from Roboflow Universe
in YOLO11-ready format.

Usage:
    # Preferred: set ROBOFLOW_API_KEY in a local .env file, then:
    python3 scripts/download_dataset.py

    # Or pass the key explicitly:
    python3 scripts/download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

Get a free API key at: https://app.roboflow.com/settings/api
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (parent of scripts/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def main():
    parser = argparse.ArgumentParser(description="Download the Bangladeshi Taka dataset from Roboflow.")
    parser.add_argument(
        "--api-key",
        default=os.getenv("ROBOFLOW_API_KEY"),
        help="Roboflow API key (default: ROBOFLOW_API_KEY from .env / environment)",
    )
    parser.add_argument("--workspace", default="tanvirtain", help="Roboflow workspace name")
    parser.add_argument("--project", default="bangladeshi-currency-detection", help="Roboflow project name")
    parser.add_argument("--version", type=int, default=1, help="Dataset version number")
    parser.add_argument("--location", default="dataset", help="Local folder to download into")
    args = parser.parse_args()

    placeholder = "your_roboflow_api_key_here"
    if not args.api_key or args.api_key.strip() == placeholder:
        print(
            "Missing Roboflow API key.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Set ROBOFLOW_API_KEY=your_real_key\n"
            "  3. Or pass --api-key YOUR_ROBOFLOW_API_KEY\n"
            "Get a free key at: https://app.roboflow.com/settings/api"
        )
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("The 'roboflow' package isn't installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov11", location=args.location)

    print(f"\nDataset downloaded to: {dataset.location}")
    print(f"Use this path when training:  --data {dataset.location}/data.yaml")


if __name__ == "__main__":
    main()
