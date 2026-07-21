"""
Downloads the Bangladeshi Currency Detection dataset from Roboflow Universe
in YOLO11-ready format.

Usage:
    python3 scripts/download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

Get a free API key at: https://app.roboflow.com/settings/api
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Download the Bangladeshi Taka dataset from Roboflow.")
    parser.add_argument("--api-key", required=True, help="Your Roboflow API key")
    parser.add_argument("--workspace", default="tanvirtain", help="Roboflow workspace name")
    parser.add_argument("--project", default="bangladeshi-currency-detection", help="Roboflow project name")
    parser.add_argument("--version", type=int, default=1, help="Dataset version number")
    parser.add_argument("--location", default="dataset", help="Local folder to download into")
    args = parser.parse_args()

    try:
        from roboflow import Roboflow
    except ImportError:
        print("The 'roboflow' package isn't installed. Run: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov11", location=args.location)

    print(f"\nDataset downloaded to: {dataset.location}")
    print(f"Use this path when training:  --data {dataset.location}/data.yaml")


if __name__ == "__main__":
    main()
