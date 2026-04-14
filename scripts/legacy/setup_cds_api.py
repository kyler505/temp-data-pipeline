#!/usr/bin/env python3
"""Setup script for CDS API credentials (ERA5 data access).

Run this script to configure your CDS API credentials for ERA5 data fetching.

Usage:
    python scripts/setup_cds_api.py --key YOUR_API_KEY

Or interactively:
    python scripts/setup_cds_api.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CDS_API_URL = "https://cds.climate.copernicus.eu/api"


def get_cdsapirc_path() -> Path:
    """Get the path to the .cdsapirc file."""
    return Path.home() / ".cdsapirc"


def write_credentials(api_key: str, url: str = CDS_API_URL) -> Path:
    """Write CDS API credentials to ~/.cdsapirc."""
    cdsapirc_path = get_cdsapirc_path()
    content = f"url: {url}\nkey: {api_key}\n"
    cdsapirc_path.write_text(content)
    return cdsapirc_path


def check_credentials() -> bool:
    """Check if CDS API credentials are configured."""
    cdsapirc_path = get_cdsapirc_path()
    if not cdsapirc_path.exists():
        return False

    content = cdsapirc_path.read_text()
    return "key:" in content and "url:" in content


def test_connection() -> bool:
    """Test CDS API connection."""
    try:
        import cdsapi
        client = cdsapi.Client()
        print("✓ CDS API client initialized successfully")
        return True
    except ImportError:
        print("⚠️  cdsapi not installed. Run: pip install cdsapi")
        return False
    except Exception as e:
        print(f"⚠️  CDS API connection failed: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Setup CDS API credentials for ERA5 data access",
        epilog="""
To get your API key:
  1. Create a free account at https://cds.climate.copernicus.eu/
  2. Go to your profile page
  3. Copy your API key

Example:
  python scripts/setup_cds_api.py --key 12345:abcdef-1234-5678-abcd-ef1234567890
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--key",
        help="CDS API key (format: UID:KEY)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test existing credentials without modifying",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show path to credentials file",
    )

    args = parser.parse_args()

    cdsapirc_path = get_cdsapirc_path()

    if args.show:
        print(f"Credentials file: {cdsapirc_path}")
        if cdsapirc_path.exists():
            print("Status: Configured")
        else:
            print("Status: Not configured")
        return 0

    if args.test:
        if check_credentials():
            print(f"Credentials found at: {cdsapirc_path}")
            return 0 if test_connection() else 1
        else:
            print(f"No credentials found at: {cdsapirc_path}")
            return 1

    # Get API key
    api_key = args.key
    if not api_key:
        if cdsapirc_path.exists():
            print(f"Credentials already exist at: {cdsapirc_path}")
            overwrite = input("Overwrite? [y/N]: ").strip().lower()
            if overwrite != "y":
                print("Aborted.")
                return 0

        print("\nGet your API key from: https://cds.climate.copernicus.eu/profile")
        print("Format: UID:KEY (e.g., 12345:abcdef-1234-5678-abcd-ef1234567890)\n")
        api_key = input("Enter your CDS API key: ").strip()

        if not api_key:
            print("No key provided. Aborted.")
            return 1

    # Validate format
    if ":" not in api_key:
        print("⚠️  API key should be in format UID:KEY")
        print("   Example: 12345:abcdef-1234-5678-abcd-ef1234567890")
        proceed = input("Continue anyway? [y/N]: ").strip().lower()
        if proceed != "y":
            return 1

    # Write credentials
    path = write_credentials(api_key)
    print(f"✓ Credentials written to: {path}")

    # Test connection
    print("\nTesting connection...")
    if test_connection():
        print("\n✓ ERA5 data fetching is now available!")
        return 0
    else:
        print("\n⚠️  Connection test failed, but credentials were saved.")
        print("   You may need to install dependencies: pip install cdsapi xarray netCDF4")
        return 1


if __name__ == "__main__":
    sys.exit(main())
