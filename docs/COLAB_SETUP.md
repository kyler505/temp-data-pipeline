# Running the Temp Data Pipeline in Google Colab

This guide explains how to set up and run the NOAA hourly data pipeline in Google Colab with Google Drive integration for persistent data storage.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Google Drive Integration](#google-drive-integration)
- [Running the Pipeline](#running-the-pipeline)
- [Data Persistence](#data-persistence)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Google account with Colab access
- Google Drive account (for persistence)
- Git repository access (public or authenticated)

---

## Quick Start (Automated Setup)

The easiest way to run the pipeline in Colab is using the **Bootstrap Utility**. This single cell handles cloning/syncing, dependency installation, and persistent storage in Google Drive.

1.  Open a new notebook in Colab.
2.  Mount Drive and clone the repo if you haven't already (one time setup):
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive
    !git clone https://github.com/kyler505/temp-data-pipeline.git
    ```
3.  In every notebook session, run this **Bootstrap Cell**:
    ```python
    import sys
    import os

    # 1. Add repo to path
    REPO_PATH = "/content/drive/MyDrive/temp-data-pipeline"
    sys.path.append(REPO_PATH)
    os.chdir(REPO_PATH)

    # 2. Sync and Setup
    !git pull  # Ensure we have the latest helper
    from tempdata.utils.colab import bootstrap
    bootstrap(use_wandb=True)
    ```

**Note:** The `bootstrap()` function automatically configures the pipeline to save all data and models directly to your Google Drive.

---

## Detailed Setup

### Step 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

This will prompt you to authenticate. Follow the link, grant permissions, and paste the authorization code.

### Step 2: Choose Your Workspace Location

**Option A: Clone to Drive (Recommended for persistence)**

```python
import os
from pathlib import Path

# Set workspace in Drive
WORKSPACE_DIR = Path('/content/drive/MyDrive/temp-data-pipeline')
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKSPACE_DIR)

# Clone if not exists
if not (WORKSPACE_DIR / '.git').exists():
    !git clone https://github.com/YOUR_USERNAME/temp-data-pipeline.git .
else:
    !git pull
```

**Option B: Clone to Colab VM (Faster, but temporary)**

```python
WORKSPACE_DIR = Path('/content/temp-data-pipeline')
if not WORKSPACE_DIR.exists():
    !git clone https://github.com/YOUR_USERNAME/temp-data-pipeline.git
os.chdir(WORKSPACE_DIR)
```

### Step 3: Install Dependencies

```python
!pip install -e .
```

This installs:
- `pandas` - Data manipulation
- `pyarrow` - Parquet file support
- `requests` - HTTP downloads

### Step 4: Verify Installation

```python
import sys
sys.path.insert(0, str(WORKSPACE_DIR))

from tempdata.fetch.noaa_hourly import fetch_noaa_hourly
print("✓ Package installed successfully")
```

---

## Google Drive Integration

### Setting Up Data Directory in Drive

For persistent storage, configure the pipeline to write data to Drive:

```python
from pathlib import Path
import sys

# Add workspace to path
WORKSPACE_DIR = Path('/content/drive/MyDrive/temp-data-pipeline')
sys.path.insert(0, str(WORKSPACE_DIR))

# Set data directory in Drive
DRIVE_DATA_DIR = Path('/content/drive/MyDrive/temp-data-pipeline-data')
DRIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Update config to use Drive location
from tempdata.config import data_root
import tempdata.config as config_module

# Override data_root to point to Drive
original_data_root = config_module.data_root
config_module.data_root = lambda: DRIVE_DATA_DIR

print(f"Data will be saved to: {DRIVE_DATA_DIR}")
```

### Alternative: Use Custom Output Directory

You can specify custom output directories directly in the function call:

```python
from tempdata.fetch.noaa_hourly import fetch_noaa_hourly
from pathlib import Path

drive_data_dir = Path('/content/drive/MyDrive/temp-data-pipeline-data')
drive_cache_dir = drive_data_dir / 'cache'

written = fetch_noaa_hourly(
    station_id='KLGA',
    start_date='2024-01-01',
    end_date='2024-02-01',
    out_dir=str(drive_data_dir / 'raw' / 'noaa_hourly' / 'KLGA'),
    cache_dir=str(drive_cache_dir / 'isd_csv' / 'KLGA'),
)
```

---

## Running the Pipeline

### Method 1: Using CLI Scripts

```python
# Fetch hourly data
!python scripts/fetch_noaa_hourly.py \
  --station KLGA \
  --start 2024-01-01 \
  --end 2024-02-01

# Or use the main pipeline script
!python scripts/run_pipeline.py \
  --station KLGA \
  --start 2024-01-01 \
  --end 2024-02-01
```

### Method 2: Direct Python API

```python
from tempdata.fetch.noaa_hourly import fetch_noaa_hourly
from pathlib import Path

# Set output to Drive
drive_output = Path('/content/drive/MyDrive/temp-data-pipeline-data/raw/noaa_hourly/KLGA')
drive_cache = Path('/content/drive/MyDrive/temp-data-pipeline-data/cache/isd_csv/KLGA')

written = fetch_noaa_hourly(
    station_id='KLGA',
    start_date='2024-01-01',
    end_date='2024-02-01',
    out_dir=str(drive_output),
    cache_dir=str(drive_cache),
    use_cache=True,  # Keep CSV cache for faster re-runs
)

print(f"Wrote {len(written)} parquet files:")
for path in written:
    print(f"  - {path}")
```

### Method 3: Batch Processing Multiple Stations

```python
from tempdata.fetch.noaa_hourly import fetch_noaa_hourly
from pathlib import Path

stations = ['KLGA']  # Add more stations as needed
start_date = '2023-01-01'
end_date = '2024-01-01'

drive_base = Path('/content/drive/MyDrive/temp-data-pipeline-data')

for station in stations:
    print(f"\nProcessing {station}...")
    written = fetch_noaa_hourly(
        station_id=station,
        start_date=start_date,
        end_date=end_date,
        out_dir=str(drive_base / 'raw' / 'noaa_hourly' / station),
        cache_dir=str(drive_base / 'cache' / 'isd_csv' / station),
    )
    print(f"  ✓ Wrote {len(written)} files for {station}")
```

---

## Data Persistence

### Understanding Data Locations

The pipeline creates two types of files:

1. **CSV Cache** (`data/raw/isd_csv/<station_id>/<year>.csv`)
   - Raw downloads from NOAA
   - Large files (~10-50 MB per year)
   - Can be regenerated, but caching saves time

2. **Parquet Output** (`data/raw/noaa_hourly/<station_id>/<year>.parquet`)
   - Processed hourly observations
   - Smaller, optimized format
   - This is your canonical data

### Recommended Drive Structure

```
MyDrive/
├── temp-data-pipeline/          # Code repository
│   ├── src/
│   ├── scripts/
│   └── stations/
│
└── temp-data-pipeline-data/      # Data (persistent)
    ├── cache/                    # CSV downloads (optional)
    │   └── isd_csv/
    │       └── KLGA/
    │           ├── 2023.csv
    │           └── 2024.csv
    │
    └── raw/                      # Processed parquet files
        └── noaa_hourly/
            └── KLGA/
                ├── 2023.parquet
                └── 2024.parquet
```

### Loading Persistent Data

```python
import pandas as pd
from pathlib import Path

# Load data from Drive
drive_data = Path('/content/drive/MyDrive/temp-data-pipeline-data')
parquet_file = drive_data / 'raw' / 'noaa_hourly' / 'KLGA' / '2024.parquet'

if parquet_file.exists():
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} rows")
    print(df.head())
    print(f"\nDate range: {df['ts_utc'].min()} to {df['ts_utc'].max()}")
else:
    print("File not found. Run the fetcher first.")
```

---

## Complete Colab Notebook Template

```python
# ============================================================================
# Cell 1: Setup and Drive Mount
# ============================================================================
from google.colab import drive
from pathlib import Path
import sys
import os

# Mount Drive
drive.mount('/content/drive')

# Set workspace
WORKSPACE_DIR = Path('/content/drive/MyDrive/temp-data-pipeline')
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKSPACE_DIR)

# Clone if needed
if not (WORKSPACE_DIR / '.git').exists():
    !git clone https://github.com/YOUR_USERNAME/temp-data-pipeline.git .
else:
    !git pull

sys.path.insert(0, str(WORKSPACE_DIR))

# ============================================================================
# Cell 2: Install Dependencies
# ============================================================================
!pip install -e .

# ============================================================================
# Cell 3: Configure Data Directories
# ============================================================================
DRIVE_DATA_DIR = Path('/content/drive/MyDrive/temp-data-pipeline-data')
DRIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Workspace: {WORKSPACE_DIR}")
print(f"Data directory: {DRIVE_DATA_DIR}")

# ============================================================================
# Cell 4: Run Pipeline
# ============================================================================
from tempdata.fetch.noaa_hourly import fetch_noaa_hourly

written = fetch_noaa_hourly(
    station_id='KLGA',
    start_date='2024-01-01',
    end_date='2024-02-01',
    out_dir=str(DRIVE_DATA_DIR / 'raw' / 'noaa_hourly' / 'KLGA'),
    cache_dir=str(DRIVE_DATA_DIR / 'cache' / 'isd_csv' / 'KLGA'),
)

print(f"\n✓ Pipeline completed. Wrote {len(written)} files.")

# ============================================================================
# Cell 5: Verify and Inspect Results
# ============================================================================
import pandas as pd

parquet_files = sorted(
    (DRIVE_DATA_DIR / 'raw' / 'noaa_hourly' / 'KLGA').glob('*.parquet')
)

if parquet_files:
    df = pd.read_parquet(parquet_files[0])
    print(f"Loaded {len(df)} rows from {parquet_files[0].name}")
    print("\nFirst few rows:")
    print(df.head())
    print(f"\nDate range: {df['ts_utc'].min()} to {df['ts_utc'].max()}")
    print(f"Temperature range: {df['temp_c'].min():.1f}°C to {df['temp_c'].max():.1f}°C")
    print(f"\nMissing temperature values: {df['temp_c'].isna().sum()}")
else:
    print("No parquet files found.")
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tempdata'"

**Solution:**
```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/temp-data-pipeline')
# Or use the WORKSPACE_DIR variable you set earlier
```

### Issue: "Station KLGA not found"

**Solution:** Ensure `stations/stations.csv` exists and contains KLGA:
```python
!cat stations/stations.csv
```

### Issue: Drive mount fails

**Solution:** Re-run the mount cell and re-authenticate:
```python
drive.mount('/content/drive', force_remount=True)
```

### Issue: Out of memory when downloading large datasets

**Solution:** Process one year at a time:
```python
for year in range(2020, 2024):
    fetch_noaa_hourly(
        station_id='KLGA',
        start_date=f'{year}-01-01',
        end_date=f'{year+1}-01-01',
        # ... other args
    )
```

### Issue: CSV downloads are slow

**Solution:**
- Use `use_cache=True` to avoid re-downloading
- Download to Drive cache once, then reuse
- Consider using `--no-cache` only for final parquet output

### Issue: Data not persisting after session ends

**Solution:** Ensure you're writing to Drive paths (`/content/drive/MyDrive/...`), not Colab VM paths (`/content/...`).

---

## Best Practices

1. **Always mount Drive first** - Do this in your first cell
2. **Use Drive for data, VM for code** - Clone code to Drive, but you can also use VM for faster git operations
3. **Cache CSVs in Drive** - Saves time on re-runs
4. **Process incrementally** - For large date ranges, process year-by-year
5. **Verify data after runs** - Always check parquet files were created and contain expected data
6. **Use descriptive Drive folder names** - Makes it easier to find data later

---

## Additional Resources

- [Google Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [Google Drive API](https://developers.google.com/drive/api)
- [NOAA Global Hourly Data](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database)

---

## Quick Reference

### Essential Commands

```python
# Mount Drive
drive.mount('/content/drive')

# Install package
!pip install -e .

# Run fetcher
!python scripts/fetch_noaa_hourly.py --station KLGA --start 2024-01-01 --end 2024-02-01

# Load parquet
df = pd.read_parquet('/content/drive/MyDrive/temp-data-pipeline-data/raw/noaa_hourly/KLGA/2024.parquet')
```

### Common CLI Options

- `--station KLGA` - Station identifier
- `--start YYYY-MM-DD` - Start date (inclusive)
- `--end YYYY-MM-DD` - End date (exclusive)
- `--force` - Re-download CSV cache
- `--no-cache` - Don't save CSV cache
