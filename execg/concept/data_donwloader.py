import os
import zipfile

from execg.misc import download_from_gdrive

# Registry mapping data_name to Google Drive file ID
TCAV_DATA_REGISTRY = {
    "physionet2021": {
        "gdrive_id": "1tZGr8xPDGfJIe_gnF74spIJti1Qlds9i",
        "archive_name": "physionet2021_numpy.zip",
        "description": "PhysioNet Challenge 2021 (preprocessed numpy files)",
        "compressed_size": "~6 GB",
        "extracted_size": "~10 GB",
    },
}


def download_tcav_concept_data(
    data_name: str,
    data_dir: str,
    download: bool = False,
    remove_archive: bool = True,
) -> str:
    """Download TCAV concept data from Google Drive.

    This downloads pre-processed numpy files for TCAV concept analysis.
    The data is compressed in a zip file and will be extracted after download.

    Args:
        data_name: Name of the dataset (must be in TCAV_DATA_REGISTRY).
            Available: {list(TCAV_DATA_REGISTRY.keys())}
        data_dir: Directory to save the extracted data.
        download: If True, download the data if not found.
            If False, raise FileNotFoundError when data is missing.
            Default: False
        remove_archive: If True, remove the zip file after extraction to save space.
            Default: True

    Returns:
        Path to the extracted data directory.

    Raises:
        ValueError: If data_name is not in registry.
        FileNotFoundError: If data doesn't exist and download is False.
        RuntimeError: If download or extraction fails.

    Example:
        >>> # Download and extract TCAV concept data
        >>> data_path = download_tcav_concept_data(
        ...     data_name="physionet2021",
        ...     data_dir="./data/tcav_concepts",
        ...     download=True
        ... )
        >>> print(f"Data available at: {data_path}")
    """
    if data_name not in TCAV_DATA_REGISTRY:
        available = list(TCAV_DATA_REGISTRY.keys())
        raise ValueError(
            f"Unknown data_name '{data_name}'. Available datasets: {available}"
        )

    data_info = TCAV_DATA_REGISTRY[data_name]
    gdrive_id = data_info["gdrive_id"]
    archive_name = data_info["archive_name"]
    compressed_size = data_info["compressed_size"]
    extracted_size = data_info["extracted_size"]

    os.makedirs(data_dir, exist_ok=True)

    # Check if data already exists (marker file)
    marker_file = os.path.join(data_dir, ".download_complete")
    if os.path.exists(marker_file):
        return data_dir

    if not download:
        print("=" * 60)
        print(f"Data not found at: {data_dir}")
        print("=" * 60)
        print(f"To download '{data_name}' data, set download=True")
        print("Storage Requirements:")
        print(f"  - Download size: {compressed_size} (compressed)")
        print(f"  - Extracted size: {extracted_size}")
        print("=" * 60)
        raise FileNotFoundError(
            f"TCAV concept data '{data_name}' not found at '{data_dir}'. "
            "Set download=True to download automatically."
        )

    # Print storage warning
    print("=" * 60)
    print(f"TCAV Concept Data Download: {data_name}")
    print("=" * 60)
    print("Storage Requirements:")
    print(f"  - Download size: {compressed_size} (compressed)")
    print(f"  - Extracted size: {extracted_size}")
    total_space = "~15 GB" if not remove_archive else extracted_size
    print(f"  - Total space needed: {total_space}")
    print("=" * 60)

    archive_path = os.path.join(data_dir, archive_name)

    # Download from Google Drive
    print(f"\nDownloading {data_name} data to {archive_path}...")
    try:
        download_from_gdrive(gdrive_id, archive_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download TCAV concept data: {e}\n"
            f"You can manually download from: "
            f"https://drive.google.com/file/d/{gdrive_id}/view "
            f"and save to: {archive_path}"
        )

    # Extract zip file - flatten to data_dir
    print(f"\nExtracting {archive_path} to {data_dir}...")
    try:
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                if filename:  # Skip directories
                    target_path = os.path.join(data_dir, filename)
                    with zip_ref.open(member) as source, open(
                        target_path, "wb"
                    ) as target:
                        target.write(source.read())
        print("Extraction completed.")
    except zipfile.BadZipFile as e:
        raise RuntimeError(
            f"Failed to extract archive: {e}\n"
            f"Please delete {archive_path} and try again."
        )

    # Remove archive file if requested
    if remove_archive and os.path.exists(archive_path):
        os.remove(archive_path)
        print(f"Removed archive file to save space: {archive_path}")

    # Create marker file to indicate successful download
    with open(marker_file, "w") as f:
        f.write("download_complete")

    print(f"\nTCAV concept data is ready at: {data_dir}")
    return data_dir
