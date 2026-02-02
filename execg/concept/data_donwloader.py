import os
import subprocess


def download_physionet_challenge(challenge_year: str = "2021", output_dir: str = "./data") -> None:
    """Download PhysioNet Challenge dataset using wget.

    Args:
        challenge_year: Challenge year (default: "2021").
        output_dir: Directory to save the downloaded data.

    Raises:
        ValueError: If challenge_year is not supported.
        AssertionError: If target path already exists.
    """
    version_mapping = {"2021": "1.0.3"}

    try:
        version = version_mapping[challenge_year]
    except KeyError:
        raise ValueError(
            f"not supported challenge year: you can choose from {version_mapping.keys()}"
        )

    os.makedirs(output_dir, exist_ok=True)

    target_path = os.path.join(
        output_dir, f"physionet.org/files/challenge-{challenge_year}/{version}"
    )
    if os.path.exists(target_path):
        assert (
            False
        ), f"file already exists in {target_path}. please remove the file and try again."

    url = f"https://physionet.org/files/challenge-{challenge_year}/{version}/training/"

    os.chdir(output_dir)

    cmd = f"wget -r -N -c -np {url}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)

    print(f"Download completed. Data saved to {output_dir}/physionet.org/")


if __name__ == "__main__":
    download_physionet_challenge(output_dir="./tmp")
