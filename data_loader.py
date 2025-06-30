import os
import zipfile
import subprocess

def download_data():
    os.makedirs("data", exist_ok=True)
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "dog-breed-identification",
        "-p", "data"
    ], check=True)

    with zipfile.ZipFile("data/dog-breed-identification.zip", "r") as zip_ref:
        zip_ref.extractall("data")
    os.remove("data/dog-breed-identification.zip")
    print("✅ Data downloaded and extracted to /data")

if __name__ == "__main__":
    download_data()
