# api to install dataset kaggle
import random

import kaggle
from pathlib import Path
import shutil


def installDataset():
    kaggle.api.authenticate()  # file kaggle json già settato
    kaggle.api.dataset_download_files("xhlulu/140k-real-and-fake-faces", path=".", unzip=True)


def loadDataset(numImages):
    fake_dir = Path("real_vs_fake/real-vs-fake/test/fake")
    real_dir = Path("real_vs_fake/real-vs-fake/test/real")
    half = numImages // 2
    fake_images = list(fake_dir.glob("*.jpg"))[:half]
    real_images = list(real_dir.glob("*.jpg"))[:half]
    images_with_labels = [(img, 1) for img in real_images] + [(img, 0) for img in fake_images]
    return images_with_labels, len(fake_images), len(real_images)


def shuffleDataset(dataset):
    random.shuffle(dataset)


def saveDataset(dataset, name):
    base_dir = Path(name)
    real_dir = base_dir / "real"
    fake_dir = base_dir / "fake"

    # Crea le directory se non esistono
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Salva le immagini nelle rispettive cartelle
    for i, (img_path, label) in enumerate(dataset):
        if label == 1:
            dest = real_dir / f"real_{i}{img_path.suffix}"
        else:
            dest = fake_dir / f"fake_{i}{img_path.suffix}"

        shutil.copy(img_path, dest)


def loadExistingDataset(folder_name):
    base_dir = Path(folder_name)
    real_dir = base_dir / "real"
    fake_dir = base_dir / "fake"

    # Verifica che le directory esistano
    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(
            f"La cartella '{folder_name}' non è strutturata correttamente (manca 'real/' o 'fake/')")

    # Carica le immagini con etichette
    real_images = [(img_path, 1) for img_path in real_dir.glob("*.jpg")]
    fake_images = [(img_path, 0) for img_path in fake_dir.glob("*.jpg")]

    dataset = real_images + fake_images
    return dataset, len(fake_images), len(real_images)


if __name__ == "__main__":
    installDataset()

