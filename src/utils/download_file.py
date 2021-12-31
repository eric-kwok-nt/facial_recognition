import requests
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download(url: str, target_path: str, file_name: str) -> None:
    """Downloads file to a target folder and a specified filename

    Args:
        url (str): URL of the file
        target_path (str): Target folder for file to be downloaded to
        file_name (str): Specified filename
    """
    if not os.path.exists(target_path):
        logger.info(f"{target_path} folder not found. Creating...")
        os.makedirs(target_path)
    model_path = os.path.join(target_path, file_name)
    r = requests.get(url, stream=True)
    total = int(r.headers.get('content-length', 0))
    with open(model_path, 'wb') as f, tqdm(
            desc=f"Download {file_name} progress",
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            ) as bar:
            for data in r.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)