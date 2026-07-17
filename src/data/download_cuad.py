"""Download the CUAD v1 dataset into data/raw/.

CUAD is distributed as a single zip archive (data.zip) from the Atticus
Project's GitHub repo. It bundles CUADv1.json - a SQuAD-style JSON export
where each contract is one "paragraph" with 41 questions (one per clause
category), each optionally answered by a span of the contract text - plus
the source contract PDFs/TXTs. We only need CUADv1.json for Week 1; it's
extracted and saved as CUAD_v1.json under data/raw/.

See: https://github.com/TheAtticusProject/cuad
"""

import logging
import urllib.request
import zipfile
from urllib.error import URLError

from src.common.config import (
    CUAD_RAW_JSON,
    CUAD_RAW_ZIP,
    CUAD_ZIP_INNER_JSON_NAME,
    CUAD_ZIP_URL,
    RAW_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _download_zip(force: bool) -> None:
    if CUAD_RAW_ZIP.exists() and not force:
        logger.info("%s already downloaded (use force=True to re-download)", CUAD_RAW_ZIP)
        return

    logger.info("Downloading CUAD dataset archive from %s", CUAD_ZIP_URL)
    try:
        urllib.request.urlretrieve(CUAD_ZIP_URL, CUAD_RAW_ZIP)
    except URLError as exc:
        raise RuntimeError(
            f"Failed to download CUAD dataset from {CUAD_ZIP_URL}. "
            "Check your network connection, or download data.zip manually from "
            "https://github.com/TheAtticusProject/cuad and place it at "
            f"{CUAD_RAW_ZIP}"
        ) from exc


def _extract_json() -> None:
    logger.info("Extracting %s from %s", CUAD_ZIP_INNER_JSON_NAME, CUAD_RAW_ZIP)
    with zipfile.ZipFile(CUAD_RAW_ZIP) as zf:
        with zf.open(CUAD_ZIP_INNER_JSON_NAME) as src, open(CUAD_RAW_JSON, "wb") as dst:
            dst.write(src.read())


def download_cuad(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if CUAD_RAW_JSON.exists() and not force:
        logger.info("CUAD_v1.json already present at %s (use force=True to re-download)", CUAD_RAW_JSON)
        return

    _download_zip(force)
    _extract_json()
    logger.info("Saved CUAD dataset to %s", CUAD_RAW_JSON)


if __name__ == "__main__":
    download_cuad()
