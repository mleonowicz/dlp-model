"""Feature engineers the road sign dataset."""
import argparse
import logging
import shutil
import zipfile
from pathlib import Path

from urllib.parse import urlparse

import boto3

import mxnet as mx
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from constants import IMG_SIZE, VALIDATION_SPLIT

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def _extract_zip(src_path: Path, dst_path: Path):
    with zipfile.ZipFile(src_path, "r") as z:
        z.extractall(dst_path)


def recordio_dump(imgs_df: pd.DataFrame, src_dir: Path, dst_dir: Path):
    record = mx.recordio.MXRecordIO(uri=str(dst_dir / "imgs.rec"), flag="w")
    for idx, label, path in tqdm(imgs_df.itertuples(True, None)):
        img = Image.open(str(src_dir / path))
        img = img.resize(IMG_SIZE, Image.ANTIALIAS)
        header = mx.recordio.IRHeader(
            0, label, idx, 0
        )  # some very well (un)documented expected format
        packed_image = mx.recordio.pack_img(
            header, np.asarray(img), quality=100, img_fmt=".jpg"
        )
        record.write(packed_image)
    record.close()


if __name__ == "__main__":
    s3 = boto3.resource("s3")

    logger.debug("Starting preprocessing.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=urlparse, required=True)
    args = parser.parse_args()
    bucket, key = args.input_data.netloc, args.input_data.path.lstrip("/")

    base_dir = Path("/opt/ml/processing")
    temp_dir = base_dir / "temp"
    train_dir = base_dir / "train"
    validation_dir = base_dir / "validation"
    test_dir = base_dir / "test"
    for d in (temp_dir, train_dir, validation_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading data from bucket: %s, key: %s",
        bucket,
        key,
    )
    zip_path = temp_dir / "GTSRB.zip"
    s3.Bucket(bucket).download_file(key, str(zip_path))
    _extract_zip(zip_path, temp_dir)

    logger.debug("Reading data.")
    train_df = pd.read_csv(temp_dir / "Train.csv")
    test_df = pd.read_csv(temp_dir / "Test.csv")
    train_df, valid_df = train_test_split(
        train_df.loc[:, ["ClassId", "Path"]],
        test_size=VALIDATION_SPLIT,
        shuffle=True,
        random_state=42,
    )

    logger.info("Writing out datasets to %s.", base_dir)
    recordio_dump(train_df, temp_dir, train_dir)
    recordio_dump(valid_df, temp_dir, validation_dir)
    recordio_dump(test_df.loc[:, ["ClassId", "Path"]], temp_dir, test_dir)

    shutil.rmtree(temp_dir)
