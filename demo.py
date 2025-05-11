"""Demo script for the wildlifeml package."""

from pathlib import Path

import numpy as np
import pandas as pd

import wildlifeml

df = pd.read_json("test_data/data__red_deer_cropped/md_unlabeled.json").T
df.loc[:, "image_path"] = df.apply(
    lambda row: f"test_data/data__red_deer_cropped/full_images/{row['file'].split('/')[-1]}",
    axis=1,
)
# use indices for unique image_id, should not be a path!!
df.loc[:, "image_id"] = [
    i.split("/")[-1].split(".")[0] + str(idx) for idx, i in enumerate(df.index)
]
df["age"] = df["file"].apply(lambda x: x.split("/")[0].split("_")[0])
df["sex"] = df["file"].apply(lambda x: x.split("/")[0].split("_")[1])
df = df.drop(columns=["file", "category"]).reset_index(drop=True)

df["dummy_metadata"] = [np.random.randint(0, 5) for i in range(len(df))]
wildlifeml.save(df, Path("test_data/labeled_bbox_data.parquet"))

# df = wildlifeml.load(Path("test_data/test__unlabeled.parquet"))

# preprocessed_df = wildlifeml.preprocess.preprocess_data(df)
# wildlifeml.save(preprocessed_df, "test_data/test__preprocessed.parquet")

# for train, test in wildlifeml.preprocess.split_data(preprocessed_df, stratify_by="age"):
#     print(f"Train: {len(train)}")
#     print(train.head())
#     wildlifeml.save(train, "test_data/test__train.parquet")
#     print(f"Test: {len(test)}")
#     print(test.head())
#     ## TO DO: remove image column and save images in data path
#     wildlifeml.save(test, "test_data/test__test.parquet")
