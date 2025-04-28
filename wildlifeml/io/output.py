import pandas as pd
from typing import Union
import tensorflow as tf
from pathlib import Path

def save(content: Union[dict, str, pd.DataFrame], filepath: Path):
    """
    Save content to a file.
    """
    if isinstance(content, pd.DataFrame):
        content.to_parquet(filepath)
    elif isinstance(content, tf.keras.Model):
        content.save(filepath)
    else:
        content = format_dict_for_text(content) if isinstance(content, dict) else content
        with open(filepath, "w") as f:
            f.write(content)

def format_dict_for_text(model_specs: dict) -> str:
    pass
