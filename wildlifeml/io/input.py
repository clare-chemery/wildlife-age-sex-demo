import pandas as pd
from pathlib import Path
from typing import Literal

import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential

AVAILABLE_MODELS = {
    "resnet50": {
        "model": tf.keras.applications.ResNet50V2,
        "preproc_func": tf.keras.applications.resnet_v2.preprocess_input,
    },
    "inception_resnet_v2": {
        "model": tf.keras.applications.InceptionResNetV2,
        "preproc_func": tf.keras.applications.inception_resnet_v2.preprocess_input,
    },
    "vgg19": {
        "model": tf.keras.applications.VGG19,
        "preproc_func": tf.keras.applications.vgg19.preprocess_input,
    },
    "xception": {
        "model": tf.keras.applications.Xception,
        "preproc_func": tf.keras.applications.xception.preprocess_input,
    },
    "densenet121": {
        "model": tf.keras.applications.DenseNet121,
        "preproc_func": tf.keras.applications.densenet.preprocess_input,
    },
    "densenet201": {
        "model": tf.keras.applications.DenseNet201,
        "preproc_func": tf.keras.applications.densenet.preprocess_input,
    },
}


def load(filepath: Path):
    """
    Load data from a parquet file or model from a keras file.
    """
    if filepath.suffix == ".parquet":
        return pd.read_parquet(filepath)
    elif filepath.suffix == ".keras":
        return tf.keras.models.load_model(filepath)
    else:
        raise ValueError(f"File {filepath} is not a parquet file or keras file.")


class ModelFactory:
    """Factory for creating Keras model objects."""

    @staticmethod
    def load(
        model_id: Literal[
            "resnet50",
            "inception_resnet_v2",
            "vgg19",
            "xception",
            "densenet121",
            "densenet201",
        ],
        num_classes: int,
        weights: str = "imagenet",
        include_top: bool = False,
        pooling: str = "avg",
    ) -> Sequential:
        """
        Return an initialized model instance from an identifier.

        Note: The model still needs to be compiled.
        """
        model_entry = AVAILABLE_MODELS[model_id]
        model_cls = model_entry["model"]

        model = Sequential()
        model.add(Lambda(model_entry["preproc_func"]))
        model.add(model_cls(weights=weights, include_top=include_top, pooling=pooling))
        model.add(Dense(num_classes, activation="softmax"))

        return model


def load_backbone_model(
    model: str | Path, num_classes: int = 2
) -> tf.keras.Model:
    """
    Load a backbone model from a string identifier or path.

    Args:
        model: Either a model name from AVAILABLE_MODELS or path to a model file
        working_dir: Working directory for resolving relative paths (only needed for string model names)
        num_classes: Number of output classes for the model

    Returns:
        tf.keras.Model: The loaded model
    """
    if model in AVAILABLE_MODELS.keys():
        return ModelFactory.load(
            model,
            num_classes=num_classes,
            weights="imagenet",
            include_top=False,
            pooling="avg",
        )
    else:
        try:
            return tf.keras.models.load_model(model)
        except FileNotFoundError:
            raise ValueError(
                f"Model not found. Please specify a valid model name from {AVAILABLE_MODELS.keys()} or provide a path to a valid model file [extensions]."
            )
