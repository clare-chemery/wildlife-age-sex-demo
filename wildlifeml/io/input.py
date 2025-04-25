import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential

AVAILABLE_MODELS = {
    'resnet50': {
        'model': tf.keras.applications.ResNet50V2,
        'preproc_func': tf.keras.applications.resnet_v2.preprocess_input,
    },
    'inception_resnet_v2': {
        'model': tf.keras.applications.InceptionResNetV2,
        'preproc_func': tf.keras.applications.inception_resnet_v2.preprocess_input,
    },
    'vgg19': {
        'model': tf.keras.applications.VGG19,
        'preproc_func': tf.keras.applications.vgg19.preprocess_input,
    },
    'xception': {
        'model': tf.keras.applications.Xception,
        'preproc_func': tf.keras.applications.xception.preprocess_input,
    },
    'densenet121': {
        'model': tf.keras.applications.DenseNet121,
        'preproc_func': tf.keras.applications.densenet.preprocess_input,
    },
    'densenet201': {
        'model': tf.keras.applications.DenseNet201,
        'preproc_func': tf.keras.applications.densenet.preprocess_input,
    },
}


class ModelFactory:
    """Factory for creating Keras model objects."""

    @staticmethod
    def load(
        model_id: str,
        num_classes: int,
        weights: str = 'imagenet',
        include_top: bool = False,
        pooling: str = 'avg',
    ) -> Sequential:
        """
        Return an initialized model instance from an identifier.

        Note: The model still needs to be compiled.
        """
        model_entry = AVAILABLE_MODELS[model_id]
        model_cls = model_entry['model']

        model = Sequential()
        model.add(Lambda(model_entry['preproc_func']))
        model.add(model_cls(weights=weights, include_top=include_top, pooling=pooling))
        model.add(Dense(num_classes, activation='softmax'))

        return model

def load_data(filename: str, working_dir: str):
    """
    Load data from a parquet file.
    """
    data_path = Path(working_dir) / "data" / Path(filename).with_suffix(".parquet")
    return pd.read_parquet(data_path)


def load_model(model: str, working_dir: str, num_classes: int = 2):
    """
    Load a model from a pickle file.
    """
    if model in AVAILABLE_MODELS.keys():
        model = ModelFactory.load(model, num_classes=num_classes, weights='imagenet', include_top=False, pooling='avg')
    else:
        try:
            model = tf.keras.models.load_model(Path(working_dir) / "models" / f"{model}.keras")
        except FileNotFoundError:
            raise ValueError(f"Model {model}.keras not found. Either specify a valid model name from {AVAILABLE_MODELS.keys()} or provide a path to a .keras file.")
    return model
