import logging
from wildlifeml.io import get_io_args, load_data, save, load_model
from wildlifeml.train import get_training_args, tune_model
from wildlifeml.utils import get_session_config


def main(backbone_model: str, train_filepath: str, model_filepath: str):
    # Load data
    train_data = load_data(train_filepath)

    # Train model
    model = load_model(backbone_model)
    tuned_model = tune_model(model, train_data)

    # Save model
    save(tuned_model, model_filepath)


if __name__ == "__main__":
    session_config = get_session_config()
    logging.basicConfig(**session_config)

    io_args = get_io_args()
    training_args = get_training_args()

    main(**io_args, **training_args)
