import logging
from wildlifeml.io import get_io_args, load_data, save
from wildlifeml.preprocess import get_preprocess_args, preprocess_data, split_data
from wildlifeml.utils import get_session_config


def main(
    raw_data_filepath: str,
    train_data_filepath: str,
    test_data_filepath: str,
):
    # Load data
    data = load_data(filepath=raw_data_filepath)

    # Preprocess data
    preprocessed_data = preprocess_data(data)

    # Split data
    train_data, test_data = split_data(preprocessed_data)

    save(train_data, filepath=train_data_filepath)
    save(test_data, filepath=test_data_filepath)


if __name__ == "__main__":
    session_config = get_session_config()
    logging.basicConfig(**session_config)

    io_args = get_io_args()
    preprocess_args = get_preprocess_args()

    main(**io_args, **preprocess_args)
