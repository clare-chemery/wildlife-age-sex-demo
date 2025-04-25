import logging
from wildlifeml.io import get_io_args, load_data, save, load_model
from wildlifeml.train import evaluate_model, get_eval_args
from wildlifeml.utils import get_session_config


def main(model_filepath: str, test_filepath: str, eval_results_filepath: str):
    # Load data
    test_data = load_data(filepath=test_filepath)

    # Load model
    model_specs, model = load_model(filepath=model_filepath)

    # Evaluate model
    results = evaluate_model(model, test_data)

    # Save results
    save(model_specs | results, filepath=eval_results_filepath)


if __name__ == "__main__":
    session_config = get_session_config()
    logging.basicConfig(**session_config)

    io_args = get_io_args(session_config["working_dir"])
    eval_args = get_eval_args()
    main(**io_args, **eval_args)
