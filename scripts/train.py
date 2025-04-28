import logging
from wildlifeml.io import load, save, load_backbone_model
from wildlifeml.train import tune_model
import argparse
import tomli
from pathlib import Path

def main(working_dir: str, train_filepath: str, backbone_model: str, model_dir: str, training_args: dict, **kwargs):
    # Load data
    train_data = load(filepath=Path(working_dir) / Path(train_filepath))

    # Train model
    model = load_backbone_model(backbone_model, num_classes=training_args.get("num_classes", 2))
    tuned_model, tuning_specs = tune_model(model, train_data, **training_args)

    # Save model
    save(tuning_specs, filepath=Path(working_dir) / Path(model_dir) / "tuning_specs.toml")
    save(tuned_model, filepath=Path(working_dir) / Path(model_dir) / "model.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test__config.toml")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        args = tomli.load(f)
    logging.basicConfig(**args["logging"])

    main(args["globals"]["working_dir"], **args["io"]["data"], **args["io"]["model"], training_args=args["train"])
