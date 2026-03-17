import warnings
from pathlib import Path

from float_categorization.modelling.core import (
    run_pipeline, 
    evaluate, 
    fine_tune_model
)

warnings.filterwarnings("ignore")
default_path = Path(__file__).parents[2]


if __name__ == "__main__":
    FILE_PATH_1 = default_path / "data" / "Copy of Float Sample Data - Sheet5.csv"
    FILE_PATH_2 = default_path / "data" / "Float Sample Data - Sheet2.csv"
    MODEL_SAVE_PATH = default_path / "models_dict" / "last_model.pt"
    OUTPUT_PATH = default_path / "analysis" / "evaluation_results.csv"
    # run_pipeline(FILE_PATH_1)

    evaluate(
        FILE_PATH_1,
        MODEL_SAVE_PATH,
        OUTPUT_PATH
    )
