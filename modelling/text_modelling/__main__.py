import warnings
from pathlib import Path

from modelling.text_modelling.core import (
    run_pipeline, 
    evaluate, 
    fine_tune_model
)

warnings.filterwarnings("ignore")
default_path = Path(__file__).parents[2] / "data"


if __name__ == "__main__":
    FILE_PATH_1 = default_path / "Copy of Float Sample Data - Sheet5.csv"
    FILE_PATH_2 = default_path / "Float Sample Data - Sheet2.csv"

#     # run_pipeline(FILE_PATH_1)

    evaluate(FILE_PATH_1)
