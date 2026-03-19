import warnings
from pathlib import Path

from float_categorization.core import run_pipeline, run_evaluate, run_inference
warnings.filterwarnings("ignore")
default_path = Path(__file__).parents[2]


if __name__ == "__main__":
    FILE_PATH_1 = default_path / "data" / "Copy of Float Sample Data - Sheet5.csv"
    FILE_PATH_2 = default_path / "data" / "Float Sample Data - Sheet2.csv"
    MODEL_SAVE_PATH = default_path / "models_dict" / "last_model.pt"
    OUTPUT_PATH = default_path / "analysis" / "evaluation_results.csv"
    # run_pipeline(FILE_PATH_1)
    run_evaluate(FILE_PATH_1, MODEL_SAVE_PATH, OUTPUT_PATH)
