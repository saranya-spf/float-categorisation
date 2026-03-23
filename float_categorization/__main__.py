import warnings
from pathlib import Path
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from float_categorization.core import run_pipeline, run_evaluate, run_inference

warnings.filterwarnings("ignore")
default_path = Path(__file__).parents[1]


if __name__ == "__main__":
    FILE_PATH_1 = default_path / "data" / "Copy of Float Sample Data - Sheet5.csv"
    FILE_PATH_2 = default_path / "data" / "Float Sample Data - Sheet2.csv"
    MODEL_SAVE_PATH = default_path / "models_dict" / "last_model.pt"
    OUTPUT_PATH = default_path / "analysis" / "evaluation_results.csv"
    
    #   XGBOOST:
    # run_pipeline(
    #     file_path=FILE_PATH_1,
    #     use_deterministic=False,
    #     save_model=False,
    #     save_dir="",
    #     use_ensemble=True,
    #     test_size = 0.3,
    #     Classifier=XGBClassifier,
    #     objective="multi:softprob",
    #     n_estimators=300,
    #     max_depth=6,
    #     learning_rate=0.1,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     min_child_weight=3,
    #     eval_metric="mlogloss",
    #     random_state=42,
    #     n_jobs=-1,
    # )

    ##  CATBOOST (Winner for now!!):
    run_pipeline(
        file_path=FILE_PATH_1,
        use_deterministic=True,
        save_model=False,
        save_dir="/Users/saranya.pal/Desktop/Projects/float_categorization/models_dict/ensemble_models",
        use_ensemble=True,
        test_size = 0.1,
        Classifier=CatBoostClassifier,
        # objective="multi:softprob",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        # subsample=0.8,
        # eval_metric="CrossEntropy",
        random_state=42,
    )
    
    
    # #   LIGHTGBM:
    # run_pipeline(
    #     file_path=FILE_PATH_1,
    #     use_deterministic=False,
    #     save_model=False,
    #     save_dir="",
    #     use_ensemble=True,
    #     test_size = 0.2,
    #     Classifier=LGBMClassifier,
    #     # objective="multi:softprob",
    #     n_estimators=300,
    #     max_depth=6,
    #     learning_rate=0.1,
    #     # subsample=0.8,
    #     # eval_metric="CrossEntropy",
    #     random_state=42,
    # )
    
    
    # #   RANDOM FOREST (Worst model till now):
    # run_pipeline(
    #     file_path=FILE_PATH_1,
    #     use_deterministic=False,
    #     save_model=False,
    #     save_dir="",
    #     use_ensemble=True,
    #     test_size = 0.2,
    #     Classifier=RandomForestClassifier,
    #     # objective="multi:softprob",
    #     n_estimators=300,
    #     max_depth=6,
    #     # subsample=0.8,
    #     # eval_metric="CrossEntropy",
    #     random_state=42,
    # )

    # run_evaluate(
    #     FILE_PATH_1, 
    #     "/Users/saranya.pal/Desktop/Projects/float_categorization/models_dict/ensemble_models/ensemble_model.joblib", 
    #     OUTPUT_PATH,
    #     use_deterministic=True,
    #     use_ensemble=True,
    # )
