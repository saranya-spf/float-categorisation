import pandas as pd

from float_categorization.runners.categorization_runner_bert import (
    run_bert_pipeline,
    inference_bert_models,
    evaluate_bert_models,
)
from float_categorization.runners.categorization_runner_ensemble import (
    run_ensemble_pipeline,
    inference_ensemble_models,
    evaluate_ensemble_models,
)


def run_pipeline(
    file_path: str,
    use_deterministic: bool = False,
    save_model: bool = True,
    save_dir: str = "",
    use_ensemble: bool = False,
    Classifier=None,
    **kwargs,
) -> None:
    if use_ensemble:
        run_ensemble_pipeline(
            file_path, Classifier, use_deterministic, save_model, save_dir, **kwargs
        )
    else:
        run_bert_pipeline(file_path, use_deterministic, save_model, save_dir)


def run_inference(
    file_path: str,
    model_dict_path: str,
    use_deterministic: bool = False,
    use_ensemble: bool = False,
) -> pd.DataFrame:
    if use_ensemble:
        return inference_ensemble_models(file_path, model_dict_path, use_deterministic)
    else:
        return inference_bert_models(file_path, model_dict_path, use_deterministic)


def run_evaluate(
    file_path: str,
    model_dict_path: str,
    output_path: str,
    use_deterministic: bool = False,
    use_ensemble: bool = False,
) -> None:
    if use_ensemble:
        evaluate_ensemble_models(
            file_path, model_dict_path, output_path, use_deterministic
        )
    else:
        evaluate_bert_models(file_path, model_dict_path, output_path, use_deterministic)

