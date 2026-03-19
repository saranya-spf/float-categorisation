import numpy as np
import scipy.sparse as sp


_NUM_CLASS_PARAM = {
    "XGBClassifier": "num_class",
    "LGBMClassifier": "num_class",
    "CatBoostClassifier": "classes_count",
}


class EnsembleClassifier:
    def __init__(self, Classifier, num_classes: int, **kwargs):
        cls_name = Classifier.__name__
        if cls_name in _NUM_CLASS_PARAM:
            kwargs[_NUM_CLASS_PARAM[cls_name]] = num_classes
        self.classifier = Classifier(**kwargs)

    def fit(self, X: sp.csr_matrix, y: np.ndarray):
        self.classifier.fit(X, y)

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        return self.classifier.predict(X)

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        return self.classifier.predict_proba(X)
