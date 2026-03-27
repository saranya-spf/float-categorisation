import numpy as np
import scipy.sparse as sp


_NUM_CLASS_PARAM = {
    "XGBClassifier": "num_class",
    "LGBMClassifier": "num_class",
    "CatBoostClassifier": "classes_count",
}

# Maps classifier name -> the kwarg name for class weights
_CLASS_WEIGHT_PARAM = {
    "CatBoostClassifier": "class_weights",
    "XGBClassifier": "sample_weight",  # handled via fit()
    "LGBMClassifier": "class_weight",
    "RandomForestClassifier": "class_weight",
}


class EnsembleClassifier:
    def __init__(self, Classifier, num_classes: int, class_weights=None, **kwargs):
        cls_name = Classifier.__name__
        if cls_name in _NUM_CLASS_PARAM:
            kwargs[_NUM_CLASS_PARAM[cls_name]] = num_classes

        self._cls_name = cls_name
        self._class_weights = class_weights

        # Pass class weights as constructor param for classifiers that support it
        if class_weights and cls_name in _CLASS_WEIGHT_PARAM:
            weight_param = _CLASS_WEIGHT_PARAM[cls_name]
            if cls_name == "CatBoostClassifier":
                # CatBoost expects a list ordered by class index
                max_class = max(class_weights.keys())
                kwargs[weight_param] = [
                    class_weights.get(i, 1.0) for i in range(max_class + 1)
                ]
            elif cls_name != "XGBClassifier":  # XGB uses sample_weight in fit()
                kwargs[weight_param] = class_weights

        self.classifier = Classifier(**kwargs)

    def fit(self, X: sp.csr_matrix, y: np.ndarray):
        # XGBoost doesn't take class_weight in constructor; use sample_weight in fit
        if self._cls_name == "XGBClassifier" and self._class_weights:
            sample_w = np.array(
                [self._class_weights.get(int(label), 1.0) for label in y]
            )
            self.classifier.fit(X, y, sample_weight=sample_w)
            return
        self.classifier.fit(X, y)

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        return self.classifier.predict(X)

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        return self.classifier.predict_proba(X)

    def get_feature_importance(self, feature_names=None):
        feature_importance = self.classifier.get_feature_importance()
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
        paired = sorted(
            zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True
        )
        print("\n--- Feature Importance (descending) ---")
        for name, importance in paired:
            if importance > 0:
                print(f"  {name}: {importance:.2f}")
