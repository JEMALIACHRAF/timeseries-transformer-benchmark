"""
Model registry — factory pattern for dynamic model instantiation.
"""
from typing import Any, Dict, Type

from src.models.baselines.prophet import ProphetForecaster
from src.models.baselines.xgboost_ts import XGBoostForecaster
from src.models.deep.nbeats_nhits import NBeatsForecaster, NHitsForecaster
from src.models.deep.patchtst import PatchTSTForecaster
from src.models.deep.tft import TFTForecaster
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Registry mapping string name → class
MODEL_REGISTRY: Dict[str, Type] = {
    "xgboost": XGBoostForecaster,
    "prophet": ProphetForecaster,
    "nbeats": NBeatsForecaster,
    "nhits": NHitsForecaster,
    "tft": TFTForecaster,
    "patchtst": PatchTSTForecaster,
}


def get_model(model_name: str, **kwargs) -> Any:
    """
    Instantiate a model by name with optional hyperparameter overrides.

    Args:
        model_name: One of ['xgboost', 'prophet', 'nbeats', 'nhits', 'tft', 'patchtst']
        **kwargs: Hyperparameters passed to the model constructor.

    Returns:
        Instantiated model object.

    Raises:
        ValueError: If model_name is not in registry.
    """
    name = model_name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    cls = MODEL_REGISTRY[name]
    model = cls(**kwargs)
    logger.info(f"Instantiated model: {name} ({cls.__name__})")
    return model


def list_models() -> list:
    """Return list of all available model names."""
    return list(MODEL_REGISTRY.keys())
