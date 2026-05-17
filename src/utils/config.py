"""
Configuration management using Hydra + OmegaConf.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str = "configs/pipeline_config.yaml") -> DictConfig:
    """Load and return config from YAML file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return OmegaConf.create(cfg)


def load_model_config(model_name: str) -> DictConfig:
    """Load model-specific config."""
    config_path = f"configs/model_configs/{model_name}.yaml"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"No config found for model '{model_name}' at {config_path}")
    return load_config(config_path)


def merge_configs(base: DictConfig, override: DictConfig) -> DictConfig:
    """Merge two configs, override takes precedence."""
    return OmegaConf.merge(base, override)


def resolve_env_vars(cfg: DictConfig) -> DictConfig:
    """Replace ${ENV_VAR} placeholders with actual environment variables."""
    cfg_str = OmegaConf.to_yaml(cfg)
    for key, value in os.environ.items():
        cfg_str = cfg_str.replace(f"${{{key}}}", value)
    return OmegaConf.create(yaml.safe_load(cfg_str))


def get_azure_config() -> Dict[str, str]:
    """Return Azure-specific config from environment."""
    return {
        "storage_account": os.getenv("ADLS_ACCOUNT_NAME", ""),
        "storage_key": os.getenv("ADLS_ACCOUNT_KEY", ""),
        "tenant_id": os.getenv("AZURE_TENANT_ID", ""),
        "client_id": os.getenv("AZURE_CLIENT_ID", ""),
        "client_secret": os.getenv("AZURE_CLIENT_SECRET", ""),
        "mlflow_tracking_uri": os.getenv(
            "MLFLOW_TRACKING_URI", "mlruns/"
        ),
        "databricks_host": os.getenv("DATABRICKS_HOST", ""),
        "databricks_token": os.getenv("DATABRICKS_TOKEN", ""),
    }
