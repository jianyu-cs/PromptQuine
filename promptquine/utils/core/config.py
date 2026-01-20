from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig


def get_hydra_output_dir():
    return HydraConfig.get().run.dir


def compose_hydra_config_store(
    name: str, 
    configs: List[dataclass]
) -> ConfigStore:
    config_fields = []
    for config_cls in configs:
        for config_field in dataclasses.fields(config_cls):
            config_fields.append((config_field.name, config_field.type,
                                  config_field))
    Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
    cs = ConfigStore.instance()
    cs.store(name=name, node=Config)
    return cs
