from typing import Optional
from pydantic import BaseModel
from enum import Enum
import yaml

class ResourceType(Enum):
    OSPOOL = 1
    ANNEX = 2

class Resource(BaseModel):
    name: Optional[str] = None
    # annex_name: Optional[str] = None
    queue_at_system: Optional[str] = None
    login_name: Optional[str] = None
    allocation: Optional[str] = None
    disk: int = "5GB"
    mem_mb: int = None
    cpus: int = None
    gpus: str = "1"
    lifetime: int = 172800
    gpu_type: Optional[str] = None
    gpu_memory: int = 8192
    two_factor_auth: bool = False
    login_node: Optional[str] = None
    resource_type: ResourceType = ResourceType.OSPOOL

def get_resource_names(yaml_path: str = "resources.yaml") -> list[str]:
    with open(yaml_path, 'r') as f:
        resource_defs = yaml.safe_load(f)
        return list(resource_defs.keys())

def get_resources_from_yaml(yaml_path: str = "resources.yaml") -> list[Resource]:
    resource_names = get_resource_names(yaml_path)
    return [get_resource_from_yaml(yaml_path, resource_name) for resource_name in resource_names]

def get_resource_from_yaml(yaml_path: str = "resources.yaml", resource_name: str = None) -> Resource:
    with open(yaml_path, 'r') as f:
        resource_defs = yaml.safe_load(f)[resource_name]
        resource_defs['queue_at_system'] = f"{resource_defs['queue']}@{resource_name}"
        return Resource(**resource_defs)
