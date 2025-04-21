from typing import Optional
from pydantic import BaseModel
from enum import Enum

class ResourceType(Enum):
    OSPOOL = 1
    ANNEX = 2

class Resource(BaseModel):
    name: str
    username: Optional[str] = None
    disk: int = "5GB"
    memory: int = "32GB"
    cpu_count: int = 1
    gpu_count: int = 1
    gpu_memory: int = 8192
    two_factor_auth: bool = False
    login_node: Optional[str] = None
    resource_type: ResourceType = ResourceType.OSPOOL