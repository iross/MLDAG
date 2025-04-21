from typing import Optional
from pydantic import BaseModel
from enum import Enum

class ResourceType(Enum):
    OSPOOL = 1
    ANNEX = 2

class Resource(BaseModel):
    # name: Optional[str] = None
    # annex_name: Optional[str] = None
    queue_at_system: Optional[str] = None
    login_name: Optional[str] = None
    allocation: Optional[str] = None
    # disk: int = "5GB"
    mem_mb: int = None
    cpus: int = None
    gpus: str = "1"
    lifetime: int = 172800
    gpu_type: Optional[str] = None
    # gpu_memory: int = 8192
    two_factor_auth: bool = False
    login_node: Optional[str] = None
    resource_type: ResourceType = ResourceType.OSPOOL