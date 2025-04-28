from pydantic import BaseModel
from typing import Optional
import random
import uuid
from Resource import Resource

class TrainingRun(BaseModel):
    resources: Optional[list[Resource]] = []
    run_uuid: Optional[str] = None
    random_seed: Optional[int] = None
    epochs: int
    epochs_per_job: int
    vars: Optional[dict] = {}

    def __init__(self, **data) -> None:
      super().__init__(**data)
      self.run_uuid = str(uuid.uuid4()).split("-")[0]
      self.random_seed = random.randint(0, 1000000)
