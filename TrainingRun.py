from pydantic import BaseModel
from typing import Optional
import random
import uuid
import yaml
from Resource import Resource

def read_config(config: str = "TrainingRun.yaml") -> None:
    """
    Read the config yaml file and set the attributes of the TrainingRun instance.
    """

    with open(config, "r") as f:
        config = yaml.safe_load(f)
    # Get the list of attributes for TrainingRun class
    training_run_attrs = TrainingRun.__annotations__.keys()
    
    for key, value in config["vars"].items():
        if key in training_run_attrs:
            config[key] = value['value']
        # TODO: parse special types of config options and resolve them within vars attribute
        if value['type'] == 'value':
            config['vars'][key] = value['value']
        elif value['type'] == 'function':
            config['vars'][key] = "TODO"
        elif value['type'] == 'range':
            config['vars'][key] = list(range(value['start'] if 'start' in value else 0, value['stop'], value['step'] if 'step' in value else 1))

        
    return TrainingRun(**config)

class TrainingRun(BaseModel):
    submit_template: str
    vars: dict
    # TODO: stuff these in vars
    resources: Optional[list[Resource]] = []
    run_uuid: Optional[str] = None
    random_seed: int = random.randint(0, 1000000)
    epochs: int
    epochs_per_job: int

    def __init__(self, **data) -> None:
      super().__init__(**data)
      self.run_uuid = str(uuid.uuid4()).split("-")[0]

    def generate_parameter_combinations(self):
        """ 
        Get all the combinations of the vars dictionary and return a list of dict holding the values for each combination.
        """
        # TODO: probably shouldn't be a method on the TrainingRun class -- makes more sense to _generate_ TrainingRun instances
        var_names = list(self.vars.keys())
        value_lists = [self.vars[var] if isinstance(self.vars[var], list) else [self.vars[var]] for var in var_names]
        
        # Use itertools.product to generate all combinations
        from itertools import product
        for values in product(*value_lists):
            # Create dict mapping var names to values for this combination
            var_dict = dict(zip(var_names, values))
            print(var_dict)
            # yield TrainingRun(**var_dict)


