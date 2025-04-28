from pydantic import BaseModel
from typing import Optional
import random
import yaml
from TrainingRun import TrainingRun
from Resource import Resource

def read_from_config(config: str = "Experiment.yaml") -> None:
    """
    Read the config yaml file and set the attributes of the TrainingRun instance.
    """

    with open(config, "r") as f:
        config = yaml.safe_load(f)
    # Get the list of attributes for TrainingRun class
    _experiment_attrs = Experiment.__annotations__.keys()
    
    for key, value in config["vars"].items():
        if key in _experiment_attrs:
            config[key] = value['value']
        # TODO: parse special types of config options and resolve them within vars attribute
        if value['type'] == 'value':
            config['vars'][key] = value['value']
        elif value['type'] == 'function':
            config['vars'][key] = "TODO"
        elif value['type'] == 'range':
            config['vars'][key] = list(range(value['start'] if 'start' in value else 0, value['stop'], value['step'] if 'step' in value else 1))

    return Experiment(**config)
class Experiment(BaseModel):
    name: Optional[str] = ""
    training_runs: Optional[list[TrainingRun]] = []
    submit_template: str
    vars: dict
    
    def __init__(self, **data) -> None:
      super().__init__(**data)

    def _add_var_permutations(self):
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
            self.training_runs.append(TrainingRun(**var_dict))

    def _add_resource_permutations(self, resources: list[Resource]) -> list[TrainingRun]:
        """
        Usage: generate a list of permutations of resources
        @param resources: dictionary whose keys are the names of all unique GLIDEIN_ResourceName s
                        currently visible in the OSPool
        @param permutations: number of permutations to generate
        @return: list of lists of permutations of resources
        """
        permutations_list = []
        if self.vars['epochs']/self.vars['epochs_per_job'] > len(resources):
            print("WARNING: Requested number of sites is less than available sites. Sites will be reused.")
            # sys.exit(1)
        while len(permutations_list) < len(self.vars['run_number']):
            print("Generating training run")
            # TODO: these will read from a config.
            tr = TrainingRun(epochs=self.vars['epochs'], epochs_per_job=self.vars['epochs_per_job'])
            permutation = []
            random.shuffle(resources)
            while len(permutation) < self.vars['epochs']/self.vars['epochs_per_job']:
                # make sure that each resource appears once before any are repeated
                if len(permutation) < len(resources):
                    permutation.append(resources[len(permutation)])
                else:
                    resource = random.choice(resources)
                    permutation.append(resource)
            tr.resources += permutation
            permutations_list.append(tr)
        self.training_runs += permutations_list
