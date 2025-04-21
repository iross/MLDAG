import yaml
import getpass
from pathlib import Path
import htcondor2 as htcondor
import typer
from typing_extensions import Annotated
from Resource import Resource
from htcondor_cli.annex import Create, Add
from htcondor_cli.annex_create import annex_name_exists
from htcondor_cli.cli import get_logger

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

# Batch name is batch_name=$(run_uuid)_$(request_gpus)g_$(request_cpus)c_$(request_memory)
# Doesn't really matter for annex name, but jotting it down here...

app = typer.Typer()

@app.command()
def create_annex(annex_name: Annotated[str, typer.Argument(help="The name of the annex to create (or add resources to).")], 
                 resource_name: Annotated[str, typer.Argument(help="The name of the remote site resource to add to the annex.")]):
    """
    Create or add to an HTCondor annex with the given name. Hooks directly into
    the htcondor CLI backend, with locally-defined configuration possible via a
    resources.yaml file.
    """
    resource = get_resource_from_yaml("resources.yaml", resource_name)
    tdict = dict(resource)
    _to_pop = []
    for key in tdict.keys():
        if key not in Create.options.keys():
            _to_pop.append(key)
    for key in _to_pop:
        tdict.pop(key)

    defaults = {}
    for key in Create.options.keys():
        if key not in tdict.keys():
            if 'default' in Create.options[key].keys():
                defaults[key] = Create.options[key]['default']
    tdict = tdict | defaults
    if annex_name_exists(annex_name):
        Add(get_logger(), annex_name=annex_name, **tdict)
    else:
        Create(get_logger(), annex_name=annex_name, **tdict)

if __name__ == "__main__":
    app()
