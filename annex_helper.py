import os
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

def scan_for_requests(request_path: str = "./") -> list[str]:
    # Scan the request_path directory for files with the .request extension.
    # The filename is the desired annex name.
    # The contents of the file contain the target resource name
    requests = []
    for file in os.listdir(request_path):
        if file.endswith('.request'):
            target_resource = open(file).read().strip()
            if target_resource not in get_resource_names():
                print(f"Resource {target_resource} not found in resources.yaml")
                continue
            requests.append((file.replace(".request", ""), target_resource))
    return requests

def create_from_requests(request_path: str = "./"):
    reqs = scan_for_requests(request_path)
    for request in reqs:
        check = create_annex(request[0], request[1])
        if check != 0:
            print(f"Failed to create annex {request[0]}")
        else:
            os.remove(f"{request_path}/{request[0]}.request")

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
        # TODO: We don't get any kind of success coming back from the class init 😢
        print(f"Annex {annex_name} already exists. Adding resources to it.")
        Add(get_logger(), annex_name=annex_name, **tdict)
    else:
        print(f"Creating new annex {annex_name}.")
        Create(get_logger(), annex_name=annex_name, **tdict)
    return 0

if __name__ == "__main__":
    app()
