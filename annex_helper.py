import yaml
import getpass
from pathlib import Path
import htcondor2 as htcondor
from Resource import Resource
from htcondor_cli.annex import Create
from htcondor_cli.cli import get_logger

def get_resource_names(yaml_path: str) -> list[str]:
    with open(yaml_path, 'r') as f:
        resource_defs = yaml.safe_load(f)
        return list(resource_defs.keys())

def get_resources_from_yaml(yaml_path: str) -> list[Resource]:
    resource_names = get_resource_names(yaml_path)
    return [get_resource_from_yaml(yaml_path, resource_name) for resource_name in resource_names]

def get_resource_from_yaml(yaml_path: str, resource_name: str) -> Resource:
    # annex_name
    # queue_at_system
    # lifetime
    # allocation
    # login_name
    # gpus
    # lifetime
    # mem_mb
    # cpus
    with open(yaml_path, 'r') as f:
        resource_defs = yaml.safe_load(f)[resource_name]
        # resource_defs['name'] = resource_name
        resource_defs['queue_at_system'] = f"{resource_defs['queue']}@{resource_name}"
        return Resource(**resource_defs)

# Batch name is batch_name=$(run_uuid)_$(request_gpus)g_$(request_cpus)c_$(request_memory)
# Doesn't really matter for annex name, but jotting it down here...

# Can probably grab from the annex CLI:
# https://github.com/htcondor/htcondor/blob/a2d2d2b11f47d318b29dca0ca65b49a6da058cad/src/condor_tools/htcondor_cli/annex.py


# Create(**options)
# with options defined at https://github.com/htcondor/htcondor/blob/a2d2d2b11f47d318b29dca0ca65b49a6da058cad/src/condor_tools/htcondor_cli/annex.py#L24
# CLI arguments I usually use:
# htcondor annex create expanse_global_2025-04-18_2 gpu-shared@expanse --project ddp468  \
# --login-name iaross --gpus 2  --lifetime 172800 --mem_mb 128000 --cpus 2

if __name__ == "__main__":
    resource = get_resource_from_yaml("resources.yaml", "expanse")
    tdict = dict(resource)

    _to_pop = []
    for key in tdict.keys():
        if key not in Create.options.keys():
            _to_pop.append(key)

    for key in _to_pop:
        tdict.pop(key)

    tdict = tdict | {
        # TODO: grab defaults from Create.options
        "nodes": 1,
        "owners": getpass.getuser(),
        "collector" : htcondor.param.get("ANNEX_COLLECTOR", "htcondor-cm-hpcannex.osgdev.chtc.io"),
        "token_file" : None,
        "password_file": Path(htcondor.param.get("ANNEX_PASSWORD_FILE", "~/.condor/annex_password_file")),
        "control_path": Path(htcondor.param.get("ANNEX_TMP_DIR", "~/.hpc-annex")),
        "login_host": None,
        "startd_noclaim_shutdown": 300,
        "gpu_type": None,
        "test": None
    }
    Create(get_logger(), annex_name="test", **tdict)

