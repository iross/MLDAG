#!/usr/bin/env python3

import argparse
import os
import textwrap
import htcondor
import random
from enum import Enum
from pydantic import BaseModel
from typing import Optional
import sys
import yaml
import uuid

SHUFFLE=False
EVAL=False

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
class Job(BaseModel):
    name: str # combination of epoch and type
    submit: str
    epoch: int
    run_uuid: str
    tr_id: int

    def get_submit_description(self, **template_vars) -> str:
        """
        Generate HTCondor submit description by filling in template variables.
        
        Args:
            **template_vars: Key-value pairs to substitute in the template
            
        Returns:
            Completed submit description string
        """
        return self.submit_template.format(**template_vars)

class TrainingRun(BaseModel):
    run_uuid: Optional[str] = None
    random_seed: int = random.randint(0, 1000000)
    epochs: int
    epochs_per_job: int
    # jobs: list[Job]
    resources: Optional[list[Resource]] = []

    def __init__(self, **data) -> None:
      super().__init__(**data)
      self.run_uuid = str(uuid.uuid4()).split("-")[0]

def get_resources() -> dict:
    """
    Usage: query the collector for a list of resources currently in the OSPool
    @return: dictionary whose keys are the names of all unique GLIDEIN_ResourceName s
             currently visible in the OSPool
    """
    print("Getting target resource list")
    collector = htcondor.Collector("cm-1.ospool.osg-htc.org")
    resources = collector.query(
        ad_type=htcondor.AdTypes.Startd,
        constraint="!isUndefined(GLIDEIN_ResourceName) && TotalGPUs > 0", #TODO: this is going to need to be constrained more.
        projection=["GLIDEIN_ResourceName"],
    )

    unique_resources = dict()

    # eliminate repeat resources to produce a unique list
    # using a dictionary to count number of occurrences, but this count is unused at the moment
    for resource in resources:
        if resource["GLIDEIN_ResourceName"] in unique_resources:
            unique_resources[resource["GLIDEIN_ResourceName"]] += 1
        else:
            unique_resources[resource["GLIDEIN_ResourceName"]] = 1

    print(f"{len(unique_resources)} resources found.")
    resources = [Resource(name=name, resource_type=ResourceType.OSPOOL) for name in unique_resources.keys()]
    return resources

def get_permutations(resources: list[Resource], config) -> list:
    """
    Usage: generate a list of permutations of resources
    @param resources: dictionary whose keys are the names of all unique GLIDEIN_ResourceName s
                      currently visible in the OSPool
    @param permutations: number of permutations to generate
    @return: list of lists of permutations of resources
    """
    permutations_list = []
    if config['epochs']/config['epochs_per_job'] > len(resources):
        print("Requested number of sites is incompatible than available. No re-use of sites is allowed (yet) so this experiment cannot be run. Exiting.")
        sys.exit(1)
    while len(permutations_list) < config['runs']:
        print("Generating training run")
        tr = TrainingRun(epochs=config['epochs'], epochs_per_job=config['epochs_per_job'])
        permutation = []
        while len(permutation) < config['epochs']/config['epochs_per_job']:
            print("...")
            resource = random.choice(resources)
            if resource not in permutation:
                permutation.append(resource)
        tr.resources += permutation
        permutations_list.append(tr)
    return permutations_list


def get_resource_names(yaml_path: str) -> list[str]:
    with open(yaml_path, 'r') as f:
        resource_defs = yaml.safe_load(f)
        return list(resource_defs.keys())

def get_resources_from_yaml(yaml_path: str) -> list[Resource]:
    resource_names = get_resource_names(yaml_path)
    return [get_resource_from_yaml(yaml_path, resource_name) for resource_name in resource_names]

def get_resource_from_yaml(yaml_path: str, resource_name: str) -> Resource:
    with open(yaml_path, 'r') as f:
        resource_defs = yaml.safe_load(f)[resource_name]
        resource_defs['name'] = resource_name
        return Resource(**resource_defs)

def get_vars(job: Job, resource: Resource, config: dict) -> str:
    return textwrap.dedent(f"""\
        VARS {job.name} epoch="{job.epoch}" run_uuid="{job.run_uuid}" ResourceName="{resource.name}" {'continue_from_checkpoint="true"' if job.tr_id > 0 else ""}
        {'VARS {job.eval_name} epoch="{job.epoch}" run_uuid="{job.training_run.run_uuid}" earlystop_marker_pathname="{job.training_run.run_prefix}.esm"' if EVAL else ''}""")


class EvaluationRun:
    def __init__(self):
        raise NotImplementedError("EvaluationRun is not implemented")


def get_submit_description(job: Job, resource: Resource, config: dict) -> str:
    # TODO this should be built based as needed from the job and resource. Need to think about how much gets done via VARS vs templating -- most probably winds up in VARS
    dag_txt = textwrap.dedent(f"""\
        SUBMIT-DESCRIPTION {resource.name}_pretrain.sub {{
                universe = container
                container_image = file:///staging/iaross/metl_global.sif

                request_disk = {resource.disk}
                request_memory = {resource.memory}
                request_cpus = {resource.cpu_count}
                request_gpus = {resource.gpu_count}
                gpus_minimum_memory = {resource.gpu_memory}
                gpus_minimum_capability = 7.5

                {f'TARGET.GLIDEIN_ResourceName == "{resource.name}"' if resource.resource_type == ResourceType.OSPOOL else ''}
                # TODO: annex name should be more flexible
                {f'MY.TargetAnnexName == "{resource.name}_annex"' if resource.resource_type == ResourceType.ANNEX else ''}

                {'environment = "WANDB_API_KEY='+str(config["wandb"]["api_key"])+'"' if "wandb" in config else ''}

                +is_resumable = true

                executable = /bin/bash
                transfer_executable = false
                arguments = pretrain.sh $(epoch) $(run_uuid)

                transfer_input_files = pretrain.sh, osdf:///ospool/ap40/data/ian.ross/processed-global.tar.gz
                if defined continue_from_checkpoint 
                    transfer_input_files = $(transfer_input_files), output/training_logs/$(run_uuid)
                endif
                transfer_output_files = output
                should_transfer_files = YES
                when_to_transfer_output = ON_EXIT_OR_EVICT

                output = $(CLUSTERID).out
                error = $(CLUSTERID).err
                log = metl.log

                queue
        }}
        SUBMIT-DESCRIPTION evaluate.sub {{
                universe = local
                executable = /bin/echo
                arguments = "TODO: Evaluation"
                queue
        }}
    """)
    return dag_txt

def get_initialization(run_prefix: str, sweep_config_name: str) -> tuple[str, str, str]:
    """Get the initialization jobs, vars and edges for a training run
    
    Args:
        run_prefix: Prefix for the run (e.g. 'run0')
        sweep_config_name: Name of the sweep config file
        
    Returns:
        Tuple of (jobs_txt, vars_txt, edges_txt) strings
    """
    jobs_txt = textwrap.dedent(f'''\
            JOB {run_prefix}-run_init run_init.sub
            JOB {run_prefix}-pproc pproc.sub
            JOB {run_prefix}-model_init model_init.sub\n''')
    
    vars_txt = textwrap.dedent(f'''\
            VARS {run_prefix}-run_init config_pathname="{sweep_config_name}" run_prefix="{run_prefix}" output_config_pathname="{run_prefix}-config.yaml"
            VARS {run_prefix}-pproc config_pathname="{run_prefix}-config.yaml" geld_pathname="ap2002_geld.json" output_tensor_pathname="{run_prefix}-ap2002.h5"
            VARS {run_prefix}-model_init config_pathname="{run_prefix}-config.yaml" output_model_pathname="{run_prefix}-model_init.pt"\n''')
    
    edges_txt = textwrap.dedent(f'''\
            PARENT sweep_init CHILD {run_prefix}-run_init
            PARENT {run_prefix}-run_init CHILD {run_prefix}-pproc {run_prefix}-model_init
            PARENT {run_prefix}-pproc {run_prefix}-model_init CHILD {run_prefix}-train_epoch0\n''')
            
    return "\n".join([jobs_txt, vars_txt, edges_txt])

# TODO: SUBMIT-DESCRIPTION for each job/resource combination of a TrainingRun,
# but with VARS to handle certain throughline variables? Or just stuff it all
# into some VARS?  The latter actually seems nice.. then we just have to create
# VARS lines for each job/resource combination. And we can do it "cleanly" by
# adding if statements within theh submit description

# TODO : oof, what does the submit workflow really look like when we're using a
# DAG? We need to specify annex name which might require `htcondor job create
# --annex-name`... Can a DAG do this for us (in a way more elegant than the node
# being a shell command)? Surely I can just set the annex name within the submit file?

def main(config):
    dag_txt = ''
    
    # TODO: This is set in the metl runtime options, so we'll  need to update that to pull it in from config or read it from the METL run config
    num_epoch = config['epochs']
    epochs_per_job = config['epochs_per_job']
    
    jobs_txt = ''
    vars_txt = ''
    edges_txt = ''

    # TODO: add flexibility in this structure?
    # Provisioning node
    # dag_txt += 'JOB sweep_init sweep_init.sub\n'
    # dag_txt += f'VARS sweep_init config_pathname="config.yaml" output_config_pathname="{sweep_config_name}"\n'

    resources = get_resources()

    for resource in resources:
        # TODO: one for OSPool and one for each annex.
        dag_txt += get_submit_description(None, resource, config)



    # Create resource permutations
    permutations = get_permutations(resources, config)
    i = 0
    for tr in permutations: # for each shishkabob
        print(tr)
        print(i)
        # Initialize the run
        run_prefix = f'run{i}'
        # dag_txt += get_initialization(run_prefix, sweep_config_name)

        for j, epoch in enumerate(range(epochs_per_job, num_epoch+1, epochs_per_job)): #gross hack
            resource = tr.resources[j]
            # input_model_postfix = 'init' if j == 0 else f'epoch{j-1}'
            job = Job(name=f'{run_prefix}-train_epoch{j}', 
                      submit=f"{resource.name}_pretrain.sub", epoch=epoch, 
                      run_uuid=tr.run_uuid, tr_id=j)

            jobs_txt += textwrap.dedent(f'''\
                    JOB {job.name} {job.submit}
                    {'JOB {job.eval_name} {job.eval_submit}' if EVAL else ''}''')
            vars_txt += get_vars(job, resource, config)

            # includes pre and post scripts for early stopping mechanism
            # TODO: why is earlystopdetector.py being called in both a pre and post?
            if EVAL:
                edges_txt += textwrap.dedent(f'''\
                        PARENT {run_prefix}-train_epoch{j} CHILD {run_prefix}-evaluate_epoch{j}
                        ''')

            # connect to successor train node
            if epoch < num_epoch:
            # if j < num_epoch/epoch - 1:
                edges_txt += f'\nPARENT {run_prefix}-train_epoch{j} CHILD {run_prefix}-train_epoch{j + 1}'

            # create newlines (pretty view)
            if j < num_epoch/epoch - 1:
                jobs_txt += '\n'
                vars_txt += '\n'
                edges_txt += '\n'

        dag_txt += '\n' + jobs_txt + '\n' + vars_txt + '\n' + edges_txt + '\n'        
        
        # flush out each shishkabob
        jobs_txt = ''
        vars_txt = ''
        edges_txt = ''
        i+=1

    # comparison node
    # TODO: define model comparison node
#     dag_txt += 'FINAL getbestmodel getbestmodel.sub\n'
#     dag_txt += 'VARS getbestmodel config_pathname="sweep.yaml"\n'
#     dag_txt += 'SCRIPT PRE getbestmodel prefinal.py sweep.yaml final_input_dir\n'

    # Cleanup
    # TODO: Any cleanup needed
    # dag_txt += f'SCRIPT POST getbestmodel cleanup.py {sweep_config_name}\n' 



    # misc directives
    dag_txt += '\nRETRY ALL_NODES 3\n'
    dag_txt += 'NODE_STATUS_FILE nodes.dag.status 30\n'

    with open('pipeline.dag', 'w') as f:
        f.write(dag_txt)
    print('generated pipeline.dag')


if __name__ == "__main__":
    # TODO: add options to visualize?
    # TODO: maybe flesh out some CLI options -- need to figure out what lives in config and what is pulled in at "submit" time
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    with open(args.config, 'r') as config:
        main(yaml.safe_load(config))
