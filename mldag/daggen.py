#!/usr/bin/env python3

import os
import sys
import typer
from typing_extensions import Annotated
import textwrap
import htcondor
import random
from pydantic import BaseModel
import yaml
from mldag.models.resource import Resource, ResourceType, get_resources_from_yaml
from mldag.models.training_run import TrainingRun
from mldag.models.experiment import Experiment, read_from_config

EVAL=False

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


def get_ospool_resources() -> dict:
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


def get_vars(job: Job, resource: Resource, training_run: TrainingRun) -> str:
    # {'VARS {job.eval_name} epoch="{job.epoch}" run_uuid="{job.training_run.run_uuid}" earlystop_marker_pathname="{job.training_run.run_prefix}.esm"' if EVAL else ''}
    vars_txt = textwrap.dedent(f"""\
        VARS {job.name} epoch="{job.epoch}" run_uuid="{job.run_uuid}" random_seed="{training_run.random_seed}" ResourceName="{resource.name}" {'continue_from_checkpoint="true"' if job.tr_id > 0 else ""}
        """)
    if len(training_run.vars.items()) > 0:
        vars_txt += textwrap.dedent(f"""\
        VARS {job.name} {" ".join([f'{key}="{str(value)}"' for key, value in training_run.vars.items()])}
        """)
    return vars_txt

def get_script(job: Job, resource: Resource, config: dict) -> str:
    # Use the exact Python that ran mldag-gen so the script works regardless
    # of PATH in the DAGMan environment.  VARS macros ($(run_uuid)) are not
    # available in SCRIPT args, so run_uuid and epoch are embedded here.
    python = sys.executable
    pre_args = f'{job.run_uuid} {job.name} {job.epoch}'
    if resource.resource_type == ResourceType.ANNEX and resource.annex:
        # --annex tells pre.py to chain pre_request_annex.sh via subprocess.
        # DAGMan allows only one SCRIPT PRE per node.
        pre_args += f' --annex {resource.name}'
    script_txt = f'SCRIPT PRE  {job.name} {python} -m mldag.provenance.pre {pre_args}\n'
    script_txt += f'SCRIPT POST {job.name} {python} -m mldag.provenance.post $JOB $RETURN $JOBID\n'
    return script_txt

def get_service(python_exe: str = "python3", dag_name: str = "") -> str:
    dagman_log_arg = f" --dagman-log {dag_name}.dagman.out" if dag_name else ""
    service_txt = textwrap.dedent(f"""\
    SUBMIT-DESCRIPTION provenance_monitor.sub {{
        universe = local
        executable = {python_exe}
        arguments = -m mldag.provenance.log_monitor --log-file metl.log --classad-dir output/provenance{dagman_log_arg}
        queue
    }}
    SERVICE provenance_monitor provenance_monitor.sub
    """)
    return service_txt

class EvaluationRun:
    def __init__(self):
        raise NotImplementedError("EvaluationRun is not implemented")


def get_submit_description(job: Job, resource: Resource, config: dict, experiment: Experiment) -> str:
    inner_txt = experiment.submit_template.format(resource = resource)

    # Hacky. Fix this.
    if "queue" in inner_txt and inner_txt.strip().endswith("queue"):
        inner_txt = inner_txt.strip().rstrip("queue")
    if resource.resource_type == ResourceType.OSPOOL:
        inner_txt += f'TARGET.GLIDEIN_ResourceName == "{resource.name}"\n'
    elif resource.resource_type == ResourceType.ANNEX and resource.annex:
        inner_txt += f'MY.TargetAnnexName = "{resource.name}_annex"\n'
    env_vars = ["PROVENANCE_RUN_ID=$(run_uuid)"]
    if "wandb" in config:
        env_vars.append(f"WANDB_API_KEY={config['wandb']['api_key']}")
    inner_txt += f'environment = "{" ".join(env_vars)}"\n'
    inner_txt += 'job_ad_file = output/provenance/$(ClusterId).ad\n'
    inner_txt += 'queue\n'

    inner_txt = textwrap.indent(inner_txt, "\t")

    dag_txt = f"SUBMIT-DESCRIPTION {resource.name}_pretrain.sub {{"
    dag_txt += f"\n{inner_txt}"
    dag_txt += "}\n"
    return dag_txt

def get_ospool_submit_description(config: dict, experiment: Experiment) -> str:
    """Create a shared submit description for OSPool resources"""
    inner_txt = experiment.submit_template.format(resource=Resource(name="ospool", resource_type=ResourceType.OSPOOL))

    # Hacky. Fix this.
    if "queue" in inner_txt and inner_txt.strip().endswith("queue"):
        inner_txt = inner_txt.strip().rstrip("queue")

    # OSPool resources will use TARGET.GLIDEIN_ResourceName variable instead of hardcoding
    inner_txt += 'TARGET.GLIDEIN_ResourceName == "$(ResourceName)"\n'

    env_vars = ["PROVENANCE_RUN_ID=$(run_uuid)"]
    if "wandb" in config:
        env_vars.append(f"WANDB_API_KEY={config['wandb']['api_key']}")
    inner_txt += f'environment = "{" ".join(env_vars)}"\n'
    inner_txt += 'job_ad_file = output/provenance/$(ClusterId).ad\n'
    inner_txt += 'queue\n'

    inner_txt = textwrap.indent(inner_txt, "\t")

    dag_txt = "SUBMIT-DESCRIPTION ospool_pretrain.sub {"
    dag_txt += f"\n{inner_txt}"
    dag_txt += "}\n"
    return dag_txt

app = typer.Typer()
@app.command()
def main(config: Annotated[str, typer.Argument(help="Path to YAML config file")] = 'config.yaml'):
    config = yaml.safe_load(open(config, 'r'))
    experiment = read_from_config("Experiment.yaml")
    ename = experiment.name.replace(' ', '_').lower()
    dag_txt = ''

    num_epoch = experiment.vars['epochs']
    epochs_per_job = experiment.vars['epochs_per_job']

    jobs_txt = ''
    vars_txt = ''
    edges_txt = ''
    script_txt = ''

    # Provisioning node
    # dag_txt += 'JOB sweep_init sweep_init.sub\n'
    # dag_txt += f'VARS sweep_init config_pathname="config.yaml" output_config_pathname="{sweep_config_name}"\n'

    dag_txt += textwrap.dedent(get_service(dag_name=ename + '.dag'))

    # Grab the resources, if targeting is desired
    resources = []
#    resources = get_ospool_resources()
    resources += get_resources_from_yaml()

    # Create experiment permutations and expansion
    # TODO: It doesn't really make sense to do both resource and var expansions,
    # but should be mulled over a bit
    # experiment._add_var_permutations()
    experiment._add_resource_permutations(resources)

    # Create job descriptions for each resource.
    resources = []
    resource_names = set()
    ospool_resources = []
    for tr in experiment.training_runs:
        for resource in tr.resources:
            if resource.name in resource_names: continue
            resources.append(resource)
            resource_names.add(resource.name)
            if resource.resource_type == ResourceType.OSPOOL:
                ospool_resources.append(resource)

    # Generate shared ospool submit description if there are any ospool resources
    if ospool_resources:
        dag_txt += get_ospool_submit_description(config, experiment)

    # Generate individual submit descriptions for non-ospool resources
    for resource in resources:
        if resource.resource_type != ResourceType.OSPOOL:
            dag_txt += get_submit_description(None, resource, config, experiment)

    i = 0
    for tr in experiment.training_runs: # for each shishkabob
        # Initialize the run
        run_prefix = f'run{i}'

        if not os.path.exists(tr.run_uuid):
            os.makedirs(tr.run_uuid)

        for j, epoch in enumerate(range(epochs_per_job, num_epoch+1, epochs_per_job)): #gross hack
            resource = tr.resources[j] if tr.resources else Resource(name="default")
            # input_model_postfix = 'init' if j == 0 else f'epoch{j-1}'
            submit_file = "ospool_pretrain.sub" if resource.resource_type == ResourceType.OSPOOL else f"{resource.name}_pretrain.sub"
            job = Job(name=f'{run_prefix}-train_epoch{j}',
                      submit=submit_file, epoch=epoch,
                      run_uuid=tr.run_uuid, tr_id=j)

            jobs_txt += textwrap.dedent(f'''\
                    JOB {job.name} {job.submit}
                    {'JOB {job.eval_name} {job.eval_submit}' if EVAL else ''}''')
            vars_txt += get_vars(job, resource, tr)

            script_txt += get_script(job, resource, config)

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


        dag_txt += jobs_txt + '\n' + vars_txt  + '\n' + edges_txt  + '\n' + script_txt + '\n'

        # flush out each shishkabob
        jobs_txt = ''
        vars_txt = ''
        edges_txt = ''
        script_txt = ''
        i+=1

    # misc directives
    dag_txt += '\nRETRY ALL_NODES 3\n'
    dag_txt += 'NODE_STATUS_FILE nodes.dag.status 30\n'

    with open(f'{ename}.dag', 'w') as f:
        f.write(dag_txt)
    print(f'generated {ename}.dag')

if __name__ == "__main__":
    app()
