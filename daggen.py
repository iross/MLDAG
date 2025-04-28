#!/usr/bin/env python3

import typer
from typing_extensions import Annotated
import textwrap
import htcondor
import random
from pydantic import BaseModel
import yaml
from Resource import Resource, ResourceType, get_resources_from_yaml
from TrainingRun import TrainingRun
from Experiment import Experiment, read_from_config
SHUFFLE=False
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


# TODO: make the vars more flexible. (e.g. for hyperparameter sweeps)
def get_vars(job: Job, resource: Resource, config: dict) -> str:
    return textwrap.dedent(f"""\
        VARS {job.name} epoch="{job.epoch}" run_uuid="{job.run_uuid}" ResourceName="{resource.name}" {'continue_from_checkpoint="true"' if job.tr_id > 0 else ""}
        {'VARS {job.eval_name} epoch="{job.epoch}" run_uuid="{job.training_run.run_uuid}" earlystop_marker_pathname="{job.training_run.run_prefix}.esm"' if EVAL else ''}""")

def get_script(job: Job, resource: Resource, config: dict) -> str:
    # TODO: any other pre-  or post-scripts? Seems resource and job specific..
    script_txt = ''
    if resource.resource_type == ResourceType.ANNEX:
        script_txt += f'SCRIPT PRE {job.name} pre_request_annex.sh {resource.name} {resource.name}_annex_{job.run_uuid}\n'
    return script_txt

def get_service() -> str:
    service_txt = textwrap.dedent("""\
    SUBMIT-DESCRIPTION annex_helper.sub {
        universe = local
        executable = /home/ian.ross/MLDAG/.venv/bin/python
        arguments = annex_helper.py watch --interval 60
        queue
    }
    SERVICE annex_helper annex_helper.sub
    """)
    return service_txt

class EvaluationRun:
    def __init__(self):
        raise NotImplementedError("EvaluationRun is not implemented")


def get_submit_description(job: Job, resource: Resource, config: dict, experiment: Experiment) -> str:
    # TODO this should be built based as needed from the job and resource. Need to think about how much gets done via VARS vs templating -- most probably winds up in VARS
    print(experiment.submit_template)
    inner_txt = experiment.submit_template.format(resource = resource)

    # Hacky. Fix this.
    if "queue" in inner_txt and inner_txt.strip().endswith("queue"):
        inner_txt = inner_txt.strip().rstrip("queue")
    if resource.resource_type == ResourceType.OSPOOL:
        inner_txt += f'TARGET.GLIDEIN_ResourceName == "{resource.name}"\n'
    elif resource.resource_type == ResourceType.ANNEX:
        inner_txt += f'MY.TargetAnnexName = "{resource.name}_annex_$(run_uuid)"\n'
    if "wandb" in config:
        inner_txt += f'environment = "WANDB_API_KEY={config["wandb"]["api_key"]}"\n'
    inner_txt += 'queue\n'

    inner_txt = textwrap.indent(inner_txt, "\t")

    dag_txt = f"SUBMIT-DESCRIPTION {resource.name}_pretrain.sub {{"
    dag_txt += f"\n{inner_txt}"
    dag_txt += "}\n"
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

app = typer.Typer()
@app.command()
def main(config: Annotated[str, typer.Argument(help="Path to YAML config file")] = 'config.yaml'):
    config = yaml.safe_load(open(config, 'r'))
    experiment = read_from_config("Experiment.yaml")
    dag_txt = ''
    
    num_epoch = experiment.vars['epochs']
    epochs_per_job = experiment.vars['epochs_per_job']
    
    jobs_txt = ''
    vars_txt = ''
    edges_txt = ''
    script_txt = ''

    # TODO: add flexibility in this structure?
    # Provisioning node
    # dag_txt += 'JOB sweep_init sweep_init.sub\n'
    # dag_txt += f'VARS sweep_init config_pathname="config.yaml" output_config_pathname="{sweep_config_name}"\n'

    dag_txt += textwrap.dedent(get_service()) 

    # Grab the resources from resources.yaml
    # resources = get_ospool_resources()
    resources = get_resources_from_yaml()
    # Create resource permutations
    experiment._add_resource_permutations(resources)
    # experiment._add_var_permutations()
    # TODO: explode vars here as well? How to couple the TrainingRun and Resource permutations? Unless we just stuff resources as a VAR

    for resource in resources:
        # TODO: one for OSPool and one for each annex.
        # TODO: get submit description based on Experiment.submit_template, with
        # values coming from TrainingRun and Resource instances per job.
        dag_txt += get_submit_description(None, resource, config, experiment)


    i = 0
    for tr in experiment.training_runs: # for each shishkabob
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

            # create newlines (pretty view)
            if j < num_epoch/epoch - 1:
                jobs_txt += '\n'
                vars_txt += '\n'
                edges_txt += '\n'
                script_txt += '\n' if script_txt !='' else ''

        dag_txt += '\n' + jobs_txt + '\n' + vars_txt + '\n' + edges_txt + '\n' + script_txt + '\n'
        
        # flush out each shishkabob
        jobs_txt = ''
        vars_txt = ''
        edges_txt = ''
        script_txt = ''
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
    app()
