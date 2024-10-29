#!/usr/bin/env python3

import argparse
import os
import textwrap
import htcondor
import random
import sys
import yaml

SHUFFLE=True

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
    return unique_resources

def get_permutations(resources: dict, permutations: int, sites_per_permutation: int) -> list:
    """
    Usage: generate a list of permutations of resources
    @param resources: dictionary whose keys are the names of all unique GLIDEIN_ResourceName s
                      currently visible in the OSPool
    @param permutations: number of permutations to generate
    @return: list of lists of permutations of resources
    """
    resource_list = list(resources.keys())
    permutations_list = []
    if sites_per_permutation > len(resource_list):
        print("Requested number of sites is incompatible than available. No re-use of sites is allowed (yet) so this experiment cannot be run. Exiting.")
        sys.exit(1)
    while len(permutations_list) < permutations:
        print(len(permutations_list))
        permutation = []
        while len(permutation) < sites_per_permutation:
            resource = random.choice(resource_list)
            if resource not in permutation:
                permutation.append(resource)
        if permutation not in permutations_list:
            permutations_list.append(permutation)
    return permutations_list

def main(config):
    dag_txt = ''
    #
    # submit descriptions
    #
    
    # preproc.sub
    # TODO: read descriptions from templates and sub in values as needed
    # TODO: resource-choosing variable
    # TODO: Use $(epoch) to pass in the max number of epochs 
    dag_txt += textwrap.dedent(f'''\
        SUBMIT-DESCRIPTION metl_pretrain.sub {{
                universe = container
                container_image = osdf:///ospool/ap40/data/ian.ross/metl.sif

                # request_disk should be at least 10x the size of the input files
                # processing propose typically uses about 3.5 GB of CPU memory and 8GB of GPU memory
                # add cpus=xx disk=yy memory=zz on the submit command line to override these defaults
                request_disk = $(disk:5GB)
                request_memory = $(memory:6GB)
                request_cpus = $(cpus:1)
                request_gpus = 1
                gpus_minimum_capability = 7.5
                gpus_minimum_memory = 8192

                {'TARGET.GLIDEIN_ResourceName == "$(ResourceName)"' if SHUFFLE else ''}

                environment = "WANDB_API_KEY=123"

                +is_resumable = true

                executable = /bin/bash
                transfer_executable = false
                arguments = pretrain.sh

                transfer_input_files = pretrain.sh
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
    ''')

    sweep_config_name = 'sweep.yaml'

    # TODO: specify resources within config
    num_shishkabob = config['runs']
    # TODO: This is set in the metl runtime options, so we'll  need to update that to pull it in from config or read it from the METL run config
    num_epoch = config['epochs']
    
    jobs_txt = ''
    vars_txt = ''
    edges_txt = ''

    # TODO: add flexibility in this structure?
    # Provisioning node
    dag_txt += 'JOB sweep_init sweep_init.sub\n'
    dag_txt += f'VARS sweep_init config_pathname="config.yaml" output_config_pathname="{sweep_config_name}"\n'

    # Create resource permutations
    permutations = get_permutations(get_resources(), num_shishkabob, num_epoch)

    # The shishkebabs (permutations)
    for i in range(num_shishkabob): # for each shishkabob
        print(i)
        # Initialize the run
        run_prefix = f'run{i}'
        # jobs_txt += textwrap.dedent(f'''\
        #         JOB {run_prefix}-run_init run_init.sub
        #         JOB {run_prefix}-pproc pproc.sub
        #         JOB {run_prefix}-model_init model_init.sub\n''')
        # vars_txt += textwrap.dedent(f'''\
        #         VARS {run_prefix}-run_init config_pathname="{sweep_config_name}" run_prefix="{run_prefix}" output_config_pathname="{run_prefix}-config.yaml"
        #         VARS {run_prefix}-pproc config_pathname="{run_prefix}-config.yaml" geld_pathname="ap2002_geld.json" output_tensor_pathname="{run_prefix}-ap2002.h5"
        #         VARS {run_prefix}-model_init config_pathname="{run_prefix}-config.yaml" output_model_pathname="{run_prefix}-model_init.pt"\n''')
        # edges_txt += textwrap.dedent(f'''\
        #         PARENT sweep_init CHILD {run_prefix}-run_init
        #         PARENT {run_prefix}-run_init CHILD {run_prefix}-pproc {run_prefix}-model_init
        #         PARENT {run_prefix}-pproc {run_prefix}-model_init CHILD {run_prefix}-train_epoch0\n''')

        # TODO: train for a given epoch range, then exit. With evaluation sidecare + short-circuiting
        # TODO: use resource list to generate shuffles (via VARS->ResourceName)

        for j in range(num_epoch): # for each epoch
            input_model_postfix = 'init' if j == 0 else f'epoch{j-1}'
            jobs_txt += textwrap.dedent(f'''\
                    JOB {run_prefix}-train_epoch{j} metl_pretrain.sub
                    JOB {run_prefix}-evaluate_epoch{j} evaluate.sub''')
            vars_txt += textwrap.dedent(f'''\
                    VARS {run_prefix}-train_epoch{j} config_pathname="{run_prefix}-config.yaml" epoch="{j} ResourceName="{permutations[i][j]}"
                    VARS {run_prefix}-evaluate_epoch{j} config_pathname="{run_prefix}-config.yaml" epoch="{j}" earlystop_marker_pathname="{run_prefix}.esm"''')
            
            # includes pre and post scripts for early stopping mechanism
            # TODO: why is earlystopdetector.py being called in both a pre and post?
            edges_txt += textwrap.dedent(f'''\
                    PARENT {run_prefix}-train_epoch{j} CHILD {run_prefix}-evaluate_epoch{j}
                    SCRIPT PRE {run_prefix}-train_epoch{j} earlystopdetector.py {run_prefix}.esm
                    SCRIPT POST {run_prefix}-train_epoch{j} earlystopdetector.py {run_prefix}.esm''')

            # connect to successor train node
            if j < num_epoch - 1:
                edges_txt += f'\nPARENT {run_prefix}-train_epoch{j} CHILD {run_prefix}-train_epoch{j + 1}'

            # create newlines (pretty view)
            if j < num_epoch - 1:
                jobs_txt += '\n'
                vars_txt += '\n'
                edges_txt += '\n'

        dag_txt += '\n' + jobs_txt + '\n' + vars_txt + '\n' + edges_txt + '\n'        
        
        # flush out each shishkabob
        jobs_txt = ''
        vars_txt = ''
        edges_txt = ''

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
