#!/usr/bin/env python3

import argparse
import os
import textwrap
import yaml

def main():

    dag_txt = ''

    #
    # submit descriptions
    #
    
    # preproc.sub
    dag_txt += textwrap.dedent('''\
        SUBMIT-DESCRIPTION sweep_init.sub {
                container_image = docker://tdnguyen25/ospool-classification:latest
                universe = container

                executable = ./prelude/sweep_init.py
                arguments = $(config_pathname) $(output_config_pathname)
                log = logs/sweep_init_$(Cluster)_$(Process).log
                error = logs/sweep_init_$(Cluster)_$(Process).err
                output = logs/sweep_init_$(Cluster)_$(Process).out

                should_transfer_files = YES
                when_to_transfer_output = ON_EXIT
                transfer_input_files = $(config_pathname)
                transfer_output_files = $(output_config_pathname)

                request_cpus = 1
                request_memory = 2GB
                request_disk = 2GB
        }


        SUBMIT-DESCRIPTION run_init.sub {
                container_image = docker://tdnguyen25/ospool-classification:latest
                universe = container

                executable = ./run/run_init.py
                arguments = $(config_pathname) $(output_config_pathname)
                log = logs/run_init_$(Cluster)_$(Process).log
                error = logs/run_init_$(Cluster)_$(Process).err
                output = logs/run_init_$(Cluster)_$(Process).out

                should_transfer_files = YES
                when_to_transfer_output = ON_EXIT
                transfer_input_files = $(config_pathname)
                transfer_output_files = $(output_config_pathname)

                request_cpus = 1
                request_memory = 2GB
                request_disk = 2GB
        }


        SUBMIT-DESCRIPTION pproc.sub {
                container_image = docker://tdnguyen25/ospool-classification:latest
                universe = container

                executable = ./run/geldparse.py
                arguments = $(config_pathname) $(geld_pathname) $(output_tensor_pathname)
                log = logs/pproc_$(Cluster)_$(Process).log
                error = logs/pproc_$(Cluster)_$(Process).err
                output = logs/pproc_$(Cluster)_$(Process).out

                should_transfer_files = YES
                when_to_transfer_output = ON_EXIT
                transfer_input_files = $(config_pathname), ../pproc/intermediate/$(geld_pathname)
                transfer_output_files = $(output_tensor_pathname)

                request_cpus = 1
                request_memory = 30GB
                request_disk = 20GB
        }


        SUBMIT-DESCRIPTION model_init.sub {
                container_image = docker://tdnguyen25/ospool-classification:latest
                universe = container

                executable = ./run/model_init.py
                arguments = $(config_pathname) $(output_model_pathname)
                log = logs/model_init_$(Cluster)_$(Process).log
                error = logs/model_init_$(Cluster)_$(Process).err
                output = logs/model_init_$(Cluster)_$(Process).out

                should_transfer_files = YES
                when_to_transfer_output = ON_EXIT
                transfer_input_files = $(config_pathname)
                transfer_output_files = $(output_model_pathname)

                request_cpus = 1
                request_memory = 4GB
                request_disk = 4GB
        }


        SUBMIT-DESCRIPTION train.sub {
                container_image = docker://tdnguyen25/ospool-classification:latest
                universe = container

                executable = ./run/ml/train.py
                arguments = $(config_pathname) $(tensor_pathname) $(model_pathname) $(output_model_pathname) $(epoch)
                log = logs/train_$(Cluster)_$(Process).log
                error = logs/train_$(Cluster)_$(Process).err
                output = logs/train_$(Cluster)_$(Process).out

                should_transfer_files = YES
                when_to_transfer_output = ON_EXIT
                transfer_input_files = $(config_pathname), $(tensor_pathname), $(model_pathname)
                transfer_output_files = $(output_model_pathname)

                requirements = (OpSysMajorVer == 8) || (OpSysMajorVer == 9)
                require_gpus = (DriverVersion >= 11.1)
                request_gpus = 1
                +WantGPULab = true
                +GPUJobLength = "short"

                request_cpus = 1
                request_memory = 15GB
                request_disk = 4GB
        }


        SUBMIT-DESCRIPTION evaluate.sub {
                container_image = docker://tdnguyen25/ospool-classification:latest
                universe = container

                executable = ./run/ml/evaluate.py
                arguments = $(config_pathname) $(tensor_pathname) $(model_pathname) $(epoch) $(earlystop_marker_pathname)
                log = logs/evaluate_$(Cluster)_$(Process).log
                error = logs/evaluate_$(Cluster)_$(Process).err
                output = logs/evaluate_$(Cluster)_$(Process).out

                should_transfer_files = YES
                when_to_transfer_output = ON_EXIT
                transfer_input_files = $(config_pathname), $(tensor_pathname), $(model_pathname) 
                transfer_output_files = output/

                requirements = (OpSysMajorVer == 8) || (OpSysMajorVer == 9)
                require_gpus = (DriverVersion >= 11.1)
                request_gpus = 1
                +WantGPULab = true
                +GPUJobLength = "short"

                request_cpus = 1
                request_memory = 15GB
                request_disk = 4GB
        }


        SUBMIT-DESCRIPTION getbestmodel.sub {
                container_image = docker://tdnguyen25/ospool-classification:latest
                universe = container

                executable = ./getbestmodel.py
                arguments = $(config_pathname) bestmodel.info
                log = logs/getbestmodel_$(Cluster)_$(Process).log
                error = logs/getbestmodel_$(Cluster)_$(Process).err
                output = logs/getbestmodel_$(Cluster)_$(Process).out

                should_transfer_files = YES
                when_to_transfer_output = ON_EXIT
                transfer_input_files = $(config_pathname), *.h5, *bestmodel.pt
                transfer_output_files = bestmodel.info

                requirements = (OpSysMajorVer == 8) || (OpSysMajorVer == 9)
                require_gpus = (DriverVersion >= 11.1)
                request_gpus = 1
                +WantGPULab = true
                +GPUJobLength = "short"

                request_cpus = 1
                request_memory = 15GB
                request_disk = 4GB
        }


    ''')

    sweep_config_name = 'sweep.yaml'

    num_shishkabob = config['runs']
    num_epoch = config['epochs']
    
    jobs_txt = ''
    vars_txt = ''
    edges_txt = ''

    dag_txt += 'JOB sweep_init sweep_init.sub\n'
    dag_txt += f'VARS sweep_init config_pathname="config.yaml" output_config_pathname="{sweep_config_name}"\n'

    for i in range(num_shishkabob): # for each shishkabob
        run_prefix = f'run{i}'
        jobs_txt += textwrap.dedent(f'''\
                JOB {run_prefix}-run_init run_init.sub
                JOB {run_prefix}-pproc pproc.sub
                JOB {run_prefix}-model_init model_init.sub\n''')
        vars_txt += textwrap.dedent(f'''\
                VARS {run_prefix}-run_init config_pathname="{sweep_config_name}" output_config_pathname="{run_prefix}-config.yaml"
                VARS {run_prefix}-pproc config_pathname="{run_prefix}-config.yaml" geld_pathname="ap2002_geld.json" output_tensor_pathname="{run_prefix}-ap2002.h5"
                VARS {run_prefix}-model_init config_pathname="{run_prefix}-config.yaml" output_model_pathname="{run_prefix}-model_init.pt"\n''')
        edges_txt += textwrap.dedent(f'''\
                PARENT sweep_init CHILD {run_prefix}-run_init
                PARENT {run_prefix}-run_init CHILD {run_prefix}-pproc {run_prefix}-model_init
                PARENT {run_prefix}-pproc {run_prefix}-model_init CHILD {run_prefix}-train_epoch0\n''')

        for j in range(num_epoch): # for each epoch
            input_model_postfix = 'init' if j == 0 else f'epoch{j-1}'
            jobs_txt += textwrap.dedent(f'''\
                    JOB {run_prefix}-train_epoch{j} train.sub
                    JOB {run_prefix}-evaluate_epoch{j} evaluate.sub''')
            vars_txt += textwrap.dedent(f'''\
                    VARS {run_prefix}-train_epoch{j} config_pathname="{run_prefix}-config.yaml" tensor_pathname="{run_prefix}-ap2002.h5" model_pathname="{run_prefix}-model_{input_model_postfix}.pt" output_model_pathname="{run_prefix}-model_epoch{j}.pt" epoch="{j}"
                    VARS {run_prefix}-evaluate_epoch{j} config_pathname="{run_prefix}-config.yaml" tensor_pathname="{run_prefix}-ap2002.h5" model_pathname="{run_prefix}-model_epoch{j}.pt" epoch="{j}" earlystop_marker_pathname="{run_prefix}.esm"''')
            
            # includes pre and post scripts for early stopping mechanism
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

    # final node
    dag_txt += 'FINAL getbestmodel getbestmodel.sub\n'
    dag_txt += 'VARS getbestmodel config_pathname="sweep.yaml"'
    dag_txt += f'SCRIPT POST getbestmodel cleanup.py {sweep_config_name}\n' 

    # misc directives
    dag_txt += '\n RETRY ALL_NODES 3\n'
    dag_txt += '\nNODE_STATUS_FILE nodes.dag.status 30\n'

    with open('pipeline.dag', 'w') as f:
        f.write(dag_txt)
    print('generated pipeline.dag')


if __name__ == "__main__":
   
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    with open(args.config, 'r') as config:
        main(yaml.safe_load(config))

