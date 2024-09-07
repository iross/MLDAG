# Description
This repository trains an LSTM model using global event logs from the OSPool to make inferences on whether held jobs should be released or removed.

In addition, distributed learning workflow via DAGMan is used for training with Weight & Biases integration for model tracking and hyperparemeters sweeping.

# Tensors

### Design
![tensor2](https://github.com/user-attachments/assets/1dd253d3-f567-41e5-adb8-e52ec501117f)
The goal is to parse job event logs into tensors that incorporates temporal information. Imagine a single tensor as a 3D data object. The orange shaded slice (call this primary job) represents the historical record of a job that has gone on a hold state, but eventually finished its execution. The remaining slices behind are records of other jobs that co-exists during the timeframe of the primary job (defined as when it was submitted to when it goes to history). This part is what gives information on the current state of the environment by sampling how the other jobs are behaving while the primary job exists. 

The actual shape of the tensor is ```e * m * j``` 
 
 ```e```: the events that Condor records (https://htcondor.readthedocs.io/en/latest/codes-other-values/job-event-log-codes.html). 

 ```m```: the number of time frames where the length of each frame in seconds is by ```timeframe_len``` in ```/distributed/config.yaml```.

 ```j```: number of jobs sampled in which the first job is the primary job and j-1 other jobs exists in the timeframe of the primary job.

 The e * m slices (historical records) are a one-hot encoding of the event logs respectively.

 All e, m, j, and timeframe_len are all configruable hyperparamters in ```/distributed/config.yaml```.

### Creation
Tensors are parsed from global event logs. In CHTC or the OSPool, gather the global event logs using ```condor_config_val EVENT_LOG``` to find the path

Most likely, there are multiple global event logs which Condor keeps, simply concatenate them together into a single catenated global event log (CGEL).

```geldparse.py``` takes two arguments: ```pproc.sub <input_CGEL_pathname> <output_CGEL_pathname>```
Edit the ```pproc.sub```'s arguments, transfer_input_files, and transfer_output_files accordingly, as it is the submit description for ```geldparse.py```

Run ```condor_submit pproc.sub``` in the ```pproc/``` directory, and the output are the tensors for training.


# Weight & Biases Integration
The distributed training workflow integrates Weight & Biases (wandb) for model tracking. To configure, see ```config.yaml```.

Configure the the fields under ```wandb``` in the yaml file to set up your wandb experiments.

Do not touch ```sweep_id``` nor ```run_id``` as they are helper runtime variables.

See wandb documentation for more details.

See ```preprocessing``` and ```training``` to define sweep parameters.


# Distributed Training
![dag](https://github.com/user-attachments/assets/aa5ea9bc-a5eb-4cca-ab1f-c42a023fd4cc)

### W&B Search Space Definition
Configure the hyperparameters search space in ```/distributed/config.yaml```

### To submit the DAGMan workflow:

** Have the cwd in ```distributed/``` **

1) Set the number of runs and epochs per run in ```distributed/config.yaml```.
2) Run ```daggen.py``` to generate the submit description ```pipeline.dag```
3) ```condor_submit_dag pipeline.dag``` to submit on Condor
4) After the workflow is complete, ```distributed/cleanup.py``` will organize the output files
5) Look for ```bestmodel.info``` file to see best performing model according to f-measure. If desired, it is possible to change the metric to evaluate for in ```distributed/config.yaml```


Training has early stopping in which the run is terminated after a certain number of epochs in which the model performance does not improve. See ```earlystop_threshold``` in ```distributed/config.yaml```.
