# Data Parsing

Given a log file "EventLog", the sequence for preprocessing is as follows:

```bash
./geld.py EventLog # Parses the global event log categorized by job id into a JSON.

./geldparse.py JSONFILE # parses output from geld.py into a tensor.
```

outdated:
./filter.py geld_out.json # Filters out jobs that does not have hold>release state sequence.
./label.py filter_out.json # Labels the jobs as transient or non-transient.
./logs2ts.py label_out.json # Converts into a time-series data format.


# Model Training
Use the docker image as the environment
```bash
docker pull tdnguyen25/pytorch-wandb:latest
```

Alternatively, you can simply submit it as a condor job 
```bash
condor_submit model/job.sub
```

In both cases, setup the config.yaml file.
- Wandb key
- Preprocessing sweep space
- Model training sweep space
- Model tensors
