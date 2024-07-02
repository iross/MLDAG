# Data Parsing

Given a log file "EventLog", the sequence for preprocessing is as follows:

```bash
./geld.py EventLog # Parses the global event log categorized by job id into a JSON.

./filter.py geld_out.json # Filters out jobs that does not have hold>release state sequence.

./label.py filter_out.json # Labels the jobs as transient or non-transient.

./logs2ts.py label_out.json # Converts into a time-series data format.
```

The final processed json file is "logs2ts_out.json".

# Model Training
Use the docker image as the environment
```bash
docker pull tdnguyen25/pytorch-wandb:latest
```

Alternatively, you can simply submit it as a condor job 
```bash
condor_submit model/job.sub
```

In both cases, insert your wand.ai key at model/ml/metadata.yaml

Put your final processed .json file in model/data and update metadata.yaml accordingly

Finally, you can configure your Sweep's search space in model/ml/hyperparameters.yaml
