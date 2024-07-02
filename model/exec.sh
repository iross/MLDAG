#!/bin/bash

echo 'running executable'

# makes sure the cwd is in ./ml for file open and save
cd ./ml

pwd

python ./train.py

exit 0
