#!/bin/bash

py_script_path=/cluster/scratch/jingyli/iml/task3
data_path=/cluster/scratch/jingyli/iml/data

# specify the test name here
test_name=FullModelLong


command="
		python ${py_script_path}/run_cluster.py
		--name jingyli_${test_name}
    --data_root ${data_path}
    --output_root ${data_path}/out/
    --model_path 0
    --user_home ${HOME}
    --feature_path feature.pt
    --train_data_percentage 0.8
    --batch_size 64
    --test_batch_size 32
		--num_epochs 40
    --learning_rate 0.001
    --triplet_margin 1
    --num_workers 0
    --log_frequency 1
    --checkpoint_frequency 2
    --only_first_n 0"


echo "*******************"
echo "   IML Task3 - v2  "
echo "*******************"
echo ""

# run
eval $command
