#!/bin/bash
### This script runs the GPT-4V + SoM models on the entire VWA shopping test set.

model="claude-3-sonnet-20240229"
result_dir="050424_classifieds_claudesonnet_som"
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s.json"

# Define the batch size variable
batch_size=50

# Define the starting and ending indices
start_idx=0
end_idx=$((start_idx + batch_size))
max_idx=234

# Loop until the starting index is less than or equal to max_idx.
while [ $start_idx -le $max_idx ]
do
    # Classifieds reset is quick, so we can do it after every example.
    curl -X POST http://127.0.0.1:9980/index.php?page=reset -d "token=4b61655535e7ed388f0d40a93600254c"
    bash prepare.sh
    python run.py \
     --instruction_path $instruction_path \
     --test_start_idx $start_idx \
     --test_end_idx $end_idx \
     --provider "anthropic"  --model $model \
     --result_dir $result_dir \
     --test_config_base_dir=config_files/test_classifieds \
     --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
     --action_set_tag som  --observation_type image_som

    # Increment the start and end indices by the batch size
    start_idx=$((start_idx + batch_size))
    end_idx=$((end_idx + batch_size))

    # Ensure the end index does not exceed 466 in the final iteration
    if [ $end_idx -gt $max_idx ]; then
        end_idx=$max_idx
    fi
done
