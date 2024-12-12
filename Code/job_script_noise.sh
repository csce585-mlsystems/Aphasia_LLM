#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# Initialize Conda
source /usr/local/anaconda3/etc/profile.d/conda.sh

# Activate the specific Conda environment
conda activate /data/guan3/llava
# Loop through noise_std from 0.05 to 0.16 in increments of 0.01
#for noise in $(seq 0.10 0.01 0.16); do
#for noise in $(seq 0 0.1 0.98); do
for noise in $(seq 0.01 .005 0.05); do
    for modPercent in $(seq 0.62 0.02 1); do
        echo "Running inference with --noise_std=$noise --modification_percentage=$modPercent"
         # Replace decimal points with underscores in filenames for compatibility
        noise_str=$(printf "%.2f" "$noise" | tr '.' '_')
        modPercent_str=$(printf "%.2f" "$modPercent" | tr '.' '_')

        # Run the script with nohup and redirect output to a unique log file
        python inference_code_gpu_test3.py --noise_std $noise --modification_percentage $modPercent > output_${noise_str}_${modPercent_str}.log 2>&1 
        

    done    
    # Wait for the current script to finish before starting the next one
    wait

    echo "Finished inference with --noise_std=$noise --modification_percentage=$modPercent"
done

echo "All jobs completed."

