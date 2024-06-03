#!/bin/bash

# Define the directory to store output files
output_directory="perf_outputs"

# Create the directory if it does not exist
mkdir -p $output_directory

# Loop through all .out files in the current directory
for file in ../KEM/*.out; do
    # Define an output file within the output directory for the aggregated results of this .out file
    output_file="${output_directory}/${file}_perf_output.txt"
    
    # Initialize or clear the output file
    echo "Performance stats for $file" > "$output_file"
    
    # Execute perf stat 100 times for each .out file
    for i in $(seq 1 1000); do
        # Run perf stat and append the output to the single output file
        echo "Run $i:" >> "$output_file"
        perf stat -e cycles -a -A ./$file >> "$output_file" 2>&1
    done
done

