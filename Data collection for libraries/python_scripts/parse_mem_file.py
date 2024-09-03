import os
import csv


def process_memory_usage_file(input_file, output_csv):
    with open(input_file, 'r') as file:
        data = file.readlines()

    runs = []
    current_run = {}
    run_id = 0

    for line in data:
        line = line.strip()
        if line.startswith("Run"):
            if current_run:
                # Calculate average for the current run and store it
                for key in current_run:
                    if current_run[key]:  # Check if the list is not empty
                        current_run[key] = sum(current_run[key]) / len(current_run[key])
                    else:
                        current_run[key] = 0
                runs.append(current_run)
            # Start a new run
            current_run = {'VmSize': [], 'VmRSS': [], 'VmData': [], 'VmStk': [], 'VmExe': [], 'VmLib': [], 'VmPTE': [],
                           'VmSwap': []}
            run_id += 1
        elif 'kB' in line:
            parts = line.split()
            key = parts[0].strip(':')
            value = int(parts[1])
            current_run[key].append(value)

    # Add the last run if it exists
    if current_run:
        for key in current_run:
            if current_run[key]:  # Check if the list is not empty
                current_run[key] = sum(current_run[key]) / len(current_run[key])
            else:
                current_run[key] = 0
        runs.append(current_run)

    # Write the results to a CSV file
    fieldnames = ['VmSize', 'VmRSS', 'VmData', 'VmStk', 'VmExe', 'VmLib', 'VmPTE']
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for run in runs:
            writer.writerow({key: run[key] for key in fieldnames})

    print(f"Average memory usage per run has been written to {output_csv}")


# Define the directory containing the input files
input_directory = '/home/heresy/Downloads/ashish-backup-0831241-0209p/clean/SIG/mem_outputs'  # Replace with your actual input directory
output_directory = '/home/heresy/Downloads/ashish-backup-0831241-0209p/clean/SIG/mem_outputs_csv'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Process each .txt file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.txt'):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_average.csv")
        process_memory_usage_file(input_file_path, output_file_path)
