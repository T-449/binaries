import os
import csv
import re


def parse_performance_file(filename, label):
    with open(filename, 'r') as file:
        data = file.read()

    runs = re.split(r'Run \d+:', data)
    parsed_data = []

    for run in runs[1:]:  # Skip the first element since it's before the first "Run"
        run_data = {'label': label}  # Initialize with the label
        cpu_cycles = re.findall(r'CPU\d+\s+([\d,]+)\s+cycles', run)
        elapsed_time = re.search(r'(\d+\.\d+) seconds time elapsed', run)

        if cpu_cycles and elapsed_time:
            for i, cycles in enumerate(cpu_cycles):
                run_data[f'CPU{i}_cycles'] = int(cycles.replace(',', ''))
            run_data['elapsed_time'] = float(elapsed_time.group(1))
            parsed_data.append(run_data)

    return parsed_data


def write_to_csv(parsed_data, output_file):
    if not parsed_data:
        print(output_file)
        return

    # Get the headers from the first run, ensuring 'label' is included
    headers = ['label'] + [key for key in parsed_data[0] if key != 'label']

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in parsed_data:
            writer.writerow(row)


def process_files(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_directory, filename)
            label = filename.split('_')[0]
            parsed_data = parse_performance_file(file_path, label)

            # Create output file name
            output_file_name = f"{os.path.splitext(filename)[0]}.csv"
            output_file_path = os.path.join(output_directory, output_file_name)

            write_to_csv(parsed_data, output_file_path)


# Set the input directory containing the .out files and the output directory for the CSV files
# input_directory = '/home/heresy/Documents/classic_signatures_code/perf_outputs'
# output_directory = '/home/heresy/Documents/classic_signatures_code/perf_outputs_csv'


input_directory = '/home/heresy/Downloads/MLClassification/binaries/SIG/perf_outputs'
output_directory = '/home/heresy/Downloads/MLClassification/binaries/SIG/perf_outputs_csv'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Process the files and generate the CSV files
process_files(input_directory, output_directory)
