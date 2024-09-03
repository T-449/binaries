#!/bin/bash

# Check if the script name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <script_name>"
  exit 1
fi

script_name=$1

# Run the target script in the background
./"$script_name" &
script_pid=$!

#echo "Started $script_name with PID $script_pid"

# Function to print memory usage
print_memory_usage() {
  if [ -e /proc/$script_pid/status ]; then
    echo "Memory usage for PID $script_pid ($script_name):"
    grep -E 'VmSize|VmRSS|VmData|VmStk|VmExe|VmLib|VmPTE|VmSwap' /proc/$script_pid/status
  else
    echo "Process $script_pid has already exited."
  fi
}

# Monitor memory usage until the script completes
while kill -0 "$script_pid" 2> /dev/null; do
  print_memory_usage
done
