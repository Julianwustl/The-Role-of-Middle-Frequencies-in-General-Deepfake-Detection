#!/bin/bash

# Check for the directory argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory=$1

# Find all files in the directory and its sub-directories
# and count them using wc
count=$(find "$directory" -type f | wc -l)

echo "Number of files in $directory and its subdirectories: $count"
