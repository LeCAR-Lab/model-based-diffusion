#!/bin/bash

# Get the list of denoise_*.png files
files=(denoise_*.png)

# Get the total number of files
total=${#files[@]}

# Iterate through the files in reverse order
for ((i=total-1; i>=0; i--)); do
    # Get the current file
    current_file=${files[i]}

    # Extract the index from the current file
    index=$(echo "$current_file" | sed 's/[^0-9]//g')

    # Calculate the new index
    new_index=$((total - index - 1))

    # Generate the new file name
    new_file=$(printf "denoise_%d.png" "$new_index")

    # Rename the file
    mv "$current_file" "$new_file"
done