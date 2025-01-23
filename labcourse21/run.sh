#!/bin/bash

# Check if the required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <cpp_file> <output_file>"
    exit 1
fi

# Get the C++ file and output file from the command-line arguments
CPP_FILE="$1"
OUTPUT_FILE="$2"
OUTPUT_DIR="output"

# Check if the C++ file exists
if [ ! -f "$CPP_FILE" ]; then
    echo "Error: $CPP_FILE does not exist."
    exit 1
fi

# Compile the C++ file
echo "Compiling $CPP_FILE..."
g++ -o "$OUTPUT_DIR/$OUTPUT_FILE" "$CPP_FILE"

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful."
    
    # Execute the compiled program
    echo "Running the program..."
    ./"$OUTPUT_DIR/$OUTPUT_FILE"
else
    echo "Compilation failed."
    exit 1
fi

