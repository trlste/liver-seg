#!/bin/bash

# Usage:
#   ./run_all.sh /path/to/folder

FOLDER="$1"

if [ -z "$FOLDER" ]; then
    echo "Usage: $0 /path/to/folder"
    exit 1
fi

if [ ! -d "$FOLDER" ]; then
    echo "Error: $FOLDER is not a directory"
    exit 1
fi

# Loop through all .tif files
find "$FOLDER" -type f -name "*Snapshot*.tif" | while read -r file; do
    # Handle case where no tif files exist
    [ -e "$file" ] || continue

    if [[ "$file" == *_otsu.tif ]]; then
        continue
    fi

    echo "Processing $file"
    python3 simple_otsu.py "$file"
done

echo "Done."
