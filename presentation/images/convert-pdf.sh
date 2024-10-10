#!/bin/bash

# Loop through all PDF files in the current directory
for pdf in *.pdf; do
    # Extract the base name (without extension)
    base_name="${pdf%.pdf}"
    # Convert the PDF to PNG
    magick convert -density 300 "$pdf" -quality 90 "${base_name}.png"
done
