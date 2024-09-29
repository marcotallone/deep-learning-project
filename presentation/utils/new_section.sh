#!/bin/bash

# Check if the input argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <name>"
  exit 1
fi

# Check if we're in the right directory (we should see the sections/ folder)
if [ ! -d "sections" ]; then
	echo "Error: sections/ folder not found. Please run this script from the root of the project."
	exit 1
fi

# Assign the input argument to a variable
name=$1

# Create the sections directory if it doesn't exist
# mkdir -p sections

# Create the <name>.tex file in the sections/ folder
file_path="sections/${name}.tex"

# Write the content to the file
cat <<EOT > "$file_path"
\documentclass[../main.tex]{subfiles}
\graphicspath{{\subfix{../images/}}} % Images path

\begin{document}

\section{$name}

\end{document}
EOT

echo "File $file_path created and initialized."
