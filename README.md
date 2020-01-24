To copy the file structure of YACHAY folder please type this command:
    find . -type d > dirs.txt

To create the file structure in the folder you wish:
    xargs mkdir -p < dirs.txt
