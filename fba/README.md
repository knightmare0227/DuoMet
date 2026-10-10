## Before running
Please download the DIAMOND alignment tool from the following link: https://github.com/bbuchfink/diamond/wiki

Installation Instructions:

Navigate to your home directory:
```bash
$ cd your_home_directory
```
Download the tool:
```bash
$ wget http://github.com/bbuchfink/diamond/releases/download/v2.1.10/diamond-linux64.tar.gz
$ tar xzf diamond-linux64.tar.gz
```

Parameter Descriptions:

<kbd>rxn_file</kbd> The CSV file of reactions you need to align. The column name for reactions should be rxn_id.

<kbd>reaction_file</kbd> The fixed GPR database file, bigg_MetaCYC_gprs.csv. Modify the path to point to your directory.

<kbd>fasta_file</kbd> The fixed FASTA sequence database, DB_CARVEME.fasta. Modify the path to point to your directory.

<kbd>folder_path</kbd> The folder containing the uploaded sequences. Change this to your directory path.

<kbd>output_csv_file</kbd> The output file for results. Modify this to your directory path.

<kbd>diamond_path</kbd> The path where DIAMOND is installed.
