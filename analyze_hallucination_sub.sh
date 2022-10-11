#!/bin/bash

#$ -S /bin/bash
#$ -o log_files/
#$ -cwd
#$ -j y
#$ -r y

FOLDER=
REF_FN=
OUTFILE=

conda  activate SE3-nvidia3

python3 analyze_hallucination.py --folder=$FOLDER --ref_fn=$REF_FN --out_file=$OUTFILE
