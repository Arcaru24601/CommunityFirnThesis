#!/bin/bash

# HL dynamic
echo 'Creating new directory...'
mkdir "$HOME/projects/Thesis/CFM_Kathi/CFM_main/results/NGRIP_HL_up/"
cd "$HOME/projects/Thesis/CFM_Kathi/CFM_main/"
python3 main.py FirnAir_NGRIP_HLd_up.json -n
echo 'Moving output files from python script to new directory!'
mv "$HOME"/projects/Thesis/CFM_Kathi/CFM_main/resultsFolder/*.hdf5 "$HOME"/projects/Thesis/CFM_Kathi/CFM_main/results/NGRIP_HL_up/


echo 'Creating new directory...'
mkdir "$HOME/projects/Thesis/CFM_Kathi/CFM_main/results/NGRIP_HL_lo/"
cd "$HOME/projects/Thesis/CFM_Kathi/CFM_main/"
python3 main.py FirnAir_NGRIP_HLd_lo.json -n
echo 'Moving output files from python script to new directory!'
mv "$HOME"/projects/Thesis/CFM_Kathi/CFM_main/resultsFolder/*.hdf5 "$HOME"/projects/Thesis/CFM_Kathi/CFM_main/results/NGRIP_HL_lo/


# Barnola
echo 'Creating new directory...'
mkdir "$HOME/projects/Thesis/CFM_Kathi/CFM_main/results/NGRIP_Barnola_up/"
cd "$HOME/projects/Thesis/CFM_Kathi/CFM_main/"
python3 main.py FirnAir_NGRIP_Bar_up.json -n
echo 'Moving output files from python script to new directory!'
mv "$HOME"/projects/Thesis/CFM_Kathi/CFM_main/resultsFolder/*.hdf5 "$HOME"/projects/Thesis/CFM_Kathi/CFM_main/results/NGRIP_Barnola_up/

echo 'Creating new directory...'
mkdir "$HOME/projects/Thesis/CFM_Kathi/CFM_main/results/NGRIP_Barnola_lo/"
cd "$HOME/projects/Thesis/CFM_Kathi/CFM_main/"
python3 main.py FirnAir_NGRIP_Bar_lo.json -n
echo 'Moving output files from python script to new directory!'
mv "$HOME"/projects/Thesis/CFM_Kathi/CFM_main/resultsFolder/*.hdf5 "$HOME"/projects/Thesis/CFM_Kathi/CFM_main/results/NGRIP_Barnola_lo/


# Goujon
echo 'Creating new directory...'
mkdir "$HOME/projects/Thesis/CFM_Kathi/CFM_main/results/NGRIP_Goujon_up/"
cd "$HOME/projects/Thesis/CFM_Kathi/CFM_main/"
python3 main.py FirnAir_NGRIP_Gou_up.json -n
echo 'Moving output files from python script to new directory!'
mv "$HOME"/projects/Thesis/CFM_Kathi/CFM_main/resultsFolder/*.hdf5 "$HOME"/projects/Thesis/CFM_Kathi/CFM_main/results/NGRIP_Goujon_up/

echo 'Creating new directory...'
mkdir "$HOME/projects/Thesis/CFM_Kathi/CFM_main/results/NGRIP_Goujon_lo/"
cd "$HOME/projects/Thesis/CFM_Kathi/CFM_main/"
python3 main.py FirnAir_NGRIP_Gou_lo.json -n
echo 'Moving output files from python script to new directory!'
mv "$HOME"/projects/Thesis/CFM_Kathi/CFM_main/resultsFolder/*.hdf5 "$HOME"/projects/Thesis/CFM_Kathi/CFM_main/results/NGRIP_Goujon_lo/