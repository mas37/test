#!/bin/bash
###

#create_config files
python3 prepare_config_1.py

#create gvectors
 
python3 ../../../panna/gvect_calculator.py --config create_gvector.ini
echo ' '
echo ' '
echo ' done building Behler-parrinello symmetry functions'
echo ' '
echo ' ...packing data to tf records'

echo ' '
echo ' '
#pack data to tfr format
python3 ../../../panna/tfr_packer.py --config create_tfr.ini

echo 'done!!!'

