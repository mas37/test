#!/bin/bash

echo '########' 
echo 'PREPARING CONFIGURATION FILES' 
echo '########' 
python3 prepare_config_3.py

activations='gaussian rbf relu'
for act_funct in ${activations} 
do

python3 ../../../panna/train.py --config train_${act_funct}.ini
echo ' '
echo ' '
echo ' done training with activation  ' ${act_funct}
echo ' '

python3 ../../../panna/evaluate.py --config validation_${act_funct}.ini

echo ' done evaluating the network with activation  ' ${act_funct}
echo 'done!!!'

done
