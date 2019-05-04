## Examples of PANNA usage

These example should be done after following the tutorial 
in the tutorial directory deligently.

The instructions on how to run each example are given below.

It is assumed that the user is already familiar with the use
of different components of PANNA from the creation of gvectors all the 
way to evaluation of trained NN.

All scripts needed are in `/path/to/panna/` as listed below.

* `gvect_calculator.py` : create gvectors  
* `tfr_packer.py` : pack data in tensorflow records format
* `train.py` : train the network
* `evaluate.py` : evaluate the network

To perform each of the task, type the following on the command line
```
python3 _script.py_ --config config_file
```
The file `config_file` is as decribed in the tutorial. 
The content of the config file depends on the particular task to be carried out.

## Contents of Examples
**Each example folder contains a README.md file that describe how to run the example.**


* Example 1 : Creation of symmetry functions and tensorflow record format water molecule.

* Example 2 : Creation of symmetry functions and derivatives and tensorflow record format for Carbon.
                     
* Example 3 : Training and validation without forces for water molecules with different activation functions.
 
* Example 4 : Neural network potential for carbon with forces.

* Example 5 : Parallel training with PANNA

* Example 6 : PANNA with LAMMPS, the case of carbon (TODO)


