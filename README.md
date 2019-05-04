## PANNA
### Properties from Artificial Neural Network Architectures

### WORK-IN-PROGRESS (not everything below is released yet)

PANNA is a package to train and validate all-to-all connected network models for BP[1] and modified-BP[2] type
local atomic environment descriptors and atomic potentials.

* Wiki : `https://gitlab.com/pannadevs/panna/wikis/home`
* Tutorial: `https://gitlab.com/pannadevs/panna/blob/master/doc/tutorial/tutorial.md`
* Mailing list: `https://groups.google.com/d/forum/pannausers`
* Source: `https://gitlab.com/pannadevs/panna`
* Bug reports: `https://gitlab.com/pannadevs/panna/issues`
    
####It provides:

* an input creation tool (atomistic calculation result -> G-vector )
* an input packaging tool for quick processing of TensorFlow ( G-vector -> TFData bundle)
* a network training tool
* a network validation tool 
* a LAMMPS plugin
* a bundle of sample data for testing
* (BONUS: an expander for ANI dataset [3]) 


####Testing:

Simple tests to check functionality can be run with:
```
    python3 ./panna/test_gvect_calculator.py 
    python3 ./panna/test_tfr_packer.py 
    python3 ./panna/test_train.py
    python3 ./panna/test_evaluate.py 
```    
All tests can be run with:
```
    python3 -m unittest
``` 
from the panna directory.

LAMMPS integration tests are coming soon. 


####REFERENCES

    [1] J. Behler and M. Parrinello, Generalized Neural-Network 
    Representation  of  High-Dimensional  Potential-Energy
    Surfaces, Phys. Rev. Lett. 98, 146401 (2007)
    [2] Justin S. Smith, Olexandr Isayev, Adrian E. Roitberg. 
    ANI-1: An extensible neural network potential with DFT accuracy 
    at force field computational cost. Chemical Science,(2017), DOI: 10.1039/C6SC05720A
    [3] Justin S. Smith, Olexandr Isayev, Adrian E. Roitberg. 
    ANI-1, A data set of 20 million calculated off-equilibrium conformations 
    for organic molecules. Scientific Data, 4 (2017), Article number: 170193, 
    DOI: 10.1038/sdata.2017.193
