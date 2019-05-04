###Example 2

This example shows how to compute the modified Behler-Parrinello (mBP) symmetry function
and pack the data into input and output for neural network training in TensorFlow records with
`gvect_calculator.py` and `tfr_packer.py` scripts. The symmetry functions are in binary format.
The example is demonstrated with layered C structure. 
 
 The procedure is enumerated below.

1 - The configuration files to build the symmetry functions and packed the to
   tensorflow record format produced using the script `prepare_config_2.py`
```
   python3 prepare_config_2.py
```

2 - In the `create_gvector.ini` file, we provide a relative path to the 
   example jsons (examples) and the path to save the binary in the 
   **IO_INFORMATION** section. Each keyword in the `create_gvector.ini` file
   contains a short description of its meaning. 

   The keyword `include_derivatives` is set to _True_ implying we are also
   interested in computing the derivatives of the gvectors. This is very 
   important if forces are to be trained in addition to training energies.

   To create the symmetry functions, run the following command
```
     python3 ../../../panna/gvect_calculator.py --config create_gvector.ini
```
3 - The `create_tfr.ini` contains information required to create the tensorflow records format
   All keywords are described therein.
   
   To create the tfr, run the following command
```
   python3 ../../../panna/tfr_packer.py --config create_tfr.ini
```

   `./run_example_2.sh` performs both steps at once.
