###Example 1

This example shows how to compute the modified Behler-Parrinello (mBP) symmetry function
and pack the data into input and output for neural network training in TensorFlow records with
`gvect_calculator.py` and `tfr_packer.py` scripts. The symmetry functions are in binary format.
The example is demonstrated for water molecules data extracted from ANI-1 database. 
 
 The procedure is enumerated below.
 
1 - The configuration files to build the symmetry functions and pack them in 
   TensorFlow record format are produced using the script `prepare_config_1.py` as
```
   python3 prepare_config_1.py 
```
   This produces `create_gvector.ini` and `create_tfr.ini`.
   
2 - In the `create_gvector.ini` file, we provide a relative path to the 
   example jsons (examples) and the path to save the binary in the 
   **IO_INFORMATION** section. Each keyword in the `create_gvector.ini` file
   contains a short description of its meaning.

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
   
   `./run_example_1.sh` performs both steps at once.
