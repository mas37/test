###Example 4

This example shows how to train neural network with different activation functions,
namely: gaussain, radial basis functions and ReLU.
Forces are also trained.

The training and validation data come from the results of example_2

The procedure is enumerated below.

1 - The script `prepare_config_4.py` prepares the training and validation configuration
    files. In the `train_config.py`, the `forces_cost` set to 0.02 means that
    there is training on forces. Similarly, we set the `compute_forces` to true in the   
    `evaluate_config.py` script. 

    To produce the training and validation scripts, run the `prepare_config_4.py` script as
```
    python3 prepare_config_4.py
```
    This prcedure produces the files `train_gaussian.ini` and `validation_gaussian.ini` for 
    gaussian activation function and so on.

2 - Train the network by using the `train.py` code in panna folder.
```    
    python3 ../../../panna/train.py --config train_{gaussain, rbf, relu}.ini 
```    
    Evaluate the network using the `evaluate.py` script as
```
    python3 ../../../panna/evaluate.py --config validation_{gaussain, rbf, relu}.ini
```
   `./run_example_4.sh` performa all the enumerated steps.

