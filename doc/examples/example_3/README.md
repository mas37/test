###Example 3

This example shows how to train neural network with different activation functions,
namely: gaussain, radial basis functions and ReLU.
Forces are not trained.

The training and validation data come from the results of example_1
 
The procedure is enumerated below.

1 - The script `prepare_config_3.py` prepares the training and validation configuration
    files. In the `train_config.py`, the `forces_cost` set to 0.00 means
    forces are not used for learning. 
    Similarly, we set the `compute_forces` to _False_ in the `evaluate_config.py`
    function.
     
    To produce the training and validation scripts, run the `prepare_config_3.py` script as
    
```
    python3 prepare_config_3.py
```

    This procedure produces the files `train_gaussian.ini` and `validation_gaussian.ini` for 
    gaussian activation function and so on.

2 - Train the network by using the `train.py` code in panna folder.
```
    python3 ../../../panna/train.py --config train_{gaussain, rbf, relu}.ini 
```    
    Evaluate the network using the `evaluate.py` script as
```
    python3 ../../../panna/evaluate.py --config validation_{gaussain, rbf, relu}.ini
``` 
    Different saved checkpoints are evaluated.

   `./run_example_3.sh` performs all the enumerate steps.

