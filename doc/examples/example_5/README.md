###Example 5: Train a neural network on more nodes.

To achieve parallel training is necessary to pass more command line parameters,
some of which depend on the installed scheduler to the train executable.
Namely

* `list_of_nodes`:
  a string that has all the name of the nodes, separated by commas, that are part of the training. 
  In SLURM environment this can be obtained with the following command:

```
  EXPANDED_NODE_LIST="$(scontrol show hostname "$SLURM_NODELIST" | paste -d, -s)"
```
* `task_index_variable`:
  the name of the global variable that contains the number of the process

* `communication_port` (optional):
  port to use for communications between nodes. Default port is 22222.
