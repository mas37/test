# HOW TO:
Since we do not distribute PANNA as a library to use the tools add the PANNA 
folder to the shell variable PYTHONPATH

```script
   export PYTHONPATH=/path/to/panna
```

# ijk.py
It is a tool that shows a heat map of the distribution of the radial part.
parameters
  - -s or --source: source folder where examples files are located
  - -r or --r_cut: radial cutoff to plot in Angstrom
  - -a or --atomic_sequence: atomic sequence, comma separated
  - -b or --bins: number of bins on radial axis, angular axis will be generated
                  accordingly. optional, default: 100
  - -n or --number_of_elements: number of elements to use to create the graph
                                useful if the data set is too big.
                                optional, default: all the elements
  - --log : put the color scale of the heat map in log scale
  - --dump : dump the data used to generate the heat map in to a readable format

example of usage:
```script
python3 tools/ijk.py  -s path/to/the/dataset -r 5.0 -a H,N,C,O -n 100
```
