# NOT YET WORKING WITH PANNA!!
# quantum espresso output parser
Up to now this is a parser wanna be, so use it at your own risk.
This code is working on PWSCF > 6.0.
Version 0.1 worked on version 5.4 some regexp changed so I don't support the old version anymore.

It is developed around SCF + BFGS calculations so if other calculations are supported it is just by chance. The code has been tested over more than 500 output.

If options capable of forcing the calculation to continue even if there has been no convergence are active in the input file the code wont fail but the data of the steps that didn't converged could be unreliable. 

# How to use it:
what you need is file_parser(file_name) function, in data_magic module (documentation in line).
Furthermore you need to add the atoms name that you are going to use in the string atoms_name in 'parser.py'

like:

```python
atoms_name = r'(?:C|H|O|N)'
atoms_name = r'(?:C|H|O|N|Ar)'
```


Apply me just apply it to all the files in a folder and give you the total result as a json with all the simulations. Apply me require some modification for your particular situation.



# data structure:
```
id_simulation:
        |
        + file name
        + id  previous simulation
        + id next simulation
        + id first simulation of the family
        + id last simulation of the family
        + number in the sequence (i, n_tot)
        + bli (bravais lattice index)
        + alat
        |   |
        |   + value
        |   + units
        |   --or--
        |   |
        |   + units
        + unit_cell_volume
        |   |
        |   + value \ only if in a.u
        |   + units / otherwise not avail
        + cell_side_units
        |   |
        |   # for now units 
        |   # like: 
        |   # - bohr 
        |   # - alat in this case look for
        |   # alat key
        + cell_side
        |   |
        |   + a1
        |   + a1
        |   + a3
        + natoms
        + nspecies
        + nelectrons
        + nksstates
        + cutoff
        + charge_cutoff
        + threshold
        + mixing
        + force_units
        + apos_units
        |   |
        |   # in bfgs several value are possible
        |   # in scf crystal only is supported
        + atoms
        |     |
        |     + number of the atom
        |     + type (letter)
        |     + v1 \
        |     + v2  > position (see_apos_units)
        |     + v3 /
        |     + f1 \
        |     + f2  > force (see force_units)
        |     + f3 /
        + cell (crystal coordinate, alat)
        |    |
        |    + v1
        |    + v2
        |    + v3
        + alat of cell
        + total_energy
        |   |
        |   + value
        |   + units
        + E_hartree
        |   |
        |   + value
        |   + units
        + E_onelectron
        |   |
        |   + value
        |   + units
        + E_xc
        |   |
        |   + value
        |   + units
        + E_ewald
        |   |
        |   + value
        |   + units
        + E_paw
        |   |
        |   + value
        |   + units
        + stress_units
        + atom_description
        |   |
        |   + atomic specie
        |   |   |
        |   |   + index in simulation
        |   |   + valence
        |   |   + mass
        |   |   + pseudopotential file
        + stress_tesnsor 
        + pressure_tesnsor
        + kind
        |   |
        |   + scf
        |   + bfgs
        |   |
        |   + value
        |   + units
```
# Errors flag:

the flag **damage** is always True if something went wrong

## serious:

    * atom_description_damage
    * atom_position_damage: code where not able to find atomic positions.
    * cell_side_units_damage

## not serious:

    * forces_damage: some could still be available
    * stress_units_damage
    * pressure_units_damage
    * stress_tensor_damage
    * stress_tensor_damage


# Read the log:
YOU MUST DO IT!

It gives you the idea of what have been parsed and what not.
Debug level write also each chunk of text and the data recovered form it.
Check if the result is OK with what you expected. 

If I will have time I will work on improving the data recovery.


