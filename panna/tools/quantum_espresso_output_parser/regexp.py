# util to regexp
unit = r'((?:Ry|a\.?u\.?|(?:b|B)ohr|\/|(?:a|A)ng|kbar|g|cm|ev)+'\
       r'(?:\^|\*\*)?\d*)'
atoms_name = r'(?:C|H|O|N)'

# qe info
qe_info = dict(
    r_PWSCF_version=r'^ *Program PWSCF (.+) starts',
    r_close=r'JOB DONE')

# information of scf actually head informations
scf_input = dict(
    r_pseudopotential=r'^ *PseudoPot. # (\d+) for *(\w{1,2})'
                      r' *read from file:\n^ *(.+\.UPF)$',
    r_bli=r'^ *bravais-lattice index *= *(\d+)',
    r_alat=r'^ *lattice parameter \(alat\) *= *([\d\.\+\-]+) *{}'.format(unit),
    r_unit_cell_volume=r'^ *unit-cell volume *= *([\d\.\+\-]+) *'
                       r'(\(a\.u\.\)\^3)',
    r_cell_side_units=r'^ *crystal axes: \(cart. coord. in units of (alat)\)',
    r_cell_side=r' *a\((\d+)\) *= *\( *(-?[\d.]+) *(-?[\d.]+) *'
                '(-?[\d.]+) *\)',
    r_natoms=r'^ *number of atoms/cell *= *(\d+)',
    r_nspecies=r'^ *number of atomic types *= *(\d+)',
    r_dspecies=r'^ +(\w{1,2}) +(-?[\d.]+) +(-?[\d.]+)',
    r_nelectrons=r'^ *number of electrons *= *([\d\.\+\-]+)',
    r_nksstates=r'^ *number of Kohn-Sham states *= *(\d+)',
    r_cutoff=r'^ *kinetic-energy cutoff *= *([\d\.\+\-]+) *{}'.format(unit),
    r_charge_cutoff=r'^ *charge density cutoff *= *([\d\.\+\-]+) *{}'.format(
                    unit),
    r_threshold=r'^ *convergence threshold *= *(\d+.\d+E?-?\d*)',
    r_mixing=r'^ *mixing beta *= *([\d\.\+\-]+)')

scf_input_cryst = dict(
    r_cryst_split_begin=r'^ *Crystallographic axes',
    r_apos=r'^ +(\d+) +({})[^=]+= \( +([\d\+\-\.]+) +([\d\+\-\.]+) +'
           '([\d\+\-\.]+) +\)'.format(atoms_name),
    r_cryst_split_end=r'^ *Dense  grid')

# data of scf
# TODO, the pressure calculation is carried out in different way
# on different QE version
# here we have the most common version

scf_output = dict(
    r_total_energy=r'^![\w =]+([\d\.\+\- ]+){}'.format(unit),
    r_E_hartree=r'^ +hartree \w+ += +([\d\.\+\-]+) +{}'.format(unit),
    r_E_onelectron=r'^ +one-electron \w+ += +([\d\.\+\-]+) +{}'.format(unit),
    r_E_xc=r'^ +xc \w+ += +([\d\.\+\-]+) +{}'.format(unit),
    r_E_ewald=r'^ +ewald \w+ += +([\d\.\+\-]+) +{}'.format(unit),
    r_E_paw=r'^ +one-center paw \w+\. += +([\d\.\+\-]+) +{}'.format(unit),
    r_force_units=r'^ *Forces acting on atoms (cartesian axes, *{})'.
                  format(unit),
    r_force=r'atom +([\d\.\+\-]+) +type +([\d\.\+\-]+)'
            ' +force = +([\d\.\+\-]+) +([\d\.\+\-]+) +([\d\.\+\-]+)',
    r_stress_units=r' +total +stress +\({}\)'.format(unit),
    r_pressure=r'\((kbar)\) +P= +([\d\.\+\-]+)',
    r_stress_and_kbar_tensor=r'^ {2,3}([\d\.\+\-]+) +([\d\.\+\-]+) +'
                             r'([\d\.\+\-]+) {8,10}([\d\.\+\-]+) +'
                             r'([\d\.\+\-]+) +([\d\.\+\-]+)$')

# information of BFGS
# obs:
# * criteria are present only if the bfgs converged
# * nstep = maximum number of step in BFGS
bfgs_set = dict(
    r_nstep=r'^ *nstep *= *(\d+)',
    r_start=r'^ +BFGS Geometry Optimization$',
    r_bfgs_converged=r' +bfgs converged in +(\d+) +scf cycles and +(\d+) +'
                     'bfgs steps',
    r_bfgs_not_converged=r'^ +The maximum number of steps has been reached.',
    r_criteria=r'^ +\(criteria: +(energy) *< *([\d\.\+\-E]+) *{},'
               r' +(force) *< *([\d\.\+\-E]+) *{},'
               r' +(cell) *< *([\d\.\+\-E]+) *{} *\)'.format(unit, unit, unit),
    r_end=r'^ +End of BFGS Geometry Optimization',
    r_final_scf=r'^ *A final scf calculation at the relaxed structure',
    r_bfgs_split=r'number of scf cycles')

# data of bfgs:
# this is the output that is fed to next step
bfgs_output = dict(
    r_scf_cycles=r'^ *number of scf cycles *= *(\d+)',
    r_bfgs_steps=r'^ *number of bfgs steps *= *(\d+)',
    r_unit_cell_volume=r'^ *new unit-cell volume *= *([\d\.\+\-]+) *{}'.format(
                       unit),
    r_cell_side_units=r'CELL_PARAMETERS \(([\w ]+= +[\d\.\+\-]+|bohr)\)',
    r_cell_side=r'^ {1,3}([\d\+\-\.]+) +([\d\+\-\.]+) +([\d\+\-\.]+)$',
    r_apos_units=r'ATOMIC_POSITIONS \((.+)\)',
    r_apos=r'^({}) +([\d\+\-\.]+) +([\d\+\-\.]+) +([\d\+\-\.]+)$'
                .format(atoms_name))
