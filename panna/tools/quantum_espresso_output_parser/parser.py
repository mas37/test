import re
import logging
from regexp import *

logger = logging.getLogger(__name__)
dump = logging.getLogger(__name__ + 'dump')


fh = logging.FileHandler('simulations_discrard.log')
formatter = logging.Formatter("%(asctime)s - %(name)s -\
%(levelname)s - %(message)s")
fh.setFormatter(formatter)
dump.addHandler(fh)
dump.propagate = False
dump.setLevel(logging.INFO)


class CorruptedData(Exception):
    def __init__(self, message, data, *args):
        super(CorruptedData, self).__init__(message)
        # clean the data form empty keys
        data_celaned = {k: v for k, v in data.items() if v != []}
        for x in args:
            data_celaned['{}_damage'.format(x)] = True
        self.parsed_data = data_celaned


class EnergyError(Exception):
    def __init__(self, message, data, *args):
        super(EnergyError, self).__init__(message)
        # clean the data form empty keys
        data_celaned = {k: v for k, v in data.items() if v != []}
        for x in args:
            data_celaned['{}_damage'.format(x)] = True
        self.parsed_data = data_celaned


def find_calculations(text, verbose=False):
    """
    given a full file check if it is a bfgs calculation or not
    if it is true it splits the text in a vector of single calculation

    output:

    if_bfgs, [('scf','text'), ('bfgs','text'),..]

    if verbose is True the output will be:
        if_bfgs, [('scf','text'), ('bfgs','text'),..], verbose_dict

    verbose_dict:
        bfgs_error = bool
        bfgs_converged = bool
        --- if bfgs_converged = True
        energy < value
        force < value
        cell < value
        -------------
        bfgs_final_scf = bool

    """
    bfgs_data = {}
    verbose_dict = {}
    for x in bfgs_set:
        data = re.findall(bfgs_set[x], text, re.MULTILINE)
        bfgs_data[x[2:]] = data

    # check if a BFGS calculation is present.
    if len(bfgs_data['start']) == 1:
        if_bfgs = True
    else:
        if_bfgs = False
        if verbose:
            return if_bfgs, [('scf', text)], verbose_dict
        else:
            return if_bfgs, [('scf', text)]

    # split the data in several calculation:
    split_data = []
    # first scf calculation
    tmp = re.split(bfgs_set['r_start'], text, flags=re.MULTILINE)
    split_data.append(('scf', tmp[0]))
    tmp = tmp[1]
    # final scf calculation
    if len(bfgs_data['final_scf']) == 1:
        tmp = re.split(bfgs_set['r_final_scf'], tmp, flags=re.MULTILINE)
        scf_last_text = tmp[1]
        # data for next step
        tmp = tmp[0]
    # final set of coordinate:
    # this is useless because those data are founded again in the last scf step
    if len(bfgs_data['end']) == 1:
        tmp = re.split(bfgs_set['r_end'], tmp, flags=re.MULTILINE)
        # enable this line to get data from 'end of bfgs' to 'a final scf'
        # last_coordinate = tmp[1]
        # data for next step
        tmp = tmp[0]
    # discard the convergence part if present
    if len(bfgs_data['bfgs_converged']) >= 1:
        tmp = re.split(bfgs_set['r_bfgs_converged'], tmp, re.MULTILINE)
        tmp = tmp[0]
    # division of all the other steps
    # this is done in the "number of scf cycles line"
    # I am open to pull requests
    bfgs_text = re.split(bfgs_set['r_bfgs_split'], tmp,
                         flags=re.MULTILINE)

    # remove first line if not too long( usually it is just a set of blank
    # spaces)
    # the first line is composed by
    #     BFGS Geometry Optimization*
    #
    #     number of scf cycles*    =   1
    if len(bfgs_text[0]) < 15:
        logger.info('removed head')
        dump.debug('removed --%r--', bfgs_text[0])
        bfgs_text.pop(0)
    # add the part removed by reg-exp
    bfgs_text = ['     number of scf cycles' + x for x in bfgs_text]

    # put all the split data in the right vector
    for x in bfgs_text:
        split_data.append(('bfgs', x))

    if len(bfgs_data['end']) == 1:
        verbose_dict['bfgs_error'] = False
        verbose_dict['bfgs_converged'] = True if \
            len(bfgs_data['bfgs_converged']) >= 1 else False
        logger.debug(verbose_dict)
        logger.debug(bfgs_data)
        if verbose_dict['bfgs_converged']:
            bfgs_data['criteria'] = bfgs_data['criteria'][0]
            verbose_dict[bfgs_data['criteria'][0]] = bfgs_data['criteria'][1]
            verbose_dict[bfgs_data['criteria'][2]] = bfgs_data['criteria'][3]
            verbose_dict[bfgs_data['criteria'][4]] = bfgs_data['criteria'][5]
    else:
        verbose_dict['bfgs_error'] = True

    if len(bfgs_data['final_scf']) == 1:
        split_data.append(('scf', scf_last_text))

    if verbose:
        return if_bfgs, split_data, verbose_dict
    else:
        return if_bfgs, split_data


def scf_in(text, scf_out_feeder=False):
    """
    given data of SCF calculation output return all the data of that simulation
    raise several error if something went wrong,
    """
    simulation = {}
    simulation['kind'] = 'scf'
    for x in scf_input:
        data = re.findall(scf_input[x], text, re.MULTILINE)
        if len(data) == 1:
            simulation[x[2:]] = data.pop()
        else:
            simulation[x[2:]] = data

    # normalization of cell description
    cell_side = simulation.pop('cell_side')
    simulation['cell_side'] = [x[1:] for x in cell_side]

    # normalization of atom description
    simulation['atom_description'] = {}
    for x, y in zip(simulation.pop('dspecies'),
                    simulation.pop('pseudopotential')):
        if x[0] == y[1]:
            pseudo_name = y[2].split('/')[-1]
            simulation['atom_description'][y[1]] = (y[0], x[1], x[2],
                                                    pseudo_name)
        else:
            raise ValueError('Inconsistency in QE output')
    if len(simulation['atom_description']) < int(simulation['nspecies']):
        dump.info('some atom are not well described')
        dump.debug(text)
        raise CorruptedData('some atom are not well described', simulation,
                            'atom_description')

    # creating the conversion table
    conversion = {int(v[0]): k for k, v in
                  simulation['atom_description'].items()}
    # number of atom:
    nat = int(simulation['natoms'])

    # normalization of atomic positions, this part should be done better
    # idea: force a division on work crystallographic axes
    crystal_text = re.split(scf_input_cryst['r_cryst_split_begin'],
                            text, flags=re.MULTILINE)[1]
    crystal_text = re.split(scf_input_cryst['r_cryst_split_end'],
                            crystal_text, flags=re.MULTILINE)[0]
    a_pos = re.findall(scf_input_cryst['r_apos'],
                       crystal_text,
                       re.MULTILINE)

    # add apos_units, only crystal is supported
    simulation['apos_units'] = ['crystal']

    if len(a_pos) < nat:
        dump.info('some position are missing')
        dump.debug(text)
        raise CorruptedData('some position are missing', simulation,
                            'atom_position')

    if scf_out_feeder:
        return (text, nat, conversion, a_pos, simulation)
    else:
        return simulation


def scf_out(text, nat, atom_conversion, positions, simulation={}):
    """
    given data of SCF calculation output return all the data of that simulation
    raise several error if something went wrong,
    nat: number of atom
    atom conversion is needed because scf does not provide atomic name.
    simulation IS CHANGED INSIDE THIS ROUTINE
    """
    keys_not_found = []
    for x in scf_output:
        data = re.findall(scf_output[x], text, re.MULTILINE)
        if len(data) == 1:
            simulation[x[2:]] = data.pop()
        elif len(data) == 0:
            keys_not_found.append(x[2:])
        else:
            simulation[x[2:]] = data

    # the normalizations MUST BE DONE in the same order as the data are
    # collected becouse the first that fails will rise an error and all the
    # others wont be applied.
    if 'total_energy' in keys_not_found:
        raise EnergyError('Energy not found',
                          simulation)
    # normalization of force and positions
    simulation['atom'] = []
    if 'force' not in keys_not_found:
        # no forces at all are available
        force = simulation.pop('force')[:nat]
        # zip by doc cut the lenght of the result to the shorter.
        for x, y in zip(positions, force):
            if int(x[0]) == int(y[0]) and x[1] == atom_conversion[int(y[1])]:
                simulation['atom'].append((x[0], x[1], x[2:], y[2:]))
            else:
                logger.debug(int(x[0]) == int(y[0]))
                logger.debug(x[1] == atom_conversion[int(y[1])])
                raise ValueError('Error in conversion step, check qe output',
                                 simulation)

        if len(force) < nat:
            # this means that not enough forces have been found on
            # the output file
            for x in range(len(force), nat):
                y = positions[x]
                simulation['atom'].append((y[0], y[1], y[2:]))
            dump.info('not enough forces, damage data')
            dump.debug(text)
            raise CorruptedData('not enough forces, damage data', simulation,
                                'forces')
    else:
        for x in range(nat):
            y = positions[x]
            simulation['atom'].append((y[0], y[1], y[2:]))
        dump.info('no forces, damage data')
        dump.debug(text)
        raise CorruptedData('no forces, damage data', simulation,
                            'forces')

    # normalization of stress and pressure information
    simulation['stress_tensor'] = []
    simulation['pressure_tensor'] = []
    if 'stress_units' in keys_not_found:
        dump.info('stress units not found')
        dump.debug(text)
        raise CorruptedData('stress units not found', simulation,
                            'stress_units')
    if 'pressure' in keys_not_found:
        dump.info('pressure not found')
        dump.debug(text)
        raise CorruptedData('pressure not found', simulation,
                            'pressure_units')
    else:
        simulation['pressure'] = [simulation['pressure'][1],
                                  simulation['pressure'][0]]

    for x in simulation.pop('stress_and_kbar_tensor'):
        simulation['stress_tensor'].append(x[:3])
        simulation['pressure_tensor'].append(x[3:])
    if len(simulation['stress_tensor']) < 3:
        dump.info('stress tensor and pressure tensor incomplete')
        dump.debug(text)
        raise CorruptedData('stress tensor and pressure tensor incomplete',
                            simulation, 'stress_tensor', 'stress_tensor')
    return simulation


def scf_complete(text):
    """
    given the output of a complete scf step it returns a dictionary
    with all the data. The output MUST HAVE AT LEAST the '! energy'
    line.
    """
    logger.info('scf calculation found')
    data = scf_in(text, True)
    simulation = scf_out(*data)
    return simulation


def bfgs_complete(text):
    """
    given the output of a complete bfgs step it returns a dictionary
    with all the data. The output MUST HAVE AT LEAST the '! energy'
    line.
    """
    logger.info('bfgs calculation found')
    simulation = {}
    simulation['kind'] = 'bfgs'
    keys_not_found = []
    for x in bfgs_output:
        data = re.findall(bfgs_output[x], text, re.MULTILINE)
        if len(data) == 1:
            simulation[x[2:]] = data.pop()
        elif len(data) == 0:
            keys_not_found.append(x[2:])
        else:
            simulation[x[2:]] = data

    # normalization of cell side units
    try:
        cell_side_units = simulation['cell_side_units'].split('=')
    except KeyError:
        logger.info('cell_side_units not found')
        logger.info('KNOWN BUG, sometimes find_bfgs doesn t do')
        logger.info('his job properly please check simulations_discrard.log')
        dump.info('cell_side_units not found')
        dump.debug(text)
        raise CorruptedData('no cell_side_units', simulation,
                            'cell_side_units')
    
    # cell side units can be only alat or bohr
    if len(cell_side_units) == 2:
        if cell_side_units[0] == 'alat':
            simulation['cell_side_units'] = cell_side_units[0]
            simulation['alat'] = cell_side_units[1]
        else:
            raise ValueError('Only -alat- is supported')

    # normalization of positions
    # FIXME I assume that the output is always in crystal coordinate
    pos = []
    tmp = simulation.pop('apos')
    conversion = {}
    nat = len(tmp)
    c_set = []

    for x in range(nat):
        pos.append([x + 1, tmp[x][0], tmp[x][1], tmp[x][2], tmp[x][3]])
        if tmp[x][0] not in c_set:
            c_set.append(tmp[x][0])

    for i, x in enumerate(c_set):
        conversion[i + 1] = x
    scf_out(text, nat, conversion, pos, simulation)
    return simulation
