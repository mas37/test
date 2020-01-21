import re
import hashlib
import logging

from regexp import *
import parser
logger = logging.getLogger(__name__)



def file_parser(file):
    """
    file: name of the file as string
    output: dictionary with data (see README)
    """
    textfile = open(file, 'r')
    filetext = textfile.read()

    if len(re.findall(qe_info['r_close'], filetext)) == 1:
        logger.info('job eneded correctly')
    elif len(re.findall(qe_info['r_close'], filetext)) == 0:
        logger.info('job ended uncorrectly')
    else:
        logger.info('wut?? THIS FILE IS NOT MENT TO BE PARSED')
        # TODO raise an error wold be better
        return {}

    # how many energies are in the file?
    # 1+ energy => OK.
    # 0 energy => no simulation.
    matches = [x for x in re.findall(scf_output['r_total_energy'], filetext,
                                     re.MULTILINE)]
    if len(matches) != 0:
        n_simulations = len(matches)
        logger.info('{} simulations found'.format(n_simulations))
    else:
        logger.info('no energy found, very bad!!!!')
        # TODO raise an error wold be better
        return {}

    # simulation initialization:
    simulations = {}

    if_bfgs, split_text = parser.find_calculations(filetext)

    if if_bfgs:
        for i, v in enumerate(split_text):
            kind, text = v
            simulation = {}
            valid_simulation = True
            damage_simulation = False
            if kind == 'scf':
                try:
                    simulation.update(parser.scf_complete(text))
                except parser.CorruptedData as e:
                    logger.info(str(e))
                    logger.info('energy recovered')
                    simulation.update(e.parsed_data)
                    simulation['damage'] = True
                    damage_simulation = True
                except parser.EnergyError as e:
                    logger.exception('Energy not found')
                    valid_simulation = False
            elif kind == 'bfgs':
                try:
                    simulation.update(parser.bfgs_complete(text))
                except parser.CorruptedData as e:
                    logger.info(str(e))
                    logger.info('energy recovered')
                    simulation.update(e.parsed_data)
                    simulation['damage'] = True
                    damage_simulation = True
                except parser.EnergyError as e:
                    logger.info(e)
                    valid_simulation = False
            else:
                raise ValueError('kind not implemented')

            # be careful: key here is the key of the previous simulation!
            if not valid_simulation:
                for k, v in simulations.items():
                    v['last'] = key
                break

            # i = 0 does not have a previous key
            if i > 0:
                previous_key = key

            key = hashlib.sha224(text.encode('utf-8')).hexdigest()

            # key manager among the simulations
            if i == 0:
                simulations[key] = dict(file=str(file),
                                        firts=key,
                                        number=(i, len(split_text) - 1))
                simulations[key].update(simulation)
                first_key = key
            elif (i == len(split_text) - 1) or damage_simulation:
                simulations[previous_key]['next'] = key
                for k, v in simulations.items():
                    v['last'] = key
                simulations[key] = dict(file=str(file),
                                        first=first_key,
                                        number=(i, len(split_text) - 1),
                                        previous=previous_key,
                                        last=key)
                simulations[key].update(simulation)
                break
            else:
                simulations[key] = dict(file=str(file),
                                        first=first_key,
                                        number=(i, len(split_text) - 1),
                                        previous=previous_key)
                simulations[key].update(simulation)
                simulations[previous_key]['next'] = key

    else:
        logger.info('no bfgs calculation founded')
        kind, text = split_text[0]
        key = hashlib.sha224(text.encode('utf-8')).hexdigest()
        simulations[key] = dict(file=str(file),
                                firts=key,
                                last=key,
                                number=(0, 0))
        try:
            simulations[key].update(scf_complete(text))
        except CorruptedData as e:
            logger.info(str(e) + '\n')
            if 'total_energy' in e.parsed_data:
                logger.info('energy recovered')
                simulations[key].update(e.parsed_data)
                simulations[key]['damage'] = True
    textfile.close()
    return simulations


if __name__ == '__main__':
    logging.basicConfig(filename='./parse.log', level=logging.DEBUG)
    s = file_parser('problematic_test/output.19-04-2017.14.50.08')
