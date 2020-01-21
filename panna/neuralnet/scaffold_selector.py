import logging
from .panna_scaffold import PannaScaffold
#from .peratom_energy_scaffold import PerAtomEnergyScaffold
#from .autoencoder_scaffold import AutoencoderScaffold

logger = logging.getLogger('panna.neuralnet')


def scaffold_selector(scaffold_type):
    logger.debug('scaffold type: %s', scaffold_type)
    if scaffold_type == 'PANNA':
        scaffold = PannaScaffold
    #elif scaffold_type == 'peratom_energy':
    #    scaffold = PerAtomEnergyScaffold
    #elif scaffold_type == 'autoencoder':
    #    scaffold = AutoencoderScaffold
    else:
        raise ValueError('Scaffold not found')
    return scaffold
