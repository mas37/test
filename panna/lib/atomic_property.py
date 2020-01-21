"""A collection of atomic properties to be used in the code when needed
does not pretend to be complete, it will be updated base on needs
"""

_PERIODIC_TABLE = {'Si': {'mass': 28.085}}


class MisssingPropertyError(ValueError):
    """ just to get a meaningful error
    """


def recover_property(species, prop):
    """ helper:
    return asked property for given species otherwise rise error
    """

    try:
        return _PERIODIC_TABLE[species][prop]
    except KeyError:
        raise MisssingPropertyError(species, prop)
