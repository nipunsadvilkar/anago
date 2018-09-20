import os.path as op
import os
import sys

from configparser import ConfigParser


def get_config():
    """
    Read configuration file and returns it as an ordered Dict'

    Returns
    ------
    config : Ordered Dict
        contains parameter values for each file

    """
    config = ConfigParser()
    config.read(op.join(op.dirname(op.abspath(__file__)), 'config.ini'))
    return config
