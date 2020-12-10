import logging
import warnings

from . import QQTensor, QQLayer, QQLoss, QQActivation, QQOptimizer, QQInitializer
from .ops import QQOperator
from .QQSetting import *

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)

warnings.formatwarning = warning_on_one_line
logging.getLogger().setLevel(TENSORQQ_LOGGER_LEVEL)

QQOperator.register()