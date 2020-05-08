# -*- coding: utf-8 -*-

from .default import *

try:
    from .local import *
except ImportError:
    pass
