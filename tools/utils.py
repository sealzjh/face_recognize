# -*- encoding: utf8 -*-
import os

EEXIST = 17


def makedir_exist_ok(dirpath):
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == EEXIST:
            pass
        else:
            raise

