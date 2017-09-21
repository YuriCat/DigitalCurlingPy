# -*- coding: utf-8 -*-
# physics.py

import math, copy
import numpy as np

import dc
import dccpp

# use c++ simulator via pybind11

def simulate(board, move):
    # call cpp function
    err = dccpp.simulate(board.stone, board.turn, move)
    return err