"""
    Utility functions for evaluating Poisson solver
"""

#!/usr/env/bin python

import numpy as np

def direct_patch(src, mask, target):
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            if mask[i][j] == 1:
                target[i][j] = src[i][j]
    return target