"""
    This module houses the Poisson class that is used for seamless
    image blending.
"""

#!/usr/env/bin python

# For backwards compactibility
from __future__ import print_function

# Import system modules
import os

# Import installed modules
import numpy as np
import cv2


class Poisson():
    """Class that holds definitions for Poisson Editing tasks

    Attributes:

    Methods:

    """
    INSIDE = -1
    EDGE = 0
    OUTSIDE = 1

    def __init__(self, src, target, mask):
        """Constructor to read input images during object creation

        Args:
            src: Source/Foreground image as a numpy array
            target: Target/Background image as a numpy array
            mask: Mask image as a numpy array. Mask image should be a
                monochannel image with binary values (0, 1)

        Returns:
            A python object of the class Poisson
        """
        self.source = src
        self.target = target
        self.mask = mask

    def __read_img_from_path(self, path):
        """Private method to load images from given path

        Args:
            path: Path to the image to load

        Raises:
            IOError: The specified image path does not exist
        """
        # Throw error if the path doesn't exist
        if not os.path.exists:
            raise IOError('No file named ' + path)

        # Return loaded image
        return cv2.imread(path)

    def read_source(self, src_path):
        """Load source/foreground image"""
        self.source = self.__read_img_from_path(src_path)

    def read_target(self, target_path):
        """Load target/background image"""
        self.target = self.__read_img_from_path(target_path)

    def read_mask(self, mask_path):
        """Load mask image"""
        self.mask = self.__read_img_from_path(mask_path)

    def mask_indices(self, mask):
        pass
