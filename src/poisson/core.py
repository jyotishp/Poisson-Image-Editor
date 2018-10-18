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
    masked_pixels = None

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
        # Cast mask image as a 3 channel image
        # 3 channel is needed to handle mask read as 3 channel image
        self.mask = np.atleast_3d(self.mask).astype(np.float)
        # Normalize the mask image
        self.mask *= 1/255
        # Binarize the mask
        # Set the values that are not 1 to zero for sanitization
        self.mask[self.mask != 1] = 0
        # Trim 3 channels to 1 channel
        self.mask = self.mask[:,:,0]
        # Extract the indices from the mask
        self.masked_pixels = self.mask.nonzero()
        self.masked_pixels = zip(self.masked_pixels[0], self.masked_pixels[1])

    def neighbourhood(self, pixel):
        """Get neighbouring four pixel locations for a given pixel location

        Args:
            pixel: tuple containing the pixel coordinates

        Returns:
            An array of tuples, each having the x and y coordinates of
            neighbouring pixels
        """
        x, y = pixel
        return [
            (x-1, y-1),
            (x-1, y),
            (x+1, y),
            (x+1, y+1)
        ]

    def laplacian(self):
        """Evaluate the Laplacian matrix"""

        # Throw exception if called before loading mask
        if not self.masked_pixels:
            raise AttributeError('Mask is not loaded')

        # Initialize the laplacian matrix
        indices_len = len(self.masked_pixels)
        self.A = np.zeros((indices_len, indices_len))

        # Evaluate the Laplacian matrix
        for i, pixel in enumerate(self.masked_pixels):
            # Diagonal elements in Laplacian matrix
            self.A[i][i] = 4

            # Fill the neighbouring pixel locations
            for npixel in self.neighbourhood(pixel):
                # Check if the neighbouring pixel is in the masked pixels
                try:
                    j = self.masked_pixels.index(npixel)
                    self.A[i][j] = -1
                except ValueError as err:
                    pass
