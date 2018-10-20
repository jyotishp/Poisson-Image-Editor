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
            (x-1, y),
            (x+1, y),
            (x, y-1),
            (x, y+1)
        ]

    def laplacian(self):
        """Evaluate the Laplacian matrix"""
        # Throw exception if called before loading mask
        if not self.masked_pixels == None:
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
                except ValueError:
                    pass

    def __apply_laplacian(self, pixel):
        """Apply Laplcian on the given pixel

        Args:
            pixel: tuple containing the pixel coordinates

        Returns:
            result: Value of the pixel after applying Laplacian
        """
        x, y = pixel
        result = 4 * self.source[i][j]
        result -= self.source[i-1][j]
        result -= self.source[i+1][j]
        result -= self.source[i][j-1]
        result -= self.source[i][j+1]
        return result

    self.__is_inside(self, pixel):
        """Tells if a given point is inside the mask

        Args:
            pixel: tuple containing the pixel coordinates

        Returns:
            True if pixel is inside the mask, False otherwise
        """
        return self.mask[pixel] == 1

    self.__is_edge(self, pixel):
        """Tells if a given point is on an edge
        
        Args:
            pixel: tuple containing the pixel coordinates

        Returns:
            True if pixel is on edge else False
        """
        # False if outside the mask
        if not self.__is_inside(pixel):
            return False
        
        # Check if neighbouring pixels are inside
        # Current pixel is already a part of the mask
        # If any of the nighbouring pixel is not inside, the current pixel
        # should be outside
        for npixel in self.neighbourhood(pixel):
            if not self.__is_inside(pixel):
                return True
        
        return False

    def evaluate_RHS(self):
        """Evaluate RHS of the Poisson equation"""
        # Throw exception if called before lading mask
        if not self.masked_pixels == None:
            raise AttributeError('Mask is not loaded')
        
        # Initialize the matrix
        self.B = np.zeros(len(self.masked_pixels))

        # Evaluate RHS
        for i, pixel in enumerate(self.masked_pixels):
            # Get value after Laplacian
            self.B[i] = self.__apply_laplacian(pixel)

            # If the pixel is on the edge, add target intensity
            if self.__is_edge(pixel):
                for npixel in self.neighbourhood(pixel):
                    self.B[i] += self.target[i]
    