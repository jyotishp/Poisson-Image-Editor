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
# from numpy.linalg import lstsq as solver
from scipy.sparse.linalg import cg as solver
import cv2
from scipy import sparse


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
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

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
        self.mask_shape = self.mask.shape
        self.mask = self.mask[:, :, 0]

        # Extract the indices from the mask
        self.masked_pixels = self.mask.nonzero()
        tmp = zip(self.masked_pixels[0], self.masked_pixels[1])
        
        # Convert zip to list
        # Python 3 doesn't support len() on finite generators
        self.masked_pixels = []
        for pixel in tmp:
            self.masked_pixels.append(pixel)

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
        if self.masked_pixels == None:
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
        self.A = sparse.csr_matrix(self.A)

    def __apply_laplacian(self, layer, pixel):
        """Apply Laplcian on the given pixel

        Args:
            layer: One channel array from source image
            pixel: tuple containing the pixel coordinates

        Returns:
            result: Value of the pixel after applying Laplacian
        """
        i, j = pixel
        result = 4 * layer[i][j]
        result -= layer[i-1][j]
        result -= layer[i+1][j]
        result -= layer[i][j-1]
        result -= layer[i][j+1]
        return result

    def __is_inside(self, pixel):
        """Tells if a given point is inside the mask

        Args:
            pixel: tuple containing the pixel coordinates

        Returns:
            True if pixel is inside the mask, False otherwise
        """
        return self.mask[pixel] == 1

    def __is_edge(self, pixel):
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
            if not self.__is_inside(npixel):
                return True

        return False

    def evaluate_RHS(self, src_layer, target_layer):
        """Evaluate RHS of the Poisson equation

        Args:
            src_layer: One channel array from the source image
            target_layer: One channel array from the target image
        """
        # Throw exception if called before lading mask
        if self.masked_pixels == None:
            raise AttributeError('Mask is not loaded')

        # Initialize the matrix
        self.B = np.zeros(len(self.masked_pixels))

        # Evaluate RHS
        for i, pixel in enumerate(self.masked_pixels):
            # Get value after Laplacian
            self.B[i] = self.__apply_laplacian(src_layer, pixel)

            # If the pixel is on the edge, add target intensity
            if self.__is_edge(pixel):
                for npixel in self.neighbourhood(pixel):
                    if not self.__is_inside(npixel):
                        self.B[i] += target_layer[npixel]

    def __check_input_dimensions(self):
        """Check if input image dimensions are the same"""
        if self.mask_shape == self.source.shape == self.target.shape:
            return True

        return False

    def seamless_blend(self):
        """Perform blending with given images

        Returns:
            result: Output image after blending

        Raises:
            ValueError: Throw exception if input images are not of
                same dimensions
        """
        # Throw error if the input dimensions are not same
        if not self.__check_input_dimensions():
            raise ValueError('Input dimensions of all images should be same')

        # Get number of channels
        channels = self.source.shape[-1]
        result = np.copy(self.target).astype(int)

        self.laplacian()

        # Replace with new intensities
        if len(self.source.shape) == 2:
            self.evaluate_RHS(self.source, self.target)

            # Solve the system of equations
            u = solver(self.A, self.B)

            for i, pixel in enumerate(self.masked_pixels):
                result[pixel] = u[0][i]

        else:
            for i in range(channels):
                self.evaluate_RHS(self.source[:,:,i], self.target[:,:,i])

                # Solve the system of equations
                u = solver(self.A, self.B)[0].round()
                self.debug = u
                tmp = np.copy(self.target[:,:,i]).astype(int)
                self.debug2 = tmp

                for j, pixel in enumerate(self.masked_pixels):
                    tmp[pixel] = u[j]

                result[:,:,i] = tmp

        return result
