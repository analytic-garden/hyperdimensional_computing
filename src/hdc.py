#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hdc.py - Functions for (SLOW) hyperdimensional computing
author: Bill Thompson
license: GPL 3
copyright: 2023-04-21
"""

import numpy as np

def hdv(N: int = 10000) -> np.ndarray:
    """Create a random high dimensions vector.

    Parameters
    ----------
    N : int, optional
        size of the high-D array, by default 10000

    Returns
    -------
    np.ndarray
        a random vector of -1, 1.
    """
    return np.random.choice([-1, 1], size = N)

def bundle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bundle two hdvs.

    Parameters
    ----------
    x : np.ndarray
        an hdv
    y : np.ndarray
        an hdv

    Returns
    -------
    np.ndarray
        an hdv combining x and y.

    Requires
    --------
    x and y must be the same length
    """
    return np.sign(x + y)
    
def bind(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bind two hdvs into a new hdv that is orthogonal to x and y.

    Parameters
    ----------
    x : np.ndarray
        an hdv
    y : np.ndarray
        an hdv

    Returns
    -------
    np.ndarray
        an hdv combining x and y.

    Requires
    --------
    x and y must be the same length
    """
    return x * y

def cos(x: np.ndarray, y: np.ndarray) -> np.float64:
    """Calculate the similarity between two hdvs.

    Parameters
    ----------
    x : np.ndarray
        an hdv
    y : np.ndarray
        an hdv
    
    Returns
    -------
    np.float64
        The cosine of the angle beween x and y.

    Requires
    --------
    x and y must be the same length
    """
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Perform an circular shift of an hdv by k steps

    Parameters
    ----------
    x : np.ndarray
        an hdv
    k : int, optional
        the number of elemnets to shift, k is negative to shift right, by default 1

    Returns
    -------
    np.ndarray
        the hdv shifted by k elements
    """
    return np.roll(x, k)

