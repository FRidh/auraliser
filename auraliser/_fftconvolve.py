"""
This module contains a modified fftconvolve which, when given N-dimensional arrays, 
doesn't perform the N-dimensional FFT but instead always performs a 1-dimensional FFT.
"""

from __future__ import division, print_function, absolute_import

try:
    from pyfftw.interfaces.numpy_fft import rfft, irfft       # Performs much better than numpy's fftpack
    import pyfftw
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import rfft, irfft

from numpy.fft import rfft, irfft

from numpy import (allclose, angle, arange, argsort, array, asarray,
                   atleast_1d, atleast_2d, cast, dot, exp, expand_dims,
                   iscomplexobj, isscalar, mean, ndarray, newaxis, ones, pi,
                   poly, polyadd, polyder, polydiv, polymul, polysub, polyval,
                   prod, product, r_, ravel, real_if_close, reshape,
                   roots, sort, sum, take, transpose, unique, where, zeros)
import numpy as np

from scipy.signal.signaltools import _check_valid_mode_shapes, _next_regular


def fftconvolve1D(in1, in2, mode="full"):
    """Convolve two two-dimensional arrays using the one-dimensional FFT.

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.

    The convolution is done per row.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`;
        if sizes of `in1` and `in2` are not equal then `in1` has to be the
        larger array.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    """
    in1 = asarray(in1)
    in2 = asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif not in1.ndim == in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return array([])

    s1 = array(in1.shape)
    s2 = array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complex) or
                      np.issubdtype(in2.dtype, np.complex))
    shape = s1 + s2 - 1

    if mode == "valid":
        _check_valid_mode_shapes(s1, s2)

    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = _next_regular(int(shape[-1]))
    fslice = slice(None)
    if not complex_result:
        ret = irfft(rfft(in1, fshape) *
                     rfft(in2, fshape), fshape)[fslice].copy()
        ret = ret.real
    else:
        ret = ifft(fft(in1, fshape) * fft(in2, fshape))[fslice].copy()

    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        shape_out = s1.copy()
        shape_out[-1] = s1[-1] - s2[-1] + 1
        return _centered(ret, shape_out)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")  
        
def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = asarray(newsize)
    currsize = array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

