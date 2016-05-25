import numpy as np
import itertools
from scintillations.stream import modulate as apply_turbulence
from scintillations.stream import transverse_speed

from streaming.stream import Stream, BlockStream
from streaming.signal import *
import streaming.signal
import logging
from acoustics.signal import impulse_response_real_even

import auraliser.tools
logger = auraliser.tools.create_logger(__name__)

def apply_atmospheric_attenuation(signal, fs, distance, nhop, atmosphere, ntaps, inverse=False, distance_reducer=np.mean):
    """Apply atmospheric attenuation to signal.

    :param distance: Iterable with distances.
    :param fs: Sample frequency.
    :param atmosphere: Atmosphere.
    :param ntaps: Amount of filter taps.
    :param sign: Sign.
    :rtype: :class:`streaming.Stream`

    Compute and apply the attenuation due to atmospheric absorption.
    The attenuation can change with distance. The attenuation is a magnitude-only filter.
    We design a linear-phase filter

    .. note:: The filter delay is compensated by dropping the first `ntaps//2` samples.

    """
    # Partition `distance` into blocks, and reduce with `distance_reducer`.
    distance = distance.blocks(nhop).map(distance_reducer)
    ir = Stream(atmosphere.impulse_response(d, fs, ntaps=ntaps, inverse=inverse) for d in distance)
    signal = convolve_overlap_save(signal, ir, nhop, ntaps)
    signal = signal.samples().drop(int(ntaps//2)) # Linear phase, correct for group delay caused by FIR filter.
    return signal


def apply_reflection_strength(emission, nhop, spectra, effective, ntaps, force_hard):
    """Apply mirror source strength.

    :param signal: Signal.
    :param nblock: Amount of samples per block.
    :param spectra: Spectrum per block.
    :param effective: Whether the source is effective or not.
    :param ntaps: Amount of filter taps.
    :param force_hard: Whether to force a hard ground.
    :returns: Signal with correct strength.

    .. warning:: This operation will cause a delay that may vary over time.

    """
    if effective is not None:
        # We have an effectiveness value for each hop (which is a block of samples)
        emission = BlockStream(map(lambda x,y: x *y, emission.blocks(nhop), effective), nblock=nhop)

    if force_hard:
        logger.info("apply_reflection_strength: Hard ground.")
    else:
        logger.info("apply_reflection_strength: Soft ground.")
        impulse_responses = Stream(impulse_response_real_even(s, ntaps) for s in spectra)
        emission = convolve_overlap_save(emission, impulse_responses, nhop, ntaps)
        # Filter has a delay we need to correct for.
        emission = emission.samples().drop(int(ntaps//2))
    return emission


#def apply_ground_reflection(signal, ir, nblock):
    #"""Apply ground reflection strength.

    #:param signal: Signal before ground reflection strength is applied.
    #:param ir: Impulse response per block.
    #:param nblock: Amount of samples per block.
    #:returns: Signal after reflection strength is applied.
    #:type: :class:`streaming.BlockStream`
    #"""

    #signal = convolve(signal=signal, impulse_responses=ir, nblock=nblock)


def apply_doppler(signal, delay, fs, initial_value=0.0, inverse=False):
    """Apply Doppler shift.

    :param signal: Signal before Doppler shift.
    :param delay: Propagation delay.
    :param fs: Constant sample frequency.
    :returns: Doppler-shifted signal.
    :rtype: :class:`streaming.Stream`

    """
    if inverse:
        delay = delay * -1 # Unary operators are not yet implemented in Stream
    return vdl(signal, times(1./fs), delay, initial_value=initial_value)

def apply_spherical_spreading(signal, distance, inverse=False):#, nblock):
    """Apply spherical spreading.

    :param signal. Signal. Iterable.
    :param distance: Distance. Iterable.
    :param nblock: Amount of samples in block.

    """
    if inverse:
        return signal * distance
    else:
        return signal / distance


def nextpow2(x):
    return int(2**np.ceil(np.log2(x)))

