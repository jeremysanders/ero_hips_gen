import numpy as N
from astropy.io import fits
import pyximport
from astropy.wcs import WCS

pyximport.install()
from tilemap_helper import makeTilemap_inner
import os.path

skyf = fits.open(os.path.join('SKYMAPS.fits'))
d = skyf['SMAPS'].data
sky_nr = d['SMAPNR'] + 0
sky_ra_min = d['RA_MIN'] + 0.0
sky_ra_max = d['RA_MAX'] + 0.0
sky_dec_min = d['DE_MIN'] + 0.0
sky_dec_max = d['DE_MAX'] + 0.0
skyf.close()


def makeTilemap(ra, dec):

    tilemap = N.zeros(ra.shape, dtype=N.intc)
    makeTilemap_inner(ra, dec, sky_nr, sky_ra_min, sky_ra_max, sky_dec_min, sky_dec_max, tilemap)

    return tilemap
