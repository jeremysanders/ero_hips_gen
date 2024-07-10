import numpy as N
from astropy.io import fits
import pyximport
from astropy.wcs import WCS

pyximport.install()
from tilemap_helper import makeTilemap_inner
import os.path

# FIXME: ugly global
fname = os.path.join(os.path.dirname(__file__), 'SKYMAPS_052022.fits')
skyf = fits.open(fname)
d = skyf['SMAPS'].data
sky_owner = d['OWNER']
good = (sky_owner == 0) | (sky_owner == 2)
sky_nr = d['SRVMAP'][good] + 0
sky_ra_min = d['RA_MIN'][good] + 0.0
sky_ra_max = d['RA_MAX'][good] + 0.0
sky_dec_min = d['DE_MIN'][good] + 0.0
sky_dec_max = d['DE_MAX'][good] + 0.0
skyf.close()

def makeTilemap(ra, dec):

    tileidxs = N.zeros(sky_nr.shape, dtype=N.intc)
    tilemap = N.zeros(ra.shape, dtype=N.intc)
    makeTilemap_inner(ra, dec, sky_nr, sky_ra_min, sky_ra_max, sky_dec_min, sky_dec_max, tilemap, tileidxs)

    tilenums = sky_nr[N.where(tileidxs)[0]]
    return tilemap, tilenums

