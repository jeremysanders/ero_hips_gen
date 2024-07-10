import math

# routines taken from Aladin: see Tile2HPX.java and ZOrderCurve2DImpls.java (ZOC_VMSB_LOOKUP_INT)

def ij2i(ij):
    # note: implict truncation from long->int in original routine
    return ij & 0xffffffff

def ij2j(ij):
    return ij >> 32

def hash2ij(h):
    h = (0x2222222222222222 & h) <<  1 | (0x4444444444444444 & h) >>  1 | (0x9999999999999999 & h)
    h = (0x0C0C0C0C0C0C0C0C & h) <<  2 | (0x3030303030303030 & h) >>  2 | (0xC3C3C3C3C3C3C3C3 & h)
    h = (0x00F000F000F000F0 & h) <<  4 | (0x0F000F000F000F00 & h) >>  4 | (0xF00FF00FF00FF00F & h)
    h = (0x0000FF000000FF00 & h) <<  8 | (0x00FF000000FF0000 & h) >>  8 | (0xFF0000FFFF0000FF & h)
    h = (0x00000000FFFF0000 & h) << 16 | (0x0000FFFF00000000 & h) >> 16 | (0xFFFF00000000FFFF & h)
    return h

class Tile2HPX:
    def __init__(self, tileOrder, inTileNside):
        self.order = tileOrder
        self.inNside = inTileNside;
        self.nsideTile = 1 << self.order
        self.nsidePix = self.nsideTile * self.inNside
        self.twiceDepth = tileOrder << 1
        self.xyMask = (1 << self.twiceDepth) - 1

    def centre(self, h):

        # Pull apart the hash elements
        d0h = h >> self.twiceDepth
        h = hash2ij(h & self.xyMask)
        iInD0h = ij2i(h)
        jInD0h = ij2j(h)
        # Compute coordinates from the center of the base pixel with x-axis = W-->E, y-axis = S-->N
        lInD0h = iInD0h - jInD0h
        hInD0h = iInD0h + jInD0h - (self.nsideTile - 1)
        # Compute coordinates of the base pixel in the projection plane
        d0hBy4Quotient = d0h >> 2
        d0hMod4 = d0h - (d0hBy4Quotient << 2)
        hD0h = 1 - d0hBy4Quotient
        lD0h = d0hMod4 << 1
        if hD0h == 0 and (lD0h == 6 or (lD0h == 4 and lInD0h > 0)): # case equatorial region
            lD0h -= 8
        elif hD0h != 0:
            lD0h += 1
            if lD0h > 3:
                lD0h -= 8

        # Finalize computation
        xy0 = (math.pi/4) * (lD0h + lInD0h / self.nsideTile)
        xy1 = (math.pi/4) * (hD0h + hInD0h / self.nsideTile)
        return xy0, xy1

    def makeHeader(self, tileIpix, hdr, comments=None):
        centreX, centreY = self.centre(tileIpix)
        centreX *= 180/math.pi
        centreY *= 180/math.pi

        scale = 45 / self.nsidePix
        crPix1 = +((self.inNside + 1) / 2.0) - 0.5 * (-centreX / scale + centreY / scale)
        crPix2 = +((self.inNside + 1) / 2.0) - 0.5 * (-centreX / scale - centreY / scale)

        hdr['CRPIX1'] = round(crPix1, 1)
        hdr['CRPIX2'] = round(crPix2, 1)
        hdr['CD1_1'] = -scale
        hdr['CD1_2'] = -scale
        hdr['CD2_1'] = +scale
        hdr['CD2_2'] = -scale
        hdr['CTYPE1'] = 'RA---HPX'
        hdr['CTYPE2'] = 'DEC--HPX'
        hdr['CRVAL1'] = 0.
        hdr['CRVAL2'] = 0.
        hdr['PV2_1'] = 4
        hdr['PV2_2'] = 3
        hdr['NPIX'] = tileIpix
        hdr['ORDER'] = self.order

        if comments:
            for cmt in comments:
                hdr['COMMENT'] = cmt

"""
from astropy.io import fits
import glob
import re

order = 7
t = Tile2HPX(order, 512)

for fn in glob.glob(f'/he9srv_local/jsanders/hips/expall_out/Norder{order}/Dir*/Npix*.fits'):
    print(fn)
    hdr = {}

    m = re.search(r'Dir([0-9]+)/Npix([0-9]+).fits', fn)
    idx = int(m.group(2))
    t.makeHeader(idx, hdr)

    f = fits.open(fn)
    p1 = f[0].header['CRPIX1']
    p2 = f[0].header['CRPIX2']

    assert p1==hdr['CRPIX1']
    assert p2==hdr['CRPIX2']

    print(p1, p2)
"""
