#!/usr/bin/env python3

import os
import warnings
import glob

import numpy as N
from astropy.io import fits
from astropy_healpix import HEALPix
from astropy.wcs import WCS, FITSFixedWarning
import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS
import forkqueue

import hips_hdr
import tilemap

import pyximport
pyximport.install()
import image_poly
import binup

outrootdir='/he9srv_local/jsanders/hips/out_img'
#outrootdir='/he9srv_local/jsanders/hips/out_exp'
inrootdir='/he9srv_local/jsanders/hips/dr1'

infnameglob = 'EXP_010/e?01_%06i_024_Image_c010.fits.gz'
#infnameglob = 'DET_010/e?01_%06i_024_ExposureMap_c010.fits.gz'

def makeidximage(imgorder=9):
    """Make an image of length 2**imgorder square with pixels numbered with healpix indexing."""

    def recuridx(out, y0, x0, base, order):
        """Recursively number pixels in output image using nested healpix indexing."""
        if order == 0:
            out[y0,x0] = base
        else:
            recuridx(out, y0*2+1, x0*2+0, base*4+0, order-1)
            recuridx(out, y0*2+0, x0*2+0, base*4+1, order-1)
            recuridx(out, y0*2+1, x0*2+1, base*4+2, order-1)
            recuridx(out, y0*2+0, x0*2+1, base*4+3, order-1)

    imgnside = 2**imgorder
    idx = N.zeros((imgnside, imgnside), dtype=N.int64)

    recuridx(idx, 0, 0, 0, imgorder)
    return idx

def mkdir(dirname):
    """Make a directory without error."""
    try:
        os.mkdir(dirname)
    except OSError:
        pass

def delfile(filename):
    try:
        os.unlink(filename)
    except OSError:
        pass
    return

def extractPixels(norder):
    outroot = os.path.join(outrootdir, 'Norder%i' % norder)
    mkdir(outroot)

    imgorder = 9
    imgsize = 2**imgorder
    imgidx = makeidximage(imgorder)

    healpix = HEALPix(nside=2**(norder+imgorder), order='nested', frame=ICRS())
    nmaps = 12*4**norder  # number of maps

    def process_hp_map(base):
        print('Processing norder=%i, pix=%i/%i' % (norder, base, nmaps))

        outdir = os.path.join(outroot, 'Dir%i' % (base//10000*10000))
        mkdir(outdir)
        outfname = os.path.join(outdir, 'Npix%i.fits' % base)
        delfile(outfname)

        pixidx = imgidx + base*(imgsize**2)
        lon, lat = healpix.healpix_to_lonlat(pixidx)
        lon = lon.to_value(u.deg)
        lat = lat.to_value(u.deg)

        # hack to drop any healpix pixels which don't contain DE data
        coords = SkyCoord(lon, lat, frame='icrs', unit='deg')
        g_l = coords.transform_to('galactic').l.to_value(u.deg)
        sel_de = (g_l > 179.94423568) & (g_l <= 359.94423568)
        if not N.any(sel_de):
            return

        corners = healpix.boundaries_skycoord(pixidx, step=1)

        data = N.zeros((imgsize, imgsize), dtype=N.float32)

        # get tilemap and list of tiles in map
        tmap, tiles = tilemap.makeTilemap(lon, lat)
        totpix = 0
        for tile in tiles:
            fn = glob.glob(os.path.join(
                inrootdir, '%03i' % (tile%1000), '%03i' % (tile//1000), infnameglob%tile))
            if not fn:
                continue

            #print('Reading', fn[0])
            fimg = fits.open(fn[0])
            imgwcs = WCS(fimg[0].header)
            img = N.ascontiguousarray(fimg[0].data.astype(N.float32))
            fimg.close()

            seltile = tmap==tile
            subcorners = corners[seltile.ravel()]
            xs, ys = imgwcs.world_to_pixel(subcorners.ravel())
            xs = N.ascontiguousarray(xs.reshape(subcorners.shape).astype(N.float32))
            ys = N.ascontiguousarray(ys.reshape(subcorners.shape).astype(N.float32))

            # special cython routine to add up pixels in the 4-sided polygons
            polysums, polynpix = image_poly.image_poly_stats(img, xs, ys)
            data[seltile] = polysums/polynpix

            totpix += N.sum(polynpix)

        if totpix > 0:
            data[~sel_de] = N.nan

            hdu = fits.PrimaryHDU(data)

            h = hips_hdr.Tile2HPX(norder, imgsize)
            h.makeHeader(base, hdu.header)
            ff = fits.HDUList([hdu])

            #print('Writing', outfname)
            ff.writeto(outfname)

    maps = range(nmaps)

    # for i in pixels:
    #     process_hp_map(i)

    with forkqueue.ForkQueue(ordered=False, numforks=80, env=locals()) as q:
        for out in q.process(process_hp_map, ((i,) for i in maps)):
            pass

def binupImage(img):
    out = img[::2,::2] + img[1::2,::2] + img[::2,1::2] + img[1::2,1::2]
    return out

def binupPixels(norder):
    """Bin up pixels from higher HIPS order to make a lower one."""

    outroot = os.path.join(outrootdir, 'Norder%i' % (norder-1))
    mkdir(outroot)
    inroot = os.path.join(outrootdir, 'Norder%i' % norder)

    imgorder = 9
    imgsize = 2**imgorder

    def readOrZero(fn):
        if os.path.exists(fn):
            with fits.open(fn) as f:
                img = f[0].data
            return img+0,True # yeah, get rid of big endianness
        else:
            img = N.zeros((imgsize,imgsize), dtype=N.float32) + N.nan
            return img,False

    def binup_tile(idx):

        outdir = os.path.join(outroot, 'Dir%i' % (idx//10000*10000))
        mkdir(outdir)
        outfname = os.path.join(outdir, 'Npix%i.fits' % idx)

        indir = os.path.join(inroot, 'Dir%i' % (idx*4//10000*10000))
        img0,e0 = readOrZero(os.path.join(indir, 'Npix%i.fits' % (idx*4+0)))
        img1,e1 = readOrZero(os.path.join(indir, 'Npix%i.fits' % (idx*4+1)))
        img2,e2 = readOrZero(os.path.join(indir, 'Npix%i.fits' % (idx*4+2)))
        img3,e3 = readOrZero(os.path.join(indir, 'Npix%i.fits' % (idx*4+3)))

        if not (e0 or e1 or e2 or e3):
            delfile(outfname)
            return

        outimg = img0*0
        s = img0.shape[0]
        img0 = binup.binUp(img0)
        img1 = binup.binUp(img1)
        img2 = binup.binUp(img2)
        img3 = binup.binUp(img3)

        outimg[s//2:,:s//2] = img0
        outimg[:s//2,:s//2] = img1
        outimg[s//2:,s//2:] = img2
        outimg[:s//2,s//2:] = img3

        print(outfname)
        hdu = fits.PrimaryHDU(outimg.astype(N.float32))

        h = hips_hdr.Tile2HPX(norder-1, imgsize)
        h.makeHeader(idx, hdu.header)

        ff = fits.HDUList([hdu])
        ff.writeto(outfname, overwrite=True)

    with forkqueue.ForkQueue(ordered=False, numforks=20, env=locals()) as q:
        for out in q.process(binup_tile, ((i,) for i in range(12*4**(norder-1)))):
            pass

def main():
    warnings.simplefilter('ignore', category=FITSFixedWarning)

    skyf = fits.open('SKYMAPS.fits')
    d = skyf['SMAPS'].data
    sky_nr = d['SMAPNR']
    sky_ra_cen = d['RA_CEN']
    sky_ra_min = d['RA_MIN']
    sky_ra_max = d['RA_MAX']
    sky_de_cen = d['DE_CEN']
    sky_de_min = d['DE_MIN']
    sky_de_max = d['DE_MAX']
    owner = d['OWNER']
    skyf.close()

    # for i in range(4,-1,-1):
    #     extractTiles(i)

    maxorder = 6
    extractPixels(maxorder)
    for i in range(maxorder,0,-1):
        binupPixels(i)

if __name__ == '__main__':
    main()
