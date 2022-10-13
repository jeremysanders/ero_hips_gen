#!/usr/bin/env python3

import os
import numpy as N
from astropy.io import fits
from astropy_healpix import HEALPix
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import forkqueue

import hips_hdr
import tilemap

rootdir='/he9srv_local/jsanders/ero_hips_gen'

def recuridx(out, y0, x0, base, order):
    """Recursively number pixels in output image using nested healpix indexing."""
    if order == 0:
        out[y0,x0] = base
    else:
        recuridx(out, y0*2+1, x0*2+0, base*4+0, order-1)
        recuridx(out, y0*2+0, x0*2+0, base*4+1, order-1)
        recuridx(out, y0*2+1, x0*2+1, base*4+2, order-1)
        recuridx(out, y0*2+0, x0*2+1, base*4+3, order-1)

def makeidximage(imgorder=9):
    imgnside = 2**imgorder
    idx = N.zeros((imgnside, imgnside), dtype=N.int64)

    recuridx(idx, 0, 0, 0, imgorder)
    return idx

def mkdir(dirname):
    try:
        os.mkdir(dirname)
    except OSError:
        pass

def extractTiles(norder):
    outroot = os.path.join(rootdir, 'Norder%i' % norder)
    mkdir(outroot)

    imgorder = 9
    imgsize = 2**imgorder
    imgidx = makeidximage(imgorder)

    healpix = HEALPix(nside=2**(norder+imgorder), order='nested')

    def process_tile(base):

        outdir = os.path.join(outroot, 'Dir%i' % (base//10000*10000))
        mkdir(outdir)

        pixidx = imgidx + base*(imgsize**2)
        lon, lat = healpix.healpix_to_lonlat(pixidx)
        lon = lon.to_value(u.deg)
        lat = lat.to_value(u.deg)

        idxs = tilemap.makeTilemap(lon, lat)
        data = idxs.astype(N.float32)
        hdu = fits.PrimaryHDU(data)

        h = hips_hdr.Tile2HPX(norder, imgsize)
        h.makeHeader(base, hdu.header)
        ff = fits.HDUList([hdu])
        outfname = os.path.join(outdir, 'Npix%i.fits' % base)
        print('Writing', outfname)
        ff.writeto(outfname, overwrite=True)

    with forkqueue.ForkQueue(ordered=False, numforks=16, env=locals()) as q:
        for out in q.process(process_tile, ((i,) for i in range(12*4**norder))):
            pass

def binupImage(img):
    out = img[::2,::2] + img[1::2,::2] + img[::2,1::2] + img[1::2,1::2]
    return out

def binupTiles(norder):
    """Bin up tiles from higher order to lower one."""
    outroot = os.path.join(rootdir, 'Norder%i' % (norder-1))
    mkdir(outroot)
    inroot = os.path.join(rootdir, 'Norder%i' % norder)

    imgorder = 9
    imgsize = 2**imgorder

    def binup_tile(idx):
        indir = os.path.join(inroot, 'Dir%i' % (idx*4//10000*10000))

        with fits.open( os.path.join(indir, 'Npix%i.fits' % (idx*4+0)) ) as f:
            img0 = f[0].data
        with fits.open( os.path.join(indir, 'Npix%i.fits' % (idx*4+1)) ) as f:
            img1 = f[0].data
        with fits.open( os.path.join(indir, 'Npix%i.fits' % (idx*4+2)) ) as f:
            img2 = f[0].data
        with fits.open( os.path.join(indir, 'Npix%i.fits' % (idx*4+3)) ) as f:
            img3 = f[0].data

        outimg = img0*0
        s = img0.shape[0]
        img0 = binupImage(img0)
        img1 = binupImage(img1)
        img2 = binupImage(img2)
        img3 = binupImage(img3)

        outimg[s//2:,:s//2] = img0
        outimg[:s//2,:s//2] = img1
        outimg[s//2:,s//2:] = img2
        outimg[:s//2,s//2:] = img3
        outimg *= 0.25

        outdir = os.path.join(outroot, 'Dir%i' % (idx//10000*10000))
        mkdir(outdir)

        outfname = os.path.join(outdir, 'Npix%i.fits' % idx)

        print(outfname)
        hdu = fits.PrimaryHDU(outimg.astype(N.float32))

        h = hips_hdr.Tile2HPX(norder-1, imgsize)
        h.makeHeader(idx, hdu.header)

        ff = fits.HDUList([hdu])
        ff.writeto(outfname, overwrite=True)

    with forkqueue.ForkQueue(ordered=False, numforks=1, env=locals()) as q:
        for out in q.process(binup_tile, ((i,) for i in range(12*4**(norder-1)))):
            pass

def main():
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

    extractTiles(4)
    for i in range(4,0,-1):
        binupTiles(i)

if __name__ == '__main__':
    main()
