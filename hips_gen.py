#!/usr/bin/env python3

import math
import os
import warnings
import glob
import argparse
import datetime

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

def extractPixels(inrootdir, outrootdir, infnameglob, norder, imgorder, mode='image', numforks=16, comments=None):
    outroot = os.path.join(outrootdir, 'Norder%i' % norder)
    mkdir(outroot)

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
            theglob = os.path.join(
                inrootdir, '%03i' % (tile%1000), '%03i' % (tile//1000), infnameglob%tile)
            fn = glob.glob(theglob)
            if not fn:
                continue

            # get coordinates of pixels
            seltile = tmap==tile
            subcorners = corners[seltile.ravel()]

            if mode == 'image':
                with fits.open(fn[0]) as fimg:
                    imgwcs = WCS(fimg[0].header)
                    img = N.ascontiguousarray(fimg[0].data.astype(N.float32))

                xs, ys = imgwcs.world_to_pixel(subcorners.ravel())
                xs = N.ascontiguousarray(
                    xs.reshape(subcorners.shape).astype(N.float32))
                ys = N.ascontiguousarray(
                    ys.reshape(subcorners.shape).astype(N.float32))

                # special cython routine to add up pixels in the 4-sided polygons
                polysums, polynpix = image_poly.image_poly_stats(img, xs, ys)
                data[seltile] = polysums/polynpix

                totpix += N.sum(polynpix)

            elif mode == 'events':
                with fits.open(fn[0]) as fevt:
                    hdu = fevt['EVENTS']
                    evtra = hdu.data['RA']
                    evtdec = hdu.data['DEC']
                    imgwcs = WCS(fevt[0].header)

                # get number of counts in each pixel
                xs, ys = imgwcs.world_to_pixel(subcorners.ravel())
                xs = N.ascontiguousarray(
                    xs.reshape(subcorners.shape).astype(N.float32))
                ys = N.ascontiguousarray(
                    ys.reshape(subcorners.shape).astype(N.float32))

                era_x, edec_y = imgwcs.world_to_pixel(
                    SkyCoord(evtra, evtdec, unit=u.deg, frame=ICRS))
                era_x = era_x.astype(N.float32)
                edec_y = edec_y.astype(N.float32)

                ra = subcorners.ra.to_value(u.deg)
                dec = subcorners.dec.to_value(u.deg)
                ncts = image_poly.events_poly_stats(era_x, edec_y, xs, ys)

                # convert to cts per sq arcsec for healpix pixel
                data[seltile] = ncts*(1/healpix.pixel_area.to_value(u.arcsec*u.arcsec))

                totpix += 1

        if totpix > 0:
            data[~sel_de] = N.nan

            hdu = fits.PrimaryHDU(data)

            h = hips_hdr.Tile2HPX(norder, imgsize)
            h.makeHeader(base, hdu.header, comments=comments)
            ff = fits.HDUList([hdu])

            #print('Writing', outfname)
            ff.writeto(outfname)

    maps = range(nmaps)
    with forkqueue.ForkQueue(ordered=False, numforks=numforks, env=locals()) as q:
        for out in q.process(process_hp_map, ((i,) for i in maps)):
            pass

def binupImage(img):
    out = img[::2,::2] + img[1::2,::2] + img[::2,1::2] + img[1::2,1::2]
    return out

def binupPixels(outrootdir, norder, hdr_comments):
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

        print('Writing', outfname)
        hdu = fits.PrimaryHDU(outimg.astype(N.float32))

        h = hips_hdr.Tile2HPX(norder-1, imgsize)
        h.makeHeader(idx, hdu.header, comments=hdr_comments)

        ff = fits.HDUList([hdu])
        ff.writeto(outfname, overwrite=True)

    print(f'Binning up sky tiles from order {norder} to make order {norder-1}')
    with forkqueue.ForkQueue(ordered=False, numforks=20, env=locals()) as q:
        for out in q.process(binup_tile, ((i,) for i in range(12*4**(norder-1)))):
            pass

def ensure0(outrootdir, maxorder, imgorder, hdr_comments):
    """Make sure there is a zero pixel image, as this seems to be checked for by viewers."""

    imgsize = 2**imgorder
    data = N.zeros((imgsize, imgsize), dtype=N.float32)
    data[:,:] = N.nan

    def makeZero(order, pixel):
        zerofn = os.path.join(outrootdir, 'Norder%i' % order, 'Dir%i' % (pixel//10000), 'Npix%i.fits' % pixel)
        if not os.path.exists(zerofn):
            hdu = fits.PrimaryHDU(data)
            h = hips_hdr.Tile2HPX(order, imgsize)
            h.makeHeader(pixel, hdu.header, comments=hdr_comments)
            ff = fits.HDUList([hdu])
            print('Writing', zerofn)
            ff.writeto(zerofn)

    for order in range(maxorder+1):
        makeZero(order, 0)

    for order in range(4):
        for pixel in range(12*4**order):
            makeZero(order, pixel)

def makeAllsky(outrootdir, order, imgorder, binorder=3):
    """Combine together tiles to make an all sky image."""

    print('Making all sky for order', order)
    npix = 12*4**order
    xwi = int(math.sqrt(npix))
    ywi = int(math.ceil(npix / xwi))

    size = 2**(imgorder-binorder)
    out = N.zeros((ywi*size, xwi*size), dtype=N.float32)
    out[:,:] = N.nan

    for pix in range(npix):
        fn = os.path.join(outrootdir, 'Norder%i' % order, 'Dir%i' % (pix//10000), 'Npix%i.fits' % pix)
        with fits.open(fn) as fin:
            data = fin[0].data + 0
        for i in range(binorder):
            data = binup.binUp(data)
        xi = pix % xwi
        yi = pix // xwi

        out[out.shape[0]-(yi+1)*size:out.shape[0]-yi*size, xi*size:(xi+1)*size] = data

    outfn = os.path.join(outrootdir, 'Norder%i' % order, 'Allsky.fits')
    fout = fits.HDUList([fits.PrimaryHDU(out)])
    print('Writing', outfn)
    fout.writeto(outfn, overwrite=True)

def main():
    parser = argparse.ArgumentParser(
        prog='hips_gen.py',
        description='Generate eROSITA FITS count, exposure and rate HiPS maps',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--max-order', default=6, type=int, help='Maximum HiPS order to generate')
    parser.add_argument('--img-order', default=9, type=int, help='Order of generated images')
    parser.add_argument('--img-name', default='Image', help='Name of input counts event files')
    parser.add_argument('--exp-name', default='ExposureMap', help='Name of input exposure files')
    parser.add_argument('--proc-ver', default='030', help='Input processing version')
    parser.add_argument('--bands', default='1234', help='Bands to process')
    parser.add_argument('--prefix', default='eROSITA', help='Output directory prefix')
    parser.add_argument('--survey', default='e01', help='Survey to process, e.g. e01, s04')
    parser.add_argument('--procs', default=8, type=int, help='Number of processes')
    parser.add_argument('--products', default='IMG EXP RAT', help='Which products to make')

    parser.add_argument('in_root', help='Input root data archive')
    parser.add_argument('output_root', help='Output root directory')

    args = parser.parse_args()

    # get rid of lots of annoying FITS warnings
    warnings.simplefilter('ignore', category=FITSFixedWarning)

    maxorder = args.max_order
    imgorder = args.img_order
    procver = args.proc_ver
    outroot = args.output_root
    inroot = args.in_root
    outprefix = args.prefix
    bands = args.bands
    survey = args.survey
    products = set(args.products.split())

    hdr_comments = [
        'Generated by hips_gen.py, https://github.com/jeremysanders/ero_hips_gen',
        'on '+ datetime.datetime.now().isoformat(),
        'arguments used:',]

    for kwd in sorted(vars(args)):
        hdr_comments.append(f'  {kwd}: {getattr(args, kwd)}')

    for band in bands:

        todo = []
        if 'EXP' in products:
            todo.append( ('EXP', f'02{band}_{args.img_name}', 'events') )
        if 'IMG' in products:
            todo.append( ('DET', f'02{band}_{args.exp_name}', 'image') )

        # counts and exposure
        outdirs = []
        for inchain, insuffix, mode in todo:

            infnameglob = f'{inchain}_{procver}/{survey[0]}?{survey[1:]}_%06i_{insuffix}_c{procver}.fits*'
            outrootdir = f'{outroot}/{outprefix}_{survey}_{insuffix}_c{procver}'
            outdirs.append(outrootdir)

            if not os.path.exists(outrootdir):
                os.makedirs(outrootdir)

                extractPixels(
                    inroot, outrootdir, infnameglob, maxorder, imgorder, mode=mode,
                    numforks=args.procs, comments=hdr_comments)
                for i in range(maxorder,0,-1):
                    binupPixels(outrootdir, i, hdr_comments)
                ensure0(outrootdir, maxorder, imgorder, hdr_comments)

                makeAllsky(outrootdir, 3, imgorder)

        # compute rates
        outrootdir = f'{outroot}/{outprefix}_{survey}_02{band}_Rate_c{procver}'
        if not os.path.exists(outrootdir) and 'RAT' in products:

            # divide cts and exposure to get rate at max order
            for fname_cts in glob.iglob(os.path.join(outdirs[0], f'Norder{maxorder}', 'Dir*', 'Npix*.fits')):
                p = fname_cts.split('/')
                fname_exp = os.path.join(outdirs[1], '/'.join(p[-3:]))
                fname_rat = os.path.join(outrootdir, '/'.join(p[-3:]))
                os.makedirs(os.path.dirname(fname_rat), exist_ok=True)

                print(fname_rat)
                f1 = fits.open(fname_cts)
                f2 = fits.open(fname_exp)
                rat = (f1[0].data / f2[0].data).astype(N.float32)
                f1[0].data[:,:] = rat
                f1.writeto(fname_rat, overwrite=True)

            # now make lower orders
            for i in range(maxorder,0,-1):
                binupPixels(outrootdir, i, hdr_comments)
            ensure0(outrootdir, maxorder, imgorder, hdr_comments)
            makeAllsky(outrootdir, 3, imgorder)

if __name__ == '__main__':
    main()
