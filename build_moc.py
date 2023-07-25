#!/usr/bin/env python3

# Build a Moc.fits file for a HiPS survey

import os.path
from astropy.io import fits
import numpy as N
import glob
import forkqueue

def makeFilename(rootdir, pix, order):
    fname = os.path.join(
        rootdir, 'Norder%i' % order, 'Dir%i' % (pix//10000*10000), 'Npix%i.fits' % pix)
    return fname

def build_moc_inner(rootdir, pix, order, norder, goodpix, badval):
    """norder = norder of directories.
    maxnorder = max norder to build moc
    """

    fname = makeFilename(rootdir, pix, order)
    if not os.path.exists(fname):
        # no good
        return False

    with fits.open(fname) as fin:
        data = fin[0].data
    allbad = N.all((data==badval) | ~N.isfinite(data))
    if allbad:
        # empty pixel
        return False

    if order == norder:
        # exists, so pass up
        return True
    else:
        good0 = build_moc_inner(rootdir, pix*4+0, order+1, norder, goodpix, badval)
        good1 = build_moc_inner(rootdir, pix*4+1, order+1, norder, goodpix, badval)
        good2 = build_moc_inner(rootdir, pix*4+2, order+1, norder, goodpix, badval)
        good3 = build_moc_inner(rootdir, pix*4+3, order+1, norder, goodpix, badval)

        if good0 and good1 and good2 and good3:
            # we're all good in this pixel
            return True
        else:
            b = 4*4**(order+1) + pix*4
            if good0:
                goodpix.append(b+0)
            if good1:
                goodpix.append(b+1)
            if good2:
                goodpix.append(b+2)
            if good3:
                goodpix.append(b+3)
            return False

def build_moc(rootdir, norder, badval=-1e30):
    print('Building MOC for', rootdir)
    goodpix = []
    for pix in range(12):
        print(rootdir, pix)
        good = build_moc_inner(rootdir, pix, 0, norder, goodpix, badval)
        if good:
            goodpix.append(4*4**0 + pix)
    goodpix = N.array(goodpix, dtype=N.int64)
    goodpix.sort()

    col1 = fits.Column(name='UNIQ', format='J', array=goodpix)
    hdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1]))
    hdr = hdu.header
    hdr['EXTNAME'] = 'MOC'
    hdr['MOCVERS'] = '2.0'
    hdr['MOCDIM'] = 'SPACE'
    hdr['ORDERING'] = 'NUNIQ'
    hdr['COORDSYS'] = 'C'
    hdr['PIXTYPE'] = 'HEALPIX'
    hdr['MOCORDER'] = norder
    hdr['MOCTYPE'] = 'IMAGE'
    hdr['MOCTOOL'] = 'build_moc.py v0.1, Jeremy Sanders, MPE'
    hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdulist.writeto(os.path.join(rootdir, 'Moc.fits'), overwrite=True)

def main():
    maxorder = 6
    with forkqueue.ForkQueue(ordered=False, numforks=32) as q:
        args = [(d,maxorder) for d in glob.glob('/hedr_local/erodr/hips/eRASS1_02?_*_c010')]
        for out in q.process(build_moc, args):
            pass

    # for indir in glob.glob('/hedr_local/erodr/hips/eRASS1_02?_*_c010'):
    #     print('Building for', indir)
    #     build_moc(indir, 7, badval=-1e30)

if __name__ == '__main__':
    main()
