#!/usr/bin/env python3

import os.path
import glob
import os

import forkqueue
import numpy as N
from astropy.io import fits
from PIL import Image, ImageFilter

from scipy.ndimage import gaussian_filter

def make_img(outdir, infname, dirs, minvals, maxvals, mode, smooth):

    channels = []
    for dn, minv, maxv in zip(dirs, minvals, maxvals):
        fname = os.path.join(dn, infname)
        print('Reading', fname)

        with fits.open(fname) as fin:
            img = fin[0].data + 0

        x = N.clip((img-minv)/(maxv-minv), 0, 1)[::-1,:]
        #a = 1000
        a = 255/10 # Aladin value
        a = 50

        if smooth and 'Allsky' not in infname:
            # hack not to smooth smallest scales
            x = gaussian_filter(x, sigma=1)

        if mode == 'log':
            outval = N.log(a*x+1)/N.log(a)
        elif mode == 'linear':
            outval = x
        elif mode == 'sqrt':
            outval = N.sqrt(x)

        outval = N.clip(outval, 0, 1)
        imgval = (255*outval).astype(N.uint8)

        alphadata = (N.where(N.isfinite(img), 255, 0).astype(N.uint8))[::-1,:]
        #imgval[~N.isfinite(img)[::-1,:]] = 0
        alpha = Image.fromarray(alphadata)

        imgdata = Image.fromarray(imgval)
        channels.append(imgdata)

    outfname = os.path.join(outdir, infname.replace('.fits', '.png'))
    os.makedirs(os.path.dirname(outfname), exist_ok=True)
    rgb = Image.merge("RGB", channels)
    rgb.save(outfname)


def make_img_files(outdir, indirs, minvals, maxvals, mode, smooth):
    cwd = os.path.abspath(os.getcwd())
    with forkqueue.ForkQueue(numforks=64, ordered=False) as q:
        os.chdir(indirs[0])
        filenames = (
            glob.glob(os.path.join('Norder*', 'Dir*', 'Npix*.fits')) +
            glob.glob(os.path.join('Norder*', 'Allsky.fits'))
        )
        os.chdir(cwd)

        args = ((outdir,x,indirs,minvals,maxvals,mode,smooth) for x in filenames)
        for x in q.process(make_img, args):
            pass

make_img_files(
    '/hedr_local/erodr/hips/eRASS1_RGB_Rate_c010',
    ['/hedr_local/erodr/hips/eRASS1_025_Rate_c010',
     '/hedr_local/erodr/hips/eRASS1_026_Rate_c010',
     '/hedr_local/erodr/hips/eRASS1_027_Rate_c010'],
    [8e-8, 3e-7, 7e-7],
    [1e-5, 1.3e-5, 6e-6],
    'log', True,
)
