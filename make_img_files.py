#!/usr/bin/env python3

import os.path
import glob
import argparse

import forkqueue
import numpy as N
from astropy.io import fits
from PIL import Image, ImageFilter

from scipy.ndimage import gaussian_filter

def make_img(infname, minval, maxval, mode, smooth):
    print('Reading', infname)
    outfname = infname.replace('.fits', '.png')
    with fits.open(infname) as fin:
        img = fin[0].data + 0

    # rvals = N.arange(256)
    # scalevals = N.log(rvals*0.1+1)
    # maxscale = scalevals[1]
    # scalevals /= (maxscale*255)

    x = N.clip((img-minval)/(maxval-minval), 0, 1)[::-1,:]
    #a = 1000
    a = 255/10 # Aladin value
    a = 50

    if smooth>0 and 'Allsky' not in infname:
        # hack not to smooth smallest scales
        x = gaussian_filter(x, sigma=smooth)

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
    imgdata.putalpha(alpha)
    imgdata.save(outfname)

def getstats(indir, minv, maxv):
    infiles = glob.glob(os.path.join(indir, 'Norder4', 'Dir*', 'Npix*.fits'))
    pixels = []
    totpix = 0
    for fn in infiles:
        with fits.open(fn) as fin:
            data = fin[0].data
            data = data[N.isfinite(data)]
            totpix += len(data)
            pixels.append(data)

    pixels = N.concatenate(pixels)
    minval, maxval = N.percentile(pixels, [1e-2, 99.95])
    return minval, maxval

def make_img_files(indir, minval, maxval, mode, smooth, numforks):

    with forkqueue.ForkQueue(numforks=numforks, ordered=False) as q:
        for dn in glob.glob(os.path.join(indir, 'Norder*')):
            filenames = (
                glob.glob(os.path.join(dn, 'Dir*', 'Npix*.fits')) +
                glob.glob(os.path.join(dn, 'Allsky.fits'))
            )

            args = ((x,minval,maxval,mode,smooth) for x in filenames)
            for x in q.process(make_img, args):
                pass

def main():
    parser = argparse.ArgumentParser(
        prog='make_img_files.py',
        description='Make eROSITA HiPS image files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--scaling', default='linear', choices=('linear', 'sqrt', 'log'),
        help='Scaling function from values to brightness')
    parser.add_argument(
        '--min', default=0.01, type=float,
        help='Minimum value or percentile')
    parser.add_argument(
        '--max', default=99.95, type=float,
        help='Maximum value or percentile')
    parser.add_argument(
        '--percentile', action='store_true',
        help='Use percentile rather than values')
    parser.add_argument(
        '--smooth', default=1, type=float,
        help='Sigma of smoothing function')
    parser.add_argument('--procs', default=8, type=int, help='Number of processes')
    parser.add_argument('root_dir', help='HiPS root directory')

    args = parser.parse_args()

    minv = args.min
    maxv = args.max
    if args.percentile:
        print('Reading statistics')
        minv, maxv = getstats(args.root_dir, minv, maxv)
        print(f'Using range of {minv} to {maxv}')

    # store range for later use
    with open(os.path.join(args.root_dir, 'rng.dat'), 'w') as fout:
        print(minv, maxv, args.scaling, file=fout)

    make_img_files(args.root_dir, minv, maxv, args.scaling, args.smooth, args.procs)

if __name__ == '__main__':
    main()

# for band in range(1, 7+1):
#     make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_02{band}_ExposureMap_c010', 0, 1000, 'sqrt', False)

# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_021_Rate_c010', 0, 1e-5, 'log', True)
# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_022_Rate_c010', 5e-7, 2e-5, 'log', True)
# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_023_Rate_c010', 6e-7, 8e-6, 'log', True)
# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_024_Rate_c010', 1e-6, 3e-5, 'log', True)
# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_025_Rate_c010', 1e-6, 3e-5, 'log', True)
# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_026_Rate_c010', 2e-7, 1.3e-5, 'log', True)
# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_027_Rate_c010', 3e-7, 6e-6, 'log', True)

# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_021_Image_c010', 0, 0.0075, 'log', True)
# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_022_Image_c010', 0, 0.0135, 'log', True)
# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_023_Image_c010', 0, 0.0045, 'log', True)
# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_024_Image_c010', 0, 0.018, 'log', True)
# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_025_Image_c010', 0, 0.006, 'log', True)
# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_026_Image_c010', 0, 0.009, 'log', True)
# make_img_files(f'/hedr_local/erodr/hips/hips/eRASS1_027_Image_c010', 0, 0.006, 'log', True)
