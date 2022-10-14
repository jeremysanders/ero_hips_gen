#!/usr/bin/env python3

import forkqueue
import glob
import subprocess
import os.path
from astropy.io import fits
import numpy as N

outroot = '/he9srv_local/jsanders/hips/dr1'

def run(x):
    print(*x)
    subprocess.run(x, check=True)

def gen_file():
    fglob = 'EXP_010/e?01_??????_024_Image_c010.fits.gz'

    for d1 in glob.glob(outroot+'/???'):
        for d2 in glob.glob(d1+'/???'):
            for fn in glob.glob(d2+'/'+fglob):
                yield (fn,)

def divfile(imgfn):
    print(imgfn)
    expfn = imgfn.replace('/EXP_010/', '/DET_010/').replace('_Image_', '_ExposureMap_')
    ratefn = imgfn.replace('/EXP_010/', '/DET_010/').replace('_Image_', '_Rate_')
    assert os.path.exists(expfn)

    fin1 = fits.open(imgfn)
    fin2 = fits.open(expfn)
    ratio = (fin1[0].data/fin2[0].data).astype(N.float32)
    fin2[0].data = ratio
    fin2.writeto(ratefn, overwrite=True)

    #run(['FITSMaths3', f'( {imgfn} / {expfn} ).astype(float32)', ratefn])

it = gen_file()
with forkqueue.ForkQueue(numforks=32, ordered=False) as q:
    for res in q.process(divfile, it):
        pass
