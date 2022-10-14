#!/usr/bin/env python3

import forkqueue
import glob
import subprocess
import os.path
import forkqueue

outroot = '/he9srv_local/jsanders/hips/dr1'
base = '/ero_archive/products/survey/proprietary'

def run(x):
    print(*x)
    subprocess.run(x, check=True)

def copyfile(fn):
    outdir = outroot+'/'+os.path.dirname(fn)
    run([
        'ssh', 'he9srv', 'mkdir', '-p', outdir,
    ])
    run([
        'scp', base+'/'+fn, 'he9srv:'+outdir
    ])

def gen_file():
    fglob = 'EXP_010/e?01_??????_024_Image_c010.fits.gz'
    fglob = 'DET_010/e?01_??????_024_ExposureMap_c010.fits.gz'

    for d1 in glob.glob(base+'/???'):
        for d2 in glob.glob(d1+'/???'):
            for fn in glob.glob(d2+'/'+fglob):
                fnout = fn[len(base)+1:]
                yield (fnout,)

it = gen_file()
with forkqueue.ForkQueue(numforks=8, ordered=False) as q:
    for res in q.process(copyfile, it):
        pass

# for x in gen_file():
#     copyfile(x)

