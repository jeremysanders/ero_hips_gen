#cython: language_level=3

import numpy as np
cimport cython
from libc.math cimport isfinite

@cython.boundscheck(False)
@cython.wraparound(False)
def binUp(float[:,::1] inimg):
    cdef int xw, yw, xwh, ywh
    cdef int x, y, xi, yi, n
    cdef float v, t
    cdef float nan

    xw = inimg.shape[1]
    yw = inimg.shape[0]
    assert xw % 2 == 0 and yw % 2 == 0

    xwh = xw//2
    ywh = yw//2

    outimg = np.zeros((ywh, xwh), dtype=np.float32)
    cdef float[:,::1] outimg_v = outimg

    nan = np.nan
    for y in range(xwh):
        for x in range(xwh):
            n = 0
            t = 0
            for yi in range(2):
                for xi in range(2):
                    v = inimg[y*2+yi,x*2+xi]
                    if isfinite(v):
                        t += v
                        n += 1
            if n == 0:
                outimg_v[y,x] = nan
            else:
                outimg_v[y,x] = t/n

    return outimg
