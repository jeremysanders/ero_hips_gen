import numpy as np
cimport cython

# Routine for making a tilemap of an image

# Lookups sky tile from pixel ra,dec in array of tiles
# tileidxs is an output array setting tiles found==1

@cython.boundscheck(False)
@cython.wraparound(False)
def makeTilemap_inner(double[:,::1] ra_img, double[:,::1] dec_img,
                      int[::1] tile_nr,
                      double[::1] ra_min, double[::1] ra_max,
                      double[::1] dec_min, double[::1] dec_max,
                      int[:,::1] tilemap,
                      int[::1] tileidxs):

    cdef int numtiles
    cdef int xw, yw

    cdef int tidx, x, y
    cdef double ra, dec
    cdef int looped

    assert ra_min.shape[0] == ra_max.shape[0]
    assert dec_min.shape[0] == dec_max.shape[0]
    assert ra_min.shape[0] == dec_max.shape[0]
    assert ra_min.shape[0] == tile_nr.shape[0]
    assert ra_min.shape[0] == tileidxs.shape[0]

    assert ra_img.shape[0] == dec_img.shape[0]
    assert ra_img.shape[1] == dec_img.shape[1]
    assert ra_img.shape[0] == tilemap.shape[0]
    assert ra_img.shape[1] == tilemap.shape[1]

    numtiles = ra_min.shape[0]
    yw = ra_img.shape[0]
    xw = ra_img.shape[1]

    tidx = 0
    for y in range(yw):
        for x in range(xw):
            ra = ra_img[y,x]
            dec = dec_img[y,x]
            looped = 0

            while not (ra >= ra_min[tidx] and ra < ra_max[tidx] and dec >= dec_min[tidx] and dec < dec_max[tidx]):
                #print(ra, ra_min[tidx], ra_max[tidx], dec, dec_min[tidx], dec_max[tidx])
                tidx += 1
                if tidx >= numtiles:
                    if looped > 0:
                        # check for infinite loop
                        tidx = -1
                        #print(ra, dec)
                        break
                    looped += 1
                    tidx = 0

            if tidx >= 0:
                tilemap[y,x] = tile_nr[tidx]
                tileidxs[tidx] = 1
            else:
                tilemap[y,x] = -1
                tidx = 0
