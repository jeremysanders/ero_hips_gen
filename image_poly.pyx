import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int get_side(float ax, float ay, float bx, float by):
    cdef float dot

    dot = ax*by-ay*bx
    if dot < 0:
        return -1
    elif dot > 0:
        return 1
    else:
        return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double fixup_x(double x, int fixup):
    if fixup and x>180:
        return x-360
    else:
        return x

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def image_poly_stats(float[:,::1] img not None,
                     float[:,::1] px not None,
                     float[:,::1] py not None):

    """Get total value inside polygons inside an image.

    img: 2D contiguous array
    px: 2D array [[p1_x1, p1_x2...], [p2_x1, p2_x2...]]
    py: 2D array [[p1_y1, p1_y2...], [p2_y1, p2_y2...]]

    Returns ([sum_p1, sum_p2...], [npix_p1, npix_p2...])
    """

    cdef int yw, xw, x, y, pt
    cdef int nitems, npts, npol, npolys
    cdef int dirn
    cdef float fx, fy

    cdef float minx, maxx, miny, maxy
    cdef int iminx, imaxx, iminy, imaxy

    cdef float seg_x, seg_y, del_x, del_y
    cdef int side

    cdef float tot
    cdef int npix

    assert px.shape[0]==py.shape[0] and px.shape[1]==py.shape[1]
    npolys = px.shape[0]
    npts = px.shape[1]
    yw = img.shape[0]
    xw = img.shape[1]

    assert npts>0
    assert npolys>0

    tot_out = np.zeros((npolys,), dtype=np.float32)
    npix_out = np.zeros((npolys,), dtype=np.int32)

    cdef float[::1] tot_out_v = tot_out
    cdef int[::1] npix_out_v = npix_out

    for npol in range(npolys):

        # find max of polygon
        minx=1e20
        miny=1e20
        maxx=-1e20
        maxy=-1e20
        for pt in range(npts):
            minx = min(minx, px[npol,pt])
            maxx = max(maxx, px[npol,pt])
            miny = min(miny, py[npol,pt])
            maxy = max(maxy, py[npol,pt])
        # integer range which encompasses the range
        iminx = max(<int>(minx)-1, 0)
        imaxx = min(<int>(maxx)+3, xw-1)
        iminy = max(<int>(miny)-1, 0)
        imaxy = min(<int>(maxy)+3, yw-1)

        tot = 0
        npix = 0

        # iterate over pixels in polygon
        for y in range(iminy, imaxy):
            for x in range(iminx, imaxx):
                # middle of pixel
                fy = y+0.5
                fx = x+0.5

                dirn = 0
                for pt in range(npts):
                    seg_y = py[npol,(pt+1)%npts] - py[npol,pt]
                    seg_x = px[npol,(pt+1)%npts] - px[npol,pt]
                    del_y = fy - py[npol,pt]
                    del_x = fx - px[npol,pt]

                    side = get_side(seg_x, seg_y, del_x, del_y)
                    if dirn == 0:  # uninitialised
                        if side != 0:
                            # initialise
                            dirn = side
                        else:
                            # we're at the edge, so say outside
                            dirn = -2
                            break
                    else:
                        if side != dirn:
                            # side was different from last time, so outside
                            dirn = -2
                            break

                if dirn != -2:
                    # inside
                    tot += img[y,x]
                    npix += 1

                    #debug
                    #img[y, x] = npol

        tot_out_v[npol] = tot
        npix_out_v[npol] = npix

    return tot_out, npix_out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int get_side_double(double ax, double ay, double bx, double by):
    cdef double dot

    dot = ax*by-ay*bx
    if dot < 0:
        return -1
    elif dot > 0:
        return 1
    else:
        return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def events_poly_stats(float[::1] evtx not None,
                      float[::1] evty not None,
                      float[:,::1] px not None,
                      float[:,::1] py not None):

    """Get total counts inside polygons

    evtx: 1D array of X coordinates
    evty: 1D array of Y coordinates
    px: 2D array [[p1_x1, p1_x2...], [p2_x1, p2_x2...]]
    py: 2D array [[p1_y1, p1_y2...], [p2_y1, p2_y2...]]

    Returns [ncts_p1, ncts_p2...]

    """

    cdef int pt, evt
    cdef int nitems, npts, npol, npolys, nevts
    cdef int dirn

    cdef float minx, maxx, miny, maxy
    cdef float ex, ey
    cdef float seg_x, seg_y, del_x, del_y
    cdef int side

    cdef int ncts

    assert evtx.shape[0] == evty.shape[0]
    assert px.shape[0]==py.shape[0] and px.shape[1]==py.shape[1]

    npolys = px.shape[0]
    npts = px.shape[1]
    nevts = evtx.shape[0]

    assert npts>0
    assert npolys>0

    ncts_out = np.zeros((npolys,), dtype=np.int32)
    cdef int[::1] ncts_out_v = ncts_out

    cdef int below, above, fixup
    for npol in range(npolys):

        # find range of polygon
        minx=1e20
        miny=1e20
        maxx=-1e20
        maxy=-1e20
        for pt in range(npts):
            minx = min(minx, px[npol,pt])
            maxx = max(maxx, px[npol,pt])
            miny = min(miny, py[npol,pt])
            maxy = max(maxy, py[npol,pt])

        ncts = 0
        for evt in range(nevts):
            ey = evty[evt]
            ex = evtx[evt]
            if ex<minx or ex>maxx or ey<miny or ey>maxy:
                continue

            dirn = 0
            for pt in range(npts):
                seg_y = py[npol,(pt+1)%npts] - py[npol,pt]
                seg_x = px[npol,(pt+1)%npts] - px[npol,pt]
                del_y = ey - py[npol,pt]
                del_x = ex - px[npol,pt]

                side = get_side(seg_x, seg_y, del_x, del_y)
                if dirn == 0:  # uninitialised
                    if side != 0:
                        # initialise
                        dirn = side
                    else:
                        # we're at the edge, so say outside
                        dirn = -2
                        break
                else:
                    if side != dirn:
                        # side was different from last time, so outside
                        dirn = -2
                        break

            if dirn != -2:
                # inside
                ncts += 1

        ncts_out_v[npol] = ncts

    return ncts_out
