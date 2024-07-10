#!python
#cython: language_level=3

import numpy as np
cimport cython

cdef int rintersects(float[4] a, float[4] b):
    return not(a[2]>b[3] or a[3]<b[2] or a[0]>b[1] or a[1]<b[0])

cdef class KDRangeTree:
    cdef object _v
    cdef object _n_v
    cdef object _n_cutval
    cdef object _n_left
    cdef object _n_right

    cdef int root
    cdef int num
    cdef int nnodes
    cdef float[:,::1] v
    cdef int[::1] n_v
    cdef float[::1] n_cutval
    cdef int[::1] n_left
    cdef int[::1] n_right

    def __cinit__(self, float[:,::1] vals):
        """Initialize set of points and build tree.

        vals: numpy 2D array (Nx2) of float32.
        """

        assert vals.shape[1] == 2
        self.num = vals.shape[0]
        self.nnodes = 0

        self._v = np.array(vals, dtype=np.float32)
        self.v = self._v

        self._n_v = np.full(self.num*2, -1, dtype=np.intc)
        self.n_v = self._n_v
        self._n_left = np.full(self.num*2, -1, dtype=np.intc)
        self.n_left = self._n_left
        self._n_left = np.full(self.num*2, -1, dtype=np.intc)
        self.n_left = self._n_left
        self._n_right = np.full(self.num*2, -1, dtype=np.intc)
        self.n_right = self._n_right
        self._n_cutval = np.full(self.num*2, np.nan, dtype=np.float32)
        self.n_cutval = self._n_cutval

        self.root = self.kdtree(0, self.num, 0)

    cdef int makeNode(self):
        self.nnodes += 1
        return self.nnodes-1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef int kdtree(self, int start, int length, int depth):
        cdef int ni, axis, mid

        if length == 0:
            return -1
        elif length == 1:
            ni = self.makeNode()
            self.n_v[ni] = start
            return ni

        axis = depth % 2

        # sort on axis
        idxs = np.argsort(self.v[start:start+length,axis])
        self._v[start:start+length] = self._v[idxs+start]

        # split on median
        mid = start + length//2 - 1

        ni = self.makeNode()
        self.n_cutval[ni] = self.v[mid,axis]
        self.n_left[ni] = self.kdtree(start, mid-start+1, depth+1)
        self.n_right[ni] = self.kdtree(mid+1, length-(mid-start)-1, depth+1)
        return ni

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef int addTree(self, int ni, out):
        if ni < 0:
            return 0

        if self.n_left[ni]<0 and self.n_right[ni]<0:
            out.append(self.n_v[ni])
        else:
            self.addTree(self.n_left[ni], out)
            self.addTree(self.n_right[ni], out)

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef int inner_query(self, int ni, float[4] qrng, float[4] crng, int depth, out):

        cdef float[4] newrng

        if self.n_left[ni] < 0 and self.n_right[ni] < 0:
            # leaf node
            if qrng[0]<=self.v[self.n_v[ni],0]<=qrng[1] and qrng[2]<=self.v[self.n_v[ni],1]<=qrng[3]:
                out.append(self.n_v[ni])
            return 0

        # entire current range is within range, so add subtrees
        if ( crng[0] >= qrng[0] and crng[1] <= qrng[1] and
             crng[2] >= qrng[2] and crng[3] <= qrng[3] ):
            self.addTree(self.n_left[ni], out)
            self.addTree(self.n_right[ni], out)
            return 0

        if self.n_left[ni] >= 0:
            newrng = crng
            newrng[(depth%2)*2+1] = self.n_cutval[ni]
            if rintersects(newrng, qrng):
                self.inner_query(self.n_left[ni], qrng, newrng, depth+1, out)
        if self.n_right[ni] >= 0:
            newrng = crng
            newrng[(depth%2)*2+0] = self.n_cutval[ni]
            if rintersects(newrng, qrng):
                self.inner_query(self.n_right[ni], qrng, newrng, depth+1, out)

        return 0

    def query(self, rng):
        """Return points within the range given by rng.

        rng: [min_coord0, max_coord0, min_coord1, maxcoord1]
        returns array of points.
        """

        cdef int i
        cdef float[4] qrng, crng

        assert len(rng) == 4
        for i in range(4):
            qrng[i] = rng[i]

        out = []
        crng[1] = np.inf
        crng[0] = -crng[1]
        crng[2] = -crng[1]
        crng[3] = crng[1]

        self.inner_query(self.root, qrng, crng, 0, out)
        if len(out) == 0:
            return np.empty((0,2), dtype=np.float32)
        else:
            return self._v[np.array(out)]


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
    cdef float[:,::1] evts_v

    assert evtx.shape[0] == evty.shape[0]
    assert px.shape[0]==py.shape[0] and px.shape[1]==py.shape[1]

    npolys = px.shape[0]
    npts = px.shape[1]

    assert npts>0
    assert npolys>0

    ncts_out = np.zeros((npolys,), dtype=np.int32)
    cdef int[::1] ncts_out_v = ncts_out

    tree = KDRangeTree(np.column_stack((evty,evtx)))

    cdef int below, above
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

        evts = tree.query([miny, maxy, minx, maxx])
        evts_v = evts
        nevts = evts_v.shape[0]

        ncts = 0
        for evt in range(nevts):
            ey = evts_v[evt, 0]
            ex = evts_v[evt, 1]

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
