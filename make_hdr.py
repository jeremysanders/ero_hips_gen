#!/usr/bin/env python3

from astropy.io import fits
import argparse
import datetime
import re
import glob
import numpy as N
import sys
import os
import forkqueue
import astropy_healpix
import astropy.units as u

def getMinMax(fname):
    with fits.open(fname) as fin:
        d = fin[0].data

    ok = N.any(N.isfinite(d))
    if not ok:
        return N.nan, N.nan
    d = d[ok]
    return d.min(), d.max()

def getPixelViewRange(indir):
    # very bright things
    exclude = set([591, 612, 590, 613, 480, 377, 693])

    alldata = []
    for fname in glob.iglob(os.path.join(indir, 'Norder3', 'Dir*', 'Npix*.fits')):
        num = int(re.search('Npix([0-9]+).fits', fname).group(1))
        if num in exclude:
            continue

        f = fits.open(fname)
        img = f[0].data
        valid = img[N.isfinite(img)]
        if len(valid) > 0:
            alldata.append(valid)
        f.close()
    alldata = N.concatenate(alldata)
    percs = N.percentile(alldata, [0, 0.01, 0.1, 1, 95, 99, 99.5, 99.75, 99.8, 99.9, 99.99])
    return percs[2], percs[8]

def process(indir, procs):
    ranges = {}
    maxorder = max([
        int(os.path.basename(x)[6:])
        for x in glob.glob(os.path.join(indir, 'Norder*'))])

    formats = "fits png"
    for order in range(3, maxorder+1):
        args = ((f,) for f in glob.iglob(os.path.join(indir, f'Norder{order}', 'Dir*', 'Npix*.fits')))

        minord = N.inf
        maxord = -N.inf
        with forkqueue.ForkQueue(numforks=procs, ordered=False) as q:
            for minv, maxv in q.process(getMinMax, args):
                if N.isfinite(minv):
                    minord = min(minord, minv)
                if N.isfinite(maxv):
                    maxord = max(maxord, maxv)
        ranges[order] = (minord, maxord)

    low, high = getPixelViewRange(indir)
    if 'ExposureMap' in indir:
        low = 0

    hipspixsize = astropy_healpix.HEALPix(2**(maxorder+9)).pixel_resolution.to_value(u.deg)

    initial_ra = 137.94690098
    initial_dec = -48.28904083
    initial_fov = 180
    title = os.path.basename(indir)

    isodatetime = datetime.datetime.now().isoformat()
    proptext = \
f'''hips_initial_fov     = { initial_fov }
hips_initial_ra      = { initial_ra }
hips_initial_dec     = { initial_dec }
creator_did          = ivo://UNK.AUTH/P/HiPSID
hips_pixel_bitpix    = -32
hips_pixel_scale     = {hipspixsize}
hips_data_range      = { ranges[maxorder][0] } {ranges[maxorder][1] }
data_pixel_bitpix    = -32
hips_sampling        = none
hips_overlay         = mean
hips_hierarchy       = median
hips_skyval          = none
hips_creator         = MPE (Jeremy Sanders)
#hips_copyright      = Copyright mention of the HiPS
obs_title            = { title }
#obs_collection      = Dataset collection name
#obs_description     = Dataset text description
#obs_ack             = Acknowledgement mention
#prov_progenitor     = Provenance of the original data (free text)
#bib_reference       = Bibcode for bibliographic reference
#bib_reference_url   = URL to bibliographic reference
#obs_copyright       = Copyright mention of the original data
#obs_copyright_url   = URL to copyright page of the original data
#t_min               = Start time in MJD ( =(Unixtime/86400)+40587  or https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl)
#t_max               = Stop time in MJD
obs_regime           = X-ray
hips_pixel_cut       = { low } { high }
#em_min              = Start in spectral coordinates in meters ( =2.998E8/freq in Hz, or =1.2398841929E-12*energy in MeV )
#em_max              = Stop in spectral coordinates in meters
hips_builder         = hips_gen.py, written by Jeremy Sanders
hips_version         = 1.4
hips_release_date    = { isodatetime }
hips_frame           = equatorial
hips_order           = {maxorder}
hips_order_min       = 0
hips_tile_width      = 512
#hips_service_url    = ex: http://yourHipsServer/null
hips_status          = public master clonableOnce
hips_tile_format     = { formats }
s_pixel_scale        = 0.0011111111111111111
dataproduct_type     = image
moc_sky_fraction     = 0.5
hips_estsize         = 55289513
hips_creation_date   = { isodatetime }
#____FOR_COMPATIBILITY_WITH_OLD_HIPS_CLIENTS____
label                = { title }
coordsys             = C
maxOrder             = { maxorder }
format               = { formats }
'''

    propfname = os.path.join(indir, 'properties')
    print('Writing', propfname)
    with open(propfname, 'w') as fout:
        fout.write(proptext)

    html=\
f"""<!doctype html>
<html lang="en">
<head>
   <script type="text/javascript" src="https://code.jquery.com/jquery-3.6.1.min.js"></script>
   <link rel="stylesheet" href="https://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.css" >
   <script type="text/javascript" src="https://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.js" charset="utf-8"></script>

   <script type="text/javascript">
     $( document ).ready(function() {{
         var aladin = A.aladin('#aladin-lite-div', {{fov:{initial_fov}, target: '{initial_ra} {initial_dec}'}});

         aladin.createImageSurvey(
             'hipsmap', 'hipsmap',
             "./",
             "equatorial", {maxorder}, {{imgFormat: 'png'}},
         );

         aladin.setImageSurvey('hipsmap');
     }});
   </script>
</head>

<body>
<h1>eROSITA HiPS Map ({title})</h1>
<p>This Web resource contains HiPS components for a progressive survey.</p>

<div id="aladin-lite-div" style="width:500px;height:500px;"></div>

<p>
This survey can be displayed
by <a href="https://aladin.u-strasbg.fr/AladinLite/">Aladin Lite</a>
(as in this page),
by <a href="https://aladin.u-strasbg.fr/java/nph-aladin.pl?frame=downloading">Aladin
Desktop</a> client (just open the base URL) or any other HiPS-aware
clients.
</p>

</body>

</html>
"""

    htmlfname = os.path.join(indir, 'index.html')
    print('Writing', htmlfname)
    with open(htmlfname, 'w') as fout:
        fout.write(html)

def main():
    parser = argparse.ArgumentParser(
        prog='make_hdr.py',
        description='Make a HiPS properties file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('root_dir', help='HiPS root directory')
    parser.add_argument('--procs', default=8, type=int, help='Number of processes')

    args = parser.parse_args()

    process(args.root_dir, args.procs)

    # for d in sorted(glob.glob('/hedr_local/erodr/hips/eRASS1_02?_*_c010')):
    #     process(d)

if __name__ == '__main__':
    main()
