from os import makedirs
from os import system
from os.path import exists
from scipy.io import loadmat
from numpy import float32
import rasterio

from modland import ModisLandTile, MODIS_LAND_TILE_PROJECTION_PROJ4

NODATA = 0
TIFF_DIRECTORY = 'tiff'

corner = ModisLandTile(0, 0, 240, 240)

transform = corner.affine_transform

profile = {
    'crs': MODIS_LAND_TILE_PROJECTION_PROJ4,
    'driver': 'GTiff',
    'transform': transform,
    'count': 1,
    'nodata': 0,
    'dtype': float32
}

arrays = loadmat('test.mat')

if not exists(TIFF_DIRECTORY):
    makedirs(TIFF_DIRECTORY)

for variable in ['LEmerged']:
    filename = "{}/{}.tif".format(TIFF_DIRECTORY, variable)
    latlon_filename = "{}/{}_latlon.tif".format(TIFF_DIRECTORY, variable)

    print(filename)

    data = arrays[variable]
    height, width = data.shape

    profile.update({
        'height': height,
        'width': width
    })

    with rasterio.open(filename, 'w', **profile) as f:
        f.write(data.astype(float32), 1)

    print(latlon_filename)

    system('gdalwarp -s_srs "{}" -t_srs "{}" -srcnodata "{}" -dstnodata "{}" {} {}'.format(
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs",
        "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        NODATA,
        NODATA,
        filename,
        latlon_filename
    ))
