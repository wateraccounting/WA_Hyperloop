"""
A series of functions to do bulk operations on geotiffs, including
reprojections.

contact: b.coerver@un-ihe.org
"""
import os
import datetime
import calendar
import collections
import subprocess
import csv
from geopy import distance
from osgeo import gdal, osr
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np

gdal.UseExceptions() 


def mm_to_km3(lu_fih, var_fihs):
    """
    Convert the unit of a series of maps from mm to km3.

    Parameters
    ----------
    lu_fih : str
        Filehandle pointing to georeferenced tiff raster map.
    fihs : ndarray
        Array with filehandles pointing to maps to be used.

    Returns
    -------
    var_new_fhs : ndarray
        Array with filehandles pointing to output maps.
    """
    area = map_pixel_area_km(lu_fih)
    geo_info = get_geoinfo(lu_fih)
    var_new_fihs = np.array()
    for var_fih in var_fihs:
        var = open_as_array(var_fih)
        var_area = (var * area) / 1e6
        var_new_fih = var_fih.replace('.tif', '_km3.tif')
        create_geotiff(var_new_fih, var_area, *geo_info)
        var_new_fihs = np.append(var_new_fihs, var_new_fih)
    return var_new_fihs


def set_classes_to_value(fih, lu_fih, classes, value):
    """
    Open a rasterfile and change certain pixels to a new value. Classes and
    lu_fih is used to create a mask. The mask is then used to set the pixel values
    in fih to value.

    Parameters
    ----------
    fih : str
        Filehandle pointing to georeferenced tiff raster map.
    lu_fih : str
        Filehandle pointing to georeferenced tiff raster map. Should have same
        dimensions as fih.
    classes : list
        List with values, the values are looked up in lu_fih, the corresponding
        pixels in fih are then changed.
    value : float or int
        Value to change the pixelvalues in fih into.
    """
    alpha = open_as_array(fih, nan_values=True)
    lulc = open_as_array(lu_fih, nan_values=True)
    mask = np.logical_or.reduce([lulc == x for x in classes])
    alpha[mask] = value
    geo_info = get_geoinfo(lu_fih)
    create_geotiff(fih, alpha, *geo_info)


def calc_mean_std(fihs):
    """
    Calculate the mean and the standard deviation per pixel for a serie of maps.

    Parameters
    ----------
    fihs : ndarray
        Array with filehandles pointing to maps to be used.

    Returns
    -------
    std : ndarray
        Array with the standard deviation for each pixel.
    mean : ndarray
        Array with the mean for each pixel.
    """
    data_sum = data_count = np.zeros_like(open_as_array(fihs[0]))

    for fih in fihs:
        data = open_as_array(fih)
        data_sum = np.nansum([data_sum, data], axis=0)

        count = np.ones_like(data)
        count[np.isnan(data)] = 0
        data_count += count

    mean = data_sum / data_count
    data_sum = np.zeros_like(data)

    for fih in fihs:
        data = (open_as_array(fih) - mean)**2 / data_count
        data_sum += data

    std = np.sqrt(data_sum)

    return std, mean


def get_gdalwarp_info(fih, subdataset=0):
    """
    Get information in string format from a geotiff or HDF4 file for use by GDALWARP.

    Parameters
    ----------
    fih : str
        Filehandle pointing to a geotiff or HDF4 file.
    subdataset = int, optional
        Value indicating a subdataset (in case of HDF4), default is 0.

    Returns
    -------
    srs : str
        The projection of the fih.
    res : str
        Resolution of the fih.
    bbox : str
        Bounding box (xmin, ymin, xmax, ymax) of the fih.
    ndv : str
        No-Data-Value of the fih.
    """
    dataset = gdal.Open(fih, gdal.GA_ReadOnly)
    tpe = dataset.GetDriver().ShortName
    if tpe == 'HDF4':
        dataset = gdal.Open(dataset.GetSubDatasets()[subdataset][0])
    ndv = str(dataset.GetRasterBand(1).GetNoDataValue())
    if ndv == 'None':
        ndv = 'nan'
    srs = dataset.GetProjectionRef()
    if not srs:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326).ExportToPrettyWkt()
        print("srs not defined, using EPSG4326.")
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    res = ' '.join([str(xsize), str(ysize)])
    geot = dataset.GetGeoTransform()
    xmin = geot[0]
    ymin = geot[3] + geot[5] * ysize
    xmax = geot[0] + geot[1] * xsize
    ymax = geot[3]
    bbox = ' '.join([str(xmin), str(ymin), str(xmax), str(ymax)])
    return srs, res, bbox, ndv


def average_series(tifs, dates, length, output_folder, para_name='Average',
                   categories=None, lu_fih=None, timescale='months'):
    """
    Compute moving averages for multiple maps.

    Parameters
    ----------
    tifs : ndarray
        Array of strings pointing to maps.
    dates : ndarray
        Array with datetime.date object referring to the dates of tifs.
    length : dict or int
        Length of moving average. When dictionary, different lengths can be used for different
        landuse categories.
    output_folder : str
        Folder to store results.
    para_name : str, optional
        Name used for output tifs. Default is 'Average'.
    categories : dict, optional
        Dictionary describing the different landuse categories, keys should be identical to keys
        in length. Default is None.
    lu_fih : str, optional
        Landuse map, default is None.
    timescale : str, optional
        Timescale of the maps in tifs. Default is 'months'.

    Returns
    -------
    output_tifs : ndarray
        Array with paths to the new maps.
    dates : ndarray
        Array with datetime.date object reffering to the dates of output_tifs.
    """
    assert_missing_dates(dates, timescale=timescale)

    masked_average = False

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if isinstance(length, dict):
        max_length = np.max(length.values())
        masked_average = True
        assert_same_keys([length, categories])
        assert_proj_res_ndv([tifs, np.array(lu_fih)])
    else:
        max_length = length
        assert_proj_res_ndv([tifs])

    geo_info = get_geoinfo(tifs[0])

    output_tifs = np.array([])

    for date in dates[(max_length-1):]:
        if masked_average:
            array = masked_moving_average(date, tifs, dates,
                                          lu_fih, length, categories)
        if not masked_average:
            array = moving_average(date, tifs, dates, moving_avg_length=length)
        tif = os.path.join(output_folder,
                           '{0}_{1}{2}.tif'.format(para_name, date.year, str(date.month).zfill(2)))
        create_geotiff(tif, array, *geo_info)
        output_tifs = np.append(output_tifs, tif)

    return output_tifs, dates[(max_length-1):]


def moving_average(date, filehandles, filedates,
                   moving_avg_length=5, method='tail'):
    """
    Compute a moving (tail) average from a series of maps.

    Parameters
    ----------
    date : object
        Datetime.date object for which the average should be computed
    filehandles : ndarray
        Filehandles of the maps.
    filedates : ndarray
        Datetime.date objects corresponding to filehandles
    moving_average_length : int, optional
        Length of the tail, default is 3.
    method : str, optional
        Select wether to calculate the 'tail' average or 'central' average.

    Returns
    -------
    summed_data : ndarray
        The averaged data.
    """
    indice = np.where(filedates == date)[0][0]
    if method == 'tail':
        assert (indice + 1) >= moving_avg_length, "Not enough data available to calculate average of length {0}".format(moving_avg_length)
        to_open = filehandles[indice-(moving_avg_length-1):(indice+1)]
    elif method == 'central':
        assert (moving_avg_length % 2 != 0), "Please provide an uneven moving_avg_length"
        assert indice >= (moving_avg_length - 1) / 2, "Not enough data available to calculate central average of length {0}".format(moving_avg_length)
        assert indice < len(filedates) - (moving_avg_length - 1) / 2, "Not enough data available to calculate central average of length {0}".format(moving_avg_length)
        to_open = filehandles[indice-(moving_avg_length-1)/2:indice+(moving_avg_length-1)/2+1]
    summed_data = open_as_array(filehandles[indice]) * 0
    for fih in to_open:
        data = open_as_array(fih)
        summed_data += data
    summed_data /= len(to_open)
    return summed_data


def masked_moving_average(date, fihs, dates, lu_fih, moving_avg_length,
                          categories, method='tail'):
    """
    Calculate temporal trailing averages dependant on landuse categories.

    Parameters
    ----------
    date : object
        datetime.date object indicating for which month the average needs to be calculated.
    fihs : ndarray
        Array with filehandles pointing to maps.
    dates : ndarray
        Array with datetime.date objects referring to the maps in fihs.
    lu_fih : str
        Filehandle pointing to a landusemaps.
    moving_avg_length : dict
        Dictionary indicating the number of months needed to calculate the temporal
        trailing average.
    categories : dict
        Dictionary indicating which landuseclasses belong to which category. Should
        have the same keys as moving_avg_length.

    Returns
    -------
    AVG : ndarray
        Array with the averaged values.
    """
    # https://stackoverflow.com/a/40857703/4288201
    def flatten(l):
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(el, str):
                for sub in flatten(el):
                    yield sub
            else:
                yield el

    assert_same_keys([moving_avg_length, categories])
    lulc = open_as_array(lu_fih)
    xsize, ysize = get_geoinfo(lu_fih)[2:4]
    avg = np.zeros((ysize, xsize)) * np.nan

    for length in np.unique(moving_avg_length.values()):
        key_list = [key for key in moving_avg_length.keys() if moving_avg_length[key] is int(length)]
        classes = list(flatten([categories[key] for key in key_list]))
        mask = np.logical_or.reduce([lulc == value for value in classes])
        avg[mask] = moving_average(date, fihs, dates, moving_avg_length=length, method=method)[mask]

    return avg


def plot_category_areas(lu_fih, categories, output_fih, area_treshold=0.01):
    """
    Plot the relative areas of landuse categories in a pie chart.

    Parameters
    ----------
    lu_fih : str
        Filehandle pointing to a landusemap
    categories : dict
        Dictionary specifying all the landuse categories.
    output_fih : str
        Filehandle indicating where to save the graph.
    area_treshold : float, optional
        Categories with a relative area lower than the treshold are not plotted
        in the pie chart. Default values is 0.01.
    """
    area_map = map_pixel_area_km(lu_fih)
    lulc = open_as_array(lu_fih)
    areas = dict()
    total_area = np.nansum(area_map[~np.isnan(lulc)])

    for key in categories.keys():
        classes = categories[key]
        mask = np.logical_or.reduce([lulc == value for value in classes])
        area = np.nansum(area_map[mask])
        if area / total_area >= area_treshold:
            areas[key] = area

    clrs = ['#6bb8cc', '#87c5ad', '#9ad28d', '#acd27a', '#c3b683',
            '#d4988b', '#b98b89', '#868583', '#497e7c']
    plt.figure(figsize=(15, 15))
    plt.clf()
    plt.title('Total Area ({0:.2f} ha)'.format(total_area/100))
    plt.pie(areas.values(), labels=areas.keys(), autopct='%1.1f%%', colors=clrs)
    plt.savefig(output_fih)


def sort_files(input_dir, year_position, month_position=None,
               day_position=None, doy_position=None, extension='tif'):
    r"""
    Substract metadata from multiple filenames.

    Parameters
    ----------
    input_dir : str
        Folder containing files.
    year_position : list
        The indices where the year is positioned in the filenames, see example.
    month_position : list, optional
        The indices where the month is positioned in the filenames, see example.
    day_position : list, optional
        The indices where the day is positioned in the filenames, see example.
    doy_position : list, optional
        The indices where the doy is positioned in the filenames, see example.
    extension : str
        Extension of the files to look for in the input_dir.

    Returns
    -------
    filehandles : ndarray
        The files with extension in input_dir.
    dates : ndarray
        The dates corresponding to the filehandles.
    years : ndarray
        The years corresponding to the filehandles.
    months : ndarray
        The years corresponding to the filehandles.
    days :ndarray
        The years corresponding to the filehandles.
    """
    dates = np.array([])
    years = np.array([])
    months = np.array([])
    days = np.array([])
    filehandles = np.array([])
    files = list_files_in_folder(input_dir, extension=extension)
    for fil in files:
        filehandles = np.append(filehandles, fil)
        year = int(fil[year_position[0]:year_position[1]])
        month = 1
        if month_position is not None:
            month = int(fil[month_position[0]:month_position[1]])
        day = 1
        if day_position is not None:
            day = int(fil[day_position[0]:day_position[1]])
        if doy_position is not None:
            date = datetime.date(year, 1, 1) + datetime.timedelta(int(fil[doy_position[0]:doy_position[1]]) - 1)
            month = date.month
            day = date.day
        else:
            date = datetime.date(year, month, day)
        years = np.append(years, year)
        months = np.append(months, month)
        days = np.append(days, day)
        dates = np.append(dates, date)
    return filehandles, dates, years, months, days


def common_dates(dates_list):
    """
    Checks for common dates between multiple lists of datetime.date objects.

    Parameters
    ----------
    dates_list : list
        Contains lists with datetime.date objects.

    Returns
    -------
    com_dates : ndarray
        Array with datetime.date objects for common dates.
    """
    com_dates = dates_list[0]
    for date_list in dates_list[1:]:
        com_dates = np.sort(list(set(com_dates).intersection(date_list)))
    return com_dates


def assert_missing_dates(dates, timescale='months', quantity=1):
    """
    Checks if a list of dates is continuous, i.e. are there temporal gaps in the dates.

    Parameters
    ----------
    dates : ndarray
        Array of datetime.date objects.
    timescale : str, optional
        Timescale to look for, default is 'months'.
    """
    current_date = dates[0]
    enddate = dates[-1]
    if timescale == 'months':
        while current_date <= enddate:
            assert current_date in dates, "{0} is missing in the dataset".format(current_date)
            current_date = current_date + relativedelta(months=quantity)


def convert_datetime_date(dates, out=None):
    """
    Convert datetime.datetime objects into datetime.date objects or viceversa.

    Parameters
    ----------
    dates : ndarray or list
        List of datetime.datetime objects.
    out : str or None, optional
        string can be either 'date' or 'datetime', if out is not None, the output will always
        be date or datetime, regardless of the type of input.

    Returns
    -------
    dates : ndarray
        Array with datetime.date objects.
    """
    if out == 'date':
        dates = np.array([datetime.date(dt.year, dt.month, dt.day) for dt in dates])
    elif out == 'datetime':
        dates = np.array([datetime.datetime(date.year, date.month, date.day, 0, 0, 0) for date in dates])
    else:
        if isinstance(dates[0], datetime.datetime):
            dates = np.array([datetime.date(dt.year, dt.month, dt.day) for dt in dates])
        elif isinstance(dates[0], datetime.date):
            dates = np.array([datetime.datetime(date.year, date.month, date.day, 0, 0, 0) for date in dates])

    return dates


def match_proj_res_ndv(source_file, target_fihs, output_dir, dtype='Float32'):
    """
    Matches the projection, resolution and no-data-value of a list of target-files
    with a source-file and saves the new maps in output_dir.

    Parameters
    ----------
    source_file : str
        The file to match the projection, resolution and ndv with.
    target_fihs : list
        The files to be reprojected.
    output_dir : str
        Folder to store the output.
    resample : str, optional
        Resampling method to use, default is 'near' (nearest neighbour).
    dtype : str, optional
        Datatype of output, default is 'float32'.
    scale : int, optional
        Multiple all maps with this value, default is None.

    Returns
    -------
    output_files : ndarray
        Filehandles of the created files.
    """
    ndv, xsize, ysize, geot, projection = get_geoinfo(source_file)[1:]
    type_dict = {gdal.GetDataTypeName(i): i for i in range(1, 12)}
    output_files = np.array([])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for target_file in target_fihs:
        filename = os.path.split(target_file)[1]
        output_file = os.path.join(output_dir, filename)
        options = gdal.WarpOptions(width=xsize,
                                   height=ysize,
                                   outputBounds=(geot[0], geot[3] + ysize * geot[5],
                                                 geot[0] + xsize * geot[1], geot[3]),
                                   outputBoundsSRS=projection,
                                   dstSRS=projection,
                                   dstNodata=ndv,
                                   outputType=type_dict[dtype])
        gdal.Warp(output_file, target_file, options=options)
        output_files = np.append(output_files, output_file)
    return output_files


def get_geoinfo(fih, subdataset=0):
    """
    Substract metadata from a geotiff, HDF4 or netCDF file.

    Parameters
    ----------
    fih : str
        Filehandle to file to be scrutinized.
    subdataset : int, optional
        Layer to be used in case of HDF4 or netCDF format, default is 0.

    Returns
    -------
    driver : str
        Driver of the fih.
    ndv : float
        No-data-value of the fih.
    xsize : int
        Amount of pixels in x direction.
    ysize : int
        Amount of pixels in y direction.
    geot : list
        List with geotransform values.
    Projection : str
        Projection of fih.
    """
    sourceds = gdal.Open(fih, gdal.GA_ReadOnly)
    tpe = sourceds.GetDriver().ShortName
    if tpe == 'HDF4' or tpe == 'netCDF':
        sourceds = gdal.Open(sourceds.GetSubDatasets()[subdataset][0])
    ndv = sourceds.GetRasterBand(1).GetNoDataValue()
    xsize = sourceds.RasterXSize
    ysize = sourceds.RasterYSize
    geot = sourceds.GetGeoTransform()
    projection = osr.SpatialReference()
    projection.ImportFromWkt(sourceds.GetProjectionRef())
    driver = gdal.GetDriverByName(tpe)
    return driver, ndv, xsize, ysize, geot, projection


def list_files_in_folder(folder, extension='tif'):
    """
    List the files in a folder with a specified extension.

    Parameters
    ----------
    folder : str
        Folder to be scrutinized.
    extension : str, optional
        Type of files to look for in folder, default is 'tif'.

    Returns
    -------
    list_of_files : list
        List with filehandles of the files found in folder with extension.
    """
    list_of_files = [os.path.join(folder, fn) for fn in next(os.walk(folder))[2] if fn.split('.')[-1] == extension]
    return list_of_files


def open_as_array(fih, bandnumber=1, nan_values=True):
    """
    Open a map as an numpy array.

    Parameters
    ----------
    fih: str
        Filehandle to map to open.
    bandnumber : int, optional
        Band or layer to open as array, default is 1.
    dtype : str, optional
        Datatype of output array, default is 'float32'.
    nan_values : boolean, optional
        Convert he no-data-values into np.nan values, note that dtype needs to
        be a float if True. Default is False.

    Returns
    -------
    array : ndarray
        array with the pixel values.
    """
    dataset = gdal.Open(fih, gdal.GA_ReadOnly)
    tpe = dataset.GetDriver().ShortName
    if tpe == 'HDF4':
        subdataset = gdal.Open(dataset.GetSubDatasets()[bandnumber][0])
        ndv = int(subdataset.GetMetadata()['_FillValue'])
    else:
        subdataset = dataset.GetRasterBand(bandnumber)
        ndv = subdataset.GetNoDataValue()
    array = subdataset.ReadAsArray()
    if nan_values:
        if len(array[array == ndv]) >0:
            array[array == ndv] = np.nan
    return array


def create_geotiff(fih, array, driver, ndv, xsize, ysize, geot, projection, compress=None):
    """
    Creates a geotiff from a numpy array.

    Parameters
    ----------
    fih : str
        Filehandle for output.
    array: ndarray
        array to convert to geotiff.
    driver : str
        Driver of the fih.
    ndv : float
        No-data-value of the fih.
    xsize : int
        Amount of pixels in x direction.
    ysize : int
        Amount of pixels in y direction.
    geot : list
        List with geotransform values.
    Projection : str
        Projection of fih.
    """
    datatypes = {gdal.GetDataTypeName(i).lower() : i for i in range(1, 12)}
    if compress != None:
        dataset = driver.Create(fih, xsize, ysize, 1, datatypes[array.dtype.name], ['COMPRESS={0}'.format(compress)])
    else:
        dataset = driver.Create(fih, xsize, ysize, 1, datatypes[array.dtype.name])
    if ndv is None:
        ndv = -9999
    array[np.isnan(array)] = ndv
    dataset.GetRasterBand(1).SetNoDataValue(ndv)
    dataset.SetGeoTransform(geot)
    dataset.SetProjection(projection.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(array)
    dataset = None
    if "nt" not in array.dtype.name:
        array[array == ndv] = np.nan


def pixel_coordinates(lon, lat, fih):
    """
    Find the corresponding pixel to a latitude and longitude.

    Parameters
    ----------
    lon : float or int
        Longitude to find.
    lat : float or int
        Latitude to find.
    fih : str
        Filehandle pointing to the file to be searched.

    Returns
    -------
    xpixel : int
        The index of the longitude.
    ypixel : int
        The index of the latitude.
    """
    sourceds = gdal.Open(fih, gdal.GA_ReadOnly)
    xsize = sourceds.RasterXSize
    ysize = sourceds.RasterYSize
    geot = sourceds.GetGeoTransform()
    assert (lon >= geot[0]) & (lon <= geot[0] + xsize * geot[1]), 'longitude is not on the map'
    assert (lat <= geot[3]) & (lat >= geot[3] + ysize * geot[5]), 'latitude is not on the map'
    location = geot[0]
    xpixel = -1
    while location <= lon:
        location += geot[1]
        xpixel += 1
    location = geot[3]
    ypixel = -1
    while location >= lat:
        location += geot[5]
        ypixel += 1
    return xpixel, ypixel


def assert_proj_res_ndv(list_of_filehandle_lists, check_ndv=True):
    """
    Check if the projection, resolution and no-data-value of all provided filehandles are the same.

    Parameters
    ----------
    list_of_filehandle_lists : list
        List with different ndarray containing filehandles to compare.
    check_ndv : boolean, optional
        Check or ignore the no-data-values, default is True.

    Examples
    --------
    >>> assert_proj_res_ndv([et_fihs, ndm_fihs, p_fihs], check_ndv = True)
    """
    longlist = np.array([])
    for fih_list in list_of_filehandle_lists:
        if isinstance(fih_list, list):
            longlist = np.append(longlist, np.array(fih_list))
        if isinstance(fih_list, np.ndarray):
            longlist = np.append(longlist, fih_list)
        if isinstance(fih_list, str):
            longlist = np.append(longlist, np.array(fih_list))
    t_srs, t_ts, t_te, t_ndv = get_gdalwarp_info(longlist[0])
    for fih in longlist[1:]:
        s_srs, s_ts, s_te, s_ndv = get_gdalwarp_info(fih)
        if check_ndv:
            assert np.all([s_ts == t_ts, s_te == t_te, s_srs == t_srs, s_ndv == t_ndv]), "{0} does not have the same Proj/Res/ndv as {1}".format(longlist[0], fih)
        else:
            assert np.all([s_ts == t_ts, s_te == t_te, s_srs == t_srs]), "{0} does not have the same Proj/Res as {1}".format(longlist[0], fih)


def map_pixel_area_km(fih, approximate_lengths=False):
    """
    Calculate the area of the pixels in a geotiff.

    Parameters
    ----------
    fih : str
        Filehandle pointing to a geotiff.
    approximate_lengths : boolean, optional
        Give the approximate length per degree [km/deg] instead of the area [km2], default is False.

    Returns
    -------
    map_area : ndarray
        The area per cell.
    """
    xsize, ysize, geot = get_geoinfo(fih)[2:-1]
    area_column = np.zeros((ysize, 1))
    for y_pixel in range(ysize):
        pnt1 = (geot[3] + y_pixel*geot[5], geot[0])
        pnt2 = (pnt1[0], pnt1[1] + geot[1])
        pnt3 = (pnt1[0] - geot[1],  pnt1[1])
        pnt4 = (pnt1[0] - geot[1], pnt1[1] + geot[1])
        u = distance.distance(pnt1, pnt2).km
        l = distance.distance(pnt3, pnt4).km
        h = distance.distance(pnt1, pnt3).km
        area_column[y_pixel, 0] = (u+l)/2*h
    map_area = np.repeat(area_column, xsize, axis=1)
    if approximate_lengths:
        pixel_approximation = np.sqrt(abs(geot[1]) * abs(geot[5]))
        map_area = np.sqrt(map_area) / pixel_approximation
    return map_area


def xdaily_to_monthly(files, dates, out_path, name_out):
    r"""

    Parameters
    ----------
    fihs : ndarray
        Array with filehandles pointing to maps.
    dates : ndarray
        Array with datetime.date objects referring to the maps in fihs.
    out_path : str
        Folder to save results.
    name_out : str
        Output files naming convention, add curly brackets to indicate
        where the year and month should be placed, e.g. r'LAI_{0}{1}.tif'
    """

    # Make sure the fiels and dates are sequential
    files = np.array([x for _, x in sorted(zip(dates, files))])
    dates = np.array(sorted(dates))

    # Check if out_path exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Check if all maps have the same projection
    assert_proj_res_ndv([files])

    # Get geo-info
    geo_info = get_geoinfo(files[0])

    # Create tuples with date couples
    date_couples = np.array(zip(dates[0:-1], dates[1:]))

    # Loop over years and months
    for yyyy, month in np.unique([(date.year, date.month) for date in dates], axis=0):

        # Check which maps are relevant for current step
        relevant = [np.any([date1.month == month and date1.year == yyyy,
                            date2.month == month and date2.year == yyyy]) for date1, date2 in date_couples]

        # Create new empty array
        monthly = np.zeros((geo_info[3], geo_info[2]), dtype=np.float32)

        # Calculate length of month
        days_in_month = calendar.monthrange(yyyy, month)[1]

        # Loop over relevant dates
        for date1, date2 in date_couples[relevant]:

            print(date1, date2)

            # Open relevant maps
            xdaily1 = open_as_array(files[dates == date1][0])
            xdaily2 = open_as_array(files[dates == date2][0])

            # Correct dateranges at month edges
            if np.any([date1.month != month, date1.year != yyyy]):

                date1 = datetime.date(yyyy, month, 1)

            if np.any([date2.month != month, date2.year != yyyy]):

                date2 = datetime.date(yyyy, month, days_in_month) + datetime.timedelta(days=1)

            # Calculate how many relevant days there are in the current substep
            relevant_days = (date2 - date1).days

            # Add values to map
            monthly += np.sum([xdaily1, xdaily2], axis=0) * 0.5 * relevant_days

            print(date1, date2)
            print(relevant_days)

        # Calculate monthly average
        monthly /= days_in_month

        # Create output filehandle
        out_fih = os.path.join(out_path, name_out.format(yyyy, str(month).zfill(2)))

        # Save array as geotif
        create_geotiff(out_fih, monthly, *geo_info, compress="LZW")

        print("{0} {1} Created".format(yyyy, month))


def convert_to_tif(z, lat, lon, output_fh, gdal_grid_path=r'C:\Program Files\QGIS 2.18\bin\gdal_grid.exe'):
    """
    Create a geotiff with WGS84 projection from three arrays specifying (x,y,z)
    values.

    Parameters
    ----------
    z : ndarray
        Array containing the z-values.
    lat : ndarray
        Array containing the latitudes (in decimal degrees) corresponding to
        the z-values.
    lon : ndarray
        Array containing the latitudes (in decimal degrees) corresponding to
        the z-values.
    output_fh : str
        String defining the location for the output file.
    gdal_grid_path : str
        Path to the gdal_grid executable.
    """
    folder, filen = os.path.split(output_fh)

    if not os.path.exists(folder):
        os.chdir(folder)

    if np.all([lat.ndim == 2, lon.ndim == 2, z.ndim == 2]):
        csv_path = os.path.join(folder, 'temp.csv')
        with open(csv_path, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['Easting', 'Northing', 'z'])
            for xindex in range(np.shape(lat)[0]):
                for yindex in range(np.shape(lat)[1]):
                    spamwriter.writerow([lon[xindex, yindex], lat[xindex, yindex], z[xindex, yindex]])

    elif np.all([lat.ndim == 1, lon.ndim == 1, z.ndim == 1]):
        csv_path = os.path.join(folder, 'temp.csv')
        with open(csv_path, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['Easting', 'Northing', 'z'])
            for xindex in range(np.shape(lat)[0]):
                spamwriter.writerow([lon[xindex], lat[xindex], z[xindex]])

    else:
        raise ValueError("convert_to_tif is not compatible with the given \
                         dimensions of z, lat and lon.")

    vrt_path = os.path.join(folder, 'temp.vrt')
    with open(vrt_path, "w") as filen:
        filen.write('<OGRVRTDataSource>')
        filen.write('\n\t<OGRVRTLayer name="temp">')
        filen.write('\n\t\t<SrcDataSource>{0}</SrcDataSource>'.format(csv_path))
        filen.write('\n\t\t<GeometryType>wkbPoint</GeometryType>')
        filen.write('\n\t\t<GeometryField encoding="PointFromColumns" x="Easting" y="Northing" z="z"/>')
        filen.write('\n\t</OGRVRTLayer>')
        filen.write('\n</OGRVRTDataSource>')

    string = [gdal_grid_path,
              '-a_srs "+proj=longlat +datum=WGS84 +no_defs "',
              '-of GTiff',
              '-l temp',
              '-a linear:radius={0}:nodata=-9999'.format(np.max([np.max(np.diff(lon)), np.max(np.diff(lat))])),
              vrt_path,
              output_fh]

    proc = subprocess.Popen(' '.join(string), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    print(out, err)

    os.remove(csv_path)
    os.remove(vrt_path)


def assert_same_keys(list_of_dictionaries):
    """
    Check if different dictionaries have the same keys.

    Parameters
    ----------
    list_of_dictionaries : list
        List containing the dictionaries to check.
    """
    length1 = len(list_of_dictionaries[0].keys())
    keys1 = list_of_dictionaries[0].keys()
    for dictionary in list_of_dictionaries[1:]:
        assert len(dictionary.keys()) == length1, "The length of the provided dictionaries do not match"
        assert np.all(np.sort(dictionary.keys()) == np.sort(keys1)), "The keys in the provided dictionaries do not match"

