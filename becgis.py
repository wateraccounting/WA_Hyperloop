import os
import subprocess
import numpy as np
import datetime
import shutil
from osgeo import gdal, osr
from dateutil.relativedelta import relativedelta
import LatLon
import matplotlib.pyplot as plt
import collections
import tempfile
from scipy import ndimage

from WA_Hyperloop.paths import get_path


def mm_to_km3(lu_fh, var_fhs):
    """
    
    """
    area =  MapPixelAreakm(lu_fh)
    driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(lu_fh)
    var_new_fhs = list()
    for var_fh in var_fhs:
        var = OpenAsArray(var_fh)
        var[np.where(var==-9999)]=np.nan
        var_area = (var*area)/1000000
        var_new_fh = var_fh.replace('.tif', '_km3.tif')
        CreateGeoTiff(var_new_fh, var_area, driver, NDV, xsize, ysize, GeoT, Projection, explicit = False)
        var_new_fhs.append(var_new_fh)
    return var_new_fhs


def FlipDict(dictionary):
    dictb = dict((v,k) for k, v in dictionary.items())
    return dictb


def calc_basinmean(perc_fh, lu_fh):
    """
    Calculate the mean of a map after masking out the areas outside an basin defined by
    its landusemap.
    
    Parameters
    ----------
    perc_fh : str
        Filehandle pointing to the map for which the mean needs to be determined.
    lu_fh : str
        Filehandle pointing to landusemap.
    
    Returns
    -------
    percentage : float
        The mean of the map within the border of the lu_fh.
    """
    output_folder = tempfile.mkdtemp()
    perc_fh = MatchProjResNDV(lu_fh, np.array([perc_fh]), output_folder)
    EWR = OpenAsArray(perc_fh[0], nan_values = True)
    LULC = OpenAsArray(lu_fh, nan_values = True)
    EWR[np.isnan(LULC)] = np.nan
    percentage = np.nanmean(EWR)
    shutil.rmtree(output_folder)
    return percentage

def set_classes_to_value(fh, lu_fh, classes, value = 0):
    """
    Open a rasterfile and change certain pixels to a new value. Classes and
    lu_fh is used to create a mask. The mask is then used to the pixel values
    in fh to value.
    
    Parameters
    ----------
    fh : str
        Filehandle pointing to georeferenced tiff raster map.
    lu_fh : str
        Filehandle pointing to georeferenced tiff raster map. Should have same
        dimensions as fh.
    classes : list
        List with values, the values are looked up in lu_fh, the corresponding 
        pixels in fh are then changed.
    value : float or int, optional
        Value to change the pixelvalues in fh into.
    """
    ALPHA = OpenAsArray(fh, nan_values = True)
    LULC = OpenAsArray(lu_fh)
    mask = np.logical_or.reduce([LULC == x for x in classes])
    ALPHA[mask] = value
    driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(lu_fh)
    CreateGeoTiff(fh, ALPHA, driver, NDV, xsize, ysize, GeoT, Projection)

    
def GapFil(input_tif, footprint, output_folder, method = 'max'):
    """
    Gapfil a raster by filling with the minimum, maximum or median of nearby pixels.
    
    Parameters
    ----------
    input_tif : str
        Raster to be filled.
    footprint : ndarray
        Boolean array describing the area in which to look for the filling value.
    output_folder : str
        Folder to store gapfilled map.
    method : str, optional
        Method to use for gapfilling, options are 'max', 'min' or 'median'. Default is 'max'.
        
    Returns
    -------
    fh : str
        Location of the gapfilled map.
    """
    driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(input_tif)
    population = OpenAsArray(input_tif, nan_values = True)
    
    if method == 'median':
        population_gaps = ndimage.median_filter(population, footprint = footprint)
    if method == 'max':
        population_gaps = ndimage.maximum_filter(population, footprint = footprint)
    if method == 'min':
        population_gaps = ndimage.minimum_filter(population, footprint = footprint)
        
    population[np.isnan(population)] = population_gaps[np.isnan(population)]
    
    fn = os.path.split(input_tif)[1].replace('.tif','_gapfilled.tif')
    fh = os.path.join(output_folder, fn)
    
    CreateGeoTiff(fh, population, driver, NDV, xsize, ysize, GeoT, Projection)
    
    return fh
    
def CalcMeanStd(fhs, std_fh = None, mean_fh = None):
    """
    Calculate the mean and the standard deviation per pixel for a serie of maps.
    
    Parameters
    ----------
    fhs : ndarray
        Array with filehandles pointing to maps to be used.
    std_fh : str
        Filehandle indicating where to store the map with standard deviations.
    mean_fh : str
        Filehandle indiciating where to store the map with mean values.
        
    Returns
    -------
    std : ndarray
        Array with the standard deviation for each pixel.
    mean : ndarray
        Array with the mean for each pixel.
    """
    driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(fhs[0])
    
    data_sum = np.zeros((ysize,xsize))
    data_count = np.zeros((ysize,xsize))
    
    for fh in fhs:
        data = OpenAsArray(fh, nan_values = True)
        data_sum = np.nansum([data_sum, data], axis = 0)
        
        count = np.ones((ysize,xsize))
        count[np.isnan(data)] = 0
        data_count += count
    
    mean = data_sum / data_count
    data_sum = np.zeros((ysize,xsize))
    
    for fh in fhs:
        data = (OpenAsArray(fh, nan_values = True) - mean)**2 / data_count
        data_sum += data
        
    std = np.sqrt(data_sum)
    
    if std_fh:
        if not os.path.exists(os.path.split(std_fh)[0]):
            os.makedirs(os.path.split(std_fh)[0])
        CreateGeoTiff(std_fh, std, driver, NDV, xsize, ysize, GeoT, Projection)
        
    if mean_fh:
        if not os.path.exists(os.path.split(mean_fh)[0]):
            os.makedirs(os.path.split(mean_fh)[0]) 
        CreateGeoTiff(mean_fh, mean, driver, NDV, xsize, ysize, GeoT, Projection)
       
    return std, mean
    
def Multiply(fh1, fh2, fh3):
    """
    Multiply two maps with eachother and store the results in a new map.
    
    Parameters
    ----------
    fh1 : str
        Filehandle pointing to map to be multiplied with fh2.
    fh2 : str
        Filehandle pointing to map to be multiplied with fh1.
    fh3 : str
        Filehandle indicating where to store the results.
    """
    FH1 = OpenAsArray(fh1, nan_values = True)
    FH2 = OpenAsArray(fh2, nan_values = True)
    
    FH3 = FH1 * FH2
    
    if not os.path.exists(os.path.split(fh3)[0]):
        os.makedirs(os.path.split(fh3)[0])
    
    driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(fh1)
    CreateGeoTiff(fh3, FH3, driver, NDV, xsize, ysize, GeoT, Projection)
    
def GetGdalWarpInfo(fh, subdataset = 0):
    """
    Get information in string format from a geotiff or HDF4 file for use by GDALWARP.

    Parameters
    ----------
    fh : str
        Filehandle pointing to a geotiff or HDF4 file.
    subdataset = int, optional
        Value indicating a subdataset (in case of HDF4), default is 0.
        
    Returns
    -------
    srs : str
        The projection of the fh.
    ts : str
        Resolution of the fh.
    te : str
        Bounding box (xmin, ymin, xmax, ymax) of the fh.
    ndv : str
        No-Data-Value of the fh.
    """
    dataset = gdal.Open(fh, gdal.GA_ReadOnly)
    Type = dataset.GetDriver().ShortName
    if Type == 'HDF4':
        dataset = gdal.Open(dataset.GetSubDatasets()[subdataset][0])
    ndv = str(dataset.GetRasterBand(1).GetNoDataValue())
    if ndv == 'None':
        ndv = 'nan'
    srs = dataset.GetProjectionRef()
    if len(srs) == 0:
        srs = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'
        print "srs not defined, using EPSG4326."
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    ts = ' '.join([str(xsize), str(ysize)]) 
    GeoT = dataset.GetGeoTransform()
    xmin = GeoT[0]
    ymin = GeoT[3] + GeoT[5] * ysize
    xmax = GeoT[0] + GeoT[1] * xsize
    ymax = GeoT[3]
    te = ' '.join([str(xmin), str(ymin), str(xmax), str(ymax)])
    return srs, ts, te, ndv

def AverageSeries(tifs, dates, length, output_folder, para_name = 'Average', categories = None, lu_fh = None, timescale = 'months'):
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
    lu_fh : str, optional
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
    AssertMissingDates(dates, timescale = timescale)
    
    masked_average = False
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if type(length) == dict:
        max_length = np.max(length.values())
        masked_average = True
        AssertSameKeys([length, categories])
        AssertProjResNDV([tifs, np.array(lu_fh)])
    else:
        max_length = length
        AssertProjResNDV([tifs])
    
    driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(tifs[0])
    
    output_tifs = np.array([])
    
    for date in dates[(max_length-1):]:
        if masked_average:
            array = MaskedMovingAverage(date, tifs, dates, lu_fh, length, categories)
        if not masked_average:
            array = MovingAverage(date, tifs, dates, moving_avg_length = length)
        tif = os.path.join(output_folder, '{0}_{1}{2}.tif'.format(para_name, date.year, str(date.month).zfill(2)))
        CreateGeoTiff(tif, array, driver, NDV, xsize, ysize, GeoT, Projection)
        output_tifs = np.append(output_tifs, tif)
    
    return output_tifs, dates[(max_length-1):]

def MaskedMovingAverage(date, fhs, dates, lu_fh, moving_avg_length, categories, method = 'tail'):
    """
    Calculate temporal trailing averages dependant on landuse categories.

    Parameters
    ----------
    date : object
        datetime.date object indicating for which month the average needs to be calculated.
    fhs : ndarray
        Array with filehandles pointing to maps.
    dates : ndarray
        Array with datetime.date objects referring to the maps in fhs.
    lu_fh : str
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
    AssertSameKeys([moving_avg_length, categories])
    LULC = OpenAsArray(lu_fh, nan_values = True)
    driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(lu_fh)
    AVG = np.zeros((ysize, xsize)) * np.nan
    
    for length in np.unique(moving_avg_length.values()):
        key_list = [key for key in moving_avg_length.keys() if moving_avg_length[key] is int(length)]
        classes = list(Flatten([categories[key] for key in key_list]))
        mask = np.logical_or.reduce([LULC == value for value in classes])
        AVG[mask] = MovingAverage(date, fhs, dates, moving_avg_length = length, method = method)[mask]
    
    return AVG

def tifs_from_waterpix(waterpix_nc, variable, dates, output_folder):
    """
    Substract tifs from waterpix out or input.
    
    Parameters
    ----------
    waterpix_nc : str
        The waterpix nc file.
    variable : str
        The variable to extract.
    dates : ndarray
        Array with datetime.date objects describing which dates to extract.
    output_folder :  str
        Folder to store output.
        
    Returns
    -------
    variable_tifs : ndarray
        Array with paths to the new tifs.
        
    Examples
    --------
    >>> import dateutil.relativedelta as relativedelta
    >>> nummonths = 12
    >>> dates = [datetime.date(2008,1,1) + relativedelta.relativedelta(months = x) for x in range(0, nummonths)]
    >>> tifs_from_waterpix(r"C:\Users\cambodia.nc", 'storage_change', dates, r"D:\\Storage_Change_v2_2008")          
    """
    import davgis
    variable_tifs = np.array([])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for date in dates:
        year = str(date.year)
        month = str(date.month).zfill(2)
        time_value = int(year + month)
        time_value = int(year)
        output_tif = os.path.join(output_folder, '{0}_{1}.tif'.format(variable, time_value))
        spa_ref = davgis.Spatial_Reference(4326)
        davgis.NetCDF_to_Raster(input_nc=waterpix_nc,
                                output_tiff=output_tif,
                                ras_variable=variable,
                                x_variable='longitude', y_variable='latitude',
                                crs=spa_ref,
                                time={'variable': 'time', 'value': time_value})
        variable_tifs = np.append(variable_tifs, output_tif)
    return variable_tifs

def plot_category_areas(lu_fh, categories, output_fh, area_treshold = 0.01):
    """
    Plot the relative areas of landuse categories in a pie chart.
    
    Parameters
    ----------
    lu_fh : str
        Filehandle pointing to a landusemap
    categories : dict
        Dictionary specifying all the landuse categories.
    output_fh : str
        Filehandle indicating where to save the graph.
    area_treshold : float, optional
        Categories with a relative area lower than the treshold are not plotted
        in the pie chart. Default values is 0.01.
    """
    AREAS = MapPixelAreakm(lu_fh)
    LULC = OpenAsArray(lu_fh, nan_values = True)
    areas = dict()
    total_area = np.nansum(AREAS[~np.isnan(LULC)])
    
    for key in categories.keys():
        classes = categories[key]
        mask = np.logical_or.reduce([LULC == value for value in classes])
        area = np.nansum(AREAS[mask])
        if area / total_area >= area_treshold:
            areas[key] = area
    
    clrs = ['#6bb8cc','#87c5ad', '#9ad28d', '#acd27a', '#c3b683', '#d4988b', '#b98b89', '#868583', '#497e7c']
    plt.figure(figsize = (15,15))
    plt.clf()
    plt.title('Total Area ({0:.2f} ha)'.format(total_area/100))
    plt.pie(areas.values(), labels = areas.keys(), autopct = '%1.1f%%', colors = clrs)
    plt.savefig(output_fh)
    
def MovingAverage(date, filehandles, filedates, moving_avg_length = 5, method = 'tail'):
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
    summed_data = OpenAsArray(filehandles[indice]) * 0
    for fh in to_open:
        data = OpenAsArray(fh, nan_values = True)
        summed_data += data
    summed_data /= len(to_open)
    return summed_data

def SortFiles(input_dir, year_position, month_position = None, day_position = None, doy_position = None, extension = 'tif'):
    """
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
        
    Examples
    --------
    If input_dir contains the following files:
    
    >>> ["D:\project_ADB\Catchments\Srepok\sheet2\i_temp\I_2003_10.tif", 
     "D:\project_ADB\Catchments\Srepok\sheet2\i_temp\I_2003_11.tif"]
     
    Then year_position and month_position should be:
    
    >>> year_position = [-11,-7]
    month_position = [-6,-4]
    """
    dates = np.array([])
    years = np.array([])
    months = np.array([])
    days = np.array([])
    filehandles = np.array([])
    files = ListFilesInFolder(input_dir, extension = extension)
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

def CommonDates(dates_list):
    """
    Checks for common dates between multiple lists of datetime.date objects.
    
    Parameters
    ----------
    dates_list : list
        Contains lists with datetime.date objects.
        
    Returns
    -------
    common_dates : ndarray
        Array with datetime.date objects for common dates.
        
    Examples
    --------
    >>> dates_list = [p_dates, et_dates]
    
    >>> CommonDates([[datetime.date(2001,1,1), datetime.date(2001,2,1)], 
                     [datetime.date(2001,2,1), datetime.date(2001,3,1)]])
        np.array([datetime.date(2001,2,1)])
    
    """
    common_dates = dates_list[0]
    for date_list in dates_list[1:]:
        common_dates = np.sort(list(set(common_dates).intersection(date_list)))
    return common_dates

def AssertMissingDates(dates, timescale = 'months', quantity = 1):
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
    if timescale is 'months':
        while (current_date <= enddate):
            assert current_date in dates, "{0} is missing in the dataset".format(current_date)
            current_date =  current_date + relativedelta(months = quantity)

def ConvertDatetimeDate(dates, out = None):
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
        dates = np.array([datetime.datetime(date.year, date.month, date.day, 0,0,0) for date in dates])
    else:
        if type(dates[0]) is datetime.datetime:
            dates = np.array([datetime.date(dt.year, dt.month, dt.day) for dt in dates])
        elif type(dates[0]) is datetime.date:
            dates = np.array([datetime.datetime(date.year, date.month, date.day, 0,0,0) for date in dates])
        
    return dates
    
def Unzip(list_of_tuples):
    """
    Creates seperate lists from values inside tuples in a list.
    
    Parameters
    ----------
    list_of_tuples : list
        List containing tuples.
        
    Returns
    -------
    out : list
        List with arrays with the values of the tuples.
        
    Examples
    --------
    >>> Unzip([(2,3,4),(5,6,7),(1,2,3)])
    [np.array([2, 5, 1]), np.array([3, 6, 2]), np.array([4, 7, 3])]
    """
    out = [np.array(list(t)) for t in zip(*list_of_tuples)]
    return out

def MatchProjResNDV(source_file, target_fhs, output_dir, resample = 'near', dtype = 'float32', scale = None, ndv_to_zero = False):
    """
    Matches the projection, resolution and no-data-value of a list of target-files
    with a source-file and saves the new maps in output_dir.
    
    Parameters
    ----------
    source_file : str
        The file to match the projection, resolution and ndv with.
    target_fhs : ndarray
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
    s_srs, s_ts, s_te, s_ndv = GetGdalWarpInfo(source_file)
    output_files = np.array([])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for target_file in target_fhs:
        folder, fn = os.path.split(target_file)
        t_srs, t_ts, t_te, t_ndv = GetGdalWarpInfo(target_file)
        output_file = os.path.join(output_dir, fn)
        if not np.all([s_ts == t_ts, s_te == t_te, s_srs == t_srs, s_ndv == t_ndv]):
            string = '{10} -overwrite -t_srs {1} -te {2} -ts {3} -srcnodata {4} -dstnodata {5} -r {6} -ot {7} -of GTiff {8} {9}'.format(t_srs, s_srs, s_te, s_ts, t_ndv, s_ndv, resample, dtype, target_file, output_file, get_path('gdalwarp'))
            proc = subprocess.Popen(string, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            #print out, err
        else:
            shutil.copy2(target_file, output_file)
        output_files = np.append(output_files, output_file)
        if not np.any([scale == 1.0, scale == None, scale == 1]):
            driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(output_file)
            DATA = OpenAsArray(output_file, nan_values = True) * scale
            CreateGeoTiff(output_file, DATA, driver, NDV, xsize, ysize, GeoT, Projection)
        if ndv_to_zero:
            driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(output_file)
            DATA = OpenAsArray(output_file, nan_values = False)
            DATA[DATA == NDV] = 0.0
            CreateGeoTiff(output_file, DATA, driver, NDV, xsize, ysize, GeoT, Projection)
    return output_files

def GetGeoInfo(fh, subdataset = 0):
    """
    Substract metadata from a geotiff, HDF4 or netCDF file.
    
    Parameters
    ----------
    fh : str
        Filehandle to file to be scrutinized.
    subdataset : int, optional
        Layer to be used in case of HDF4 or netCDF format, default is 0.
        
    Returns
    -------
    driver : str
        Driver of the fh.
    NDV : float
        No-data-value of the fh.
    xsize : int
        Amount of pixels in x direction.
    ysize : int
        Amount of pixels in y direction.
    GeoT : list
        List with geotransform values.
    Projection : str
        Projection of fh.
    """
    SourceDS = gdal.Open(fh, gdal.GA_ReadOnly)
    Type = SourceDS.GetDriver().ShortName
    if Type == 'HDF4' or Type == 'netCDF':
        SourceDS = gdal.Open(SourceDS.GetSubDatasets()[subdataset][0])
    NDV = SourceDS.GetRasterBand(1).GetNoDataValue()
    xsize = SourceDS.RasterXSize
    ysize = SourceDS.RasterYSize
    GeoT = SourceDS.GetGeoTransform()
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    driver = gdal.GetDriverByName(Type)
    return driver, NDV, xsize, ysize, GeoT, Projection

def ListFilesInFolder(folder, extension='tif'):
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
    list_of_files = [os.path.join(folder,fn) for fn in next(os.walk(folder))[2] if fn.split('.')[-1] == extension]
    return list_of_files

def OpenAsArray(fh, bandnumber = 1, dtype = 'float32', nan_values = False):
    """
    Open a map as an numpy array. 
    
    Parameters
    ----------
    fh: str
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
    Array : ndarray
        Array with the pixel values.
    """
    datatypes = {"uint8": np.uint8, "int8": np.int8, "uint16": np.uint16, "int16":  np.int16, "Int16":  np.int16, "uint32": np.uint32,
    "int32": np.int32, "float32": np.float32, "float64": np.float64, "complex64": np.complex64, "complex128": np.complex128,
    "Int32": np.int32, "Float32": np.float32, "Float64": np.float64, "Complex64": np.complex64, "Complex128": np.complex128,}
    DataSet = gdal.Open(fh, gdal.GA_ReadOnly)
    Type = DataSet.GetDriver().ShortName
    if Type == 'HDF4':
        Subdataset = gdal.Open(DataSet.GetSubDatasets()[bandnumber][0])
        NDV = int(Subdataset.GetMetadata()['_FillValue'])
    else:
        Subdataset = DataSet.GetRasterBand(bandnumber)
        NDV = Subdataset.GetNoDataValue()
    Array = Subdataset.ReadAsArray().astype(datatypes[dtype])
    if nan_values:
        Array[Array == NDV] = np.nan
    return Array

def GetMonthLabels():
    """
    Function to create a dictionary with two digit month labels, alternative to 
    applying zfill(2) to a string.
    
    Returns
    -------
    month_labels : dict
        Dictionary with two digit months labels.
    """
    month_labels = {1:'01',2:'02',3:'03',4:'04',5:'05',6:'06',7:'07',8:'08',9:'09',10:'10',11:'11',12:'12'}
    return month_labels
    
def CreateGeoTiff(fh, Array, driver, NDV, xsize, ysize, GeoT, Projection, explicit = True):
    """
    Creates a geotiff from a numpy array.
    
    Parameters
    ----------
    fh : str
        Filehandle for output.
    Array: ndarray
        Array to convert to geotiff.
    driver : str
        Driver of the fh.
    NDV : float
        No-data-value of the fh.
    xsize : int
        Amount of pixels in x direction.
    ysize : int
        Amount of pixels in y direction.
    GeoT : list
        List with geotransform values.
    Projection : str
        Projection of fh.    
    """
    datatypes = {"uint8": 1, "int8": 1, "uint16": 2, "int16": 3, "Int16": 3, "uint32": 4,
    "int32": 5, "float32": 6, "float64": 7, "complex64": 10, "complex128": 11,
    "Int32": 5, "Float32": 6, "Float64": 7, "Complex64": 10, "Complex128": 11,}
    DataSet = driver.Create(fh,xsize,ysize,1,datatypes[Array.dtype.name])
    if NDV is None:
        NDV = -9999
    if explicit:
        Array[np.isnan(Array)] = NDV
    DataSet.GetRasterBand(1).SetNoDataValue(NDV)
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection(Projection.ExportToWkt())
    DataSet.GetRasterBand(1).WriteArray(Array)
    DataSet = None
    if "nt" not in Array.dtype.name:
        Array[Array == NDV] = np.nan

def PixelCoordinates(lon,lat,fh):
    """
    Find the corresponding pixel to a latitude and longitude.
    
    Parameters
    ----------
    lon : float or int
        Longitude to find.
    lat : float or int
        Latitude to find.
    fh : str
        Filehandle pointing to the file to be searched.
        
    Returns
    -------
    xpixel : int
        The index of the longitude.
    ypixel : int
        The index of the latitude.
    """
    SourceDS = gdal.Open(fh, gdal.GA_ReadOnly)
    xsize = SourceDS.RasterXSize
    ysize = SourceDS.RasterYSize
    GeoT = SourceDS.GetGeoTransform()
    assert (lon >= GeoT[0]) & (lon <= GeoT[0] + xsize * GeoT[1]), 'longitude is not on the map'
    assert (lat <= GeoT[3]) & (lat >= GeoT[3] + ysize * GeoT[5]), 'latitude is not on the map'
    location = GeoT[0]
    xpixel = -1
    while location <= lon:
        location += GeoT[1]
        xpixel += 1
    location = GeoT[3]
    ypixel = -1
    while location >= lat:
        location += GeoT[5]
        ypixel += 1
    return xpixel, ypixel   

def AssertProjResNDV(list_of_filehandle_lists, check_NDV = True):
    """ 
    Check if the projection, resolution and no-data-value of all provided filehandles are the same. 
    
    Parameters
    ----------
    list_of_filehandle_lists : list
        List with different ndarray containing filehandles to compare.
    check_NDV : boolean, optional
        Check or ignore the no-data-values, default is True.
        
    Examples
    --------   
    >>> AssertProjResNDV([et_fhs, ndm_fhs, p_fhs], check_NDV = True)
    """
    longlist = np.array([])
    for fh_list in list_of_filehandle_lists:
        if type(fh_list) is np.ndarray:
            longlist = np.append(longlist, fh_list)
        if type(fh_list) is str:
            longlist = np.append(longlist, np.array(fh_list))
    t_srs, t_ts, t_te, t_ndv = GetGdalWarpInfo(longlist[0])
    for fh in longlist[1:]:
        s_srs, s_ts, s_te, s_ndv = GetGdalWarpInfo(fh)
        if check_NDV:
            assert np.all([s_ts == t_ts, s_te == t_te, s_srs == t_srs, s_ndv == t_ndv]), "{0} does not have the same Proj/Res/NDV as {1}".format(longlist[0], fh)
        else:
            assert np.all([s_ts == t_ts, s_te == t_te, s_srs == t_srs]), "{0} does not have the same Proj/Res as {1}".format(longlist[0], fh)

def MapPixelAreakm(fh, approximate_lengths = False):
    """ 
    Calculate the area of the pixels in a geotiff.
    
    Parameters
    ----------
    fh : str
        Filehandle pointing to a geotiff.
    approximate_lengths : boolean, optional
        Give the approximate length per degree [km/deg] instead of the area [km2], default is False.
        
    Returns
    -------
    map_area : ndarray
        The area per cell.
    """
    driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(fh)
    AreaColumn = np.zeros((ysize,1))
    for y in range(ysize):
        P1 = LatLon.LatLon(GeoT[3] + y*GeoT[1], GeoT[0])
        P2 = LatLon.LatLon(float(str(P1.lat)), float(str(P1.lon)) + GeoT[1])
        P3 = LatLon.LatLon(float(str(P1.lat)) - GeoT[1], float(str(P1.lon)))
        P4 = LatLon.LatLon(float(str(P1.lat)) - GeoT[1], float(str(P1.lon)) + GeoT[1])
        u = P1.distance(P2)
        l = P3.distance(P4)
        h = P1.distance(P3)
        AreaColumn[y,0] = (u+l)/2*h
    map_area = np.repeat(AreaColumn, xsize, axis = 1)
    if approximate_lengths:
        pixel_approximation = np.sqrt(abs(GeoT[1]) * abs(GeoT[5]))
        map_area = np.sqrt(map_area) / pixel_approximation
    return map_area 

def ZonalStats(fhs, dates, output_dir, quantity, unit, location, color = '#6bb8cc'):
    """
    Calculate and plot some statictics of a timeseries of maps.
    
    Parameters
    ----------
    fhs : ndarray
        Filehandles pointing to maps.
    dates : ndarray
        Datetime.date object corresponding to fhs.
    output_dir : str
        Folder to save the graphs.
    quantity : str
        Quantity of the maps.
    unit : str
        Unit of the maps.
    location : str
        Location name of the maps.
    color : str, optional
        Color in which the graphs will be plotted, default is '#6bb8cc'.
        
    Returns
    -------
    monthly_average : float
        Monthly spatial average.
    yearly_average : float
        Yearly spatial average.
        
    Examples
    --------
    >>> ZonalStats(p_fhs, p_dates, output_dir, 'Precipitation', 'mm/month', 'North-Vietnam')
    
    >>> ZonalStats(et_fhs, et_dates, output_dir, 'Evapotranspiration', 'mm/month', 'South-Vietnam')
    
    """
    ts = np.array([])
    
    data_monthly_ts = dict()
    data_monthly_counter = dict()
    months = np.unique([date.month for date in dates])
    
    for month in months:
        data_monthly_ts[month] = 0
        data_monthly_counter[month] = 0
    
    data_yearly_ts = dict()
    data_yearly_counter = dict()
    years = np.unique([date.year for date in dates])
    
    for year in years:
        data_yearly_ts[year] = 0
        data_yearly_counter[year] = 0
    
    for date in dates:
        
        DATA = OpenAsArray(fhs[dates == date][0], nan_values = True)
        data = np.nanmean(DATA)
        ts = np.append(ts, data)
        data_monthly_ts[date.month] += data
        data_monthly_counter[date.month] += 1
        data_yearly_ts[date.year] += data
        data_yearly_counter[date.year] += 1
    
    monthly_ts = np.array(data_monthly_ts.values()) / np.array(data_monthly_counter.values())
    months = np.array(data_monthly_ts.keys())
    
    yearly_mask = np.array(data_yearly_counter.values()) == 12
    yearly_ts = np.array(data_yearly_ts.values())[yearly_mask] / np.array(data_yearly_counter.values())[yearly_mask]
    years = np.array(data_yearly_ts.keys())[yearly_mask]
      
    idx = np.argsort(dates)
      
    fig = plt.figure(figsize = (10,5))
    plt.clf()
    plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
    ax = plt.subplot(111)
    ax.plot(dates[idx], ts[idx], '-k')
    ax.fill_between(dates[idx], ts[idx], color = color)
    ax.set_xlabel('Time')
    ax.set_ylabel(quantity + ' ' + unit)
    ax.set_title(quantity + ', ' + location)
    fig.autofmt_xdate()
    [i.set_zorder(10) for i in ax.spines.itervalues()]
    plt.savefig(os.path.join(output_dir, quantity + '_' + location + '_ts.png'))
    plt.close(fig)
        
    fig = plt.figure(figsize = (10,5))
    plt.clf()
    plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
    ax = plt.subplot(111)
    ax.bar(months - 0.4, monthly_ts, 0.8, color = color)
    ax.set_xlabel('Time [month]')
    ax.set_xlim([0, max(months)+1])
    ax.set_xticks(months)
    ax.set_ylabel(quantity + ' ' + unit)
    ax.set_title('Monthly average ' + quantity + ', ' + location)
    [i.set_zorder(10) for i in ax.spines.itervalues()]
    plt.savefig(os.path.join(output_dir, quantity + '_' + location + '_monthly.png'))
    plt.close(fig)
        
    fig = plt.figure(figsize = (10,5))
    plt.clf()
    plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
    ax = plt.subplot(111)
    ax.bar(years - 0.4, yearly_ts, 0.8, color = color)
    ax.set_xlabel('Time [year]')
    ax.set_xlim([min(years) - 1, max(years)+1])
    ax.set_ylabel(quantity + ' ' + unit)
    ax.set_title('Yearly average ' + quantity + ', ' + location)
    [i.set_zorder(10) for i in ax.spines.itervalues()]
    plt.savefig(os.path.join(output_dir, quantity + '_' + location + '_yearly.png'))
    plt.close(fig)
    
    monthly_max = np.nanmax(monthly_ts)
    monthly_average = np.nanmean(monthly_ts)
    yearly_average = np.nanmean(yearly_ts)
    
    return monthly_max, monthly_average, yearly_average

def MergeDictionaries(list_of_dictionaries):
    """
    Merge multiple dictionaries into one, gives a warning if keys are 
    overwritten.
    
    Parameters
    ----------
    list_of_dictionaries : list
        List containing the dictionaries to merge.
        
    Returns
    -------
    merged_dict : dict
        The combined dictionary.
    """
    merged_dict = dict()
    expected_length = 0
    for dic in list_of_dictionaries:
        expected_length += len(dic.keys())
        merged_dict = dict(merged_dict.items() + dic.items())
    if expected_length is not len(merged_dict):
        print "WARNING: It seems some station(s) with similar keys have been overwritten ({0} != {1}), keys: {2}".format(expected_length, len(merged_dict))
    return merged_dict    

def AssertSameKeys(list_of_dictionaries):
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

def AssertPresentKeys(dictionary1, dictionary2):
    """
    Check if the keys of dictionary 1 are present in dictionary 2.
    
    Parameters
    ----------
    dictionary1 : dict
        Dictionary with keys.
    dictionary2 : dict
        Another dictionary with keys.
    """
    for key in dictionary1.keys():
        assert key in dictionary2.keys(), "{0} key is missing in dictionary"

def Flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in Flatten(el):
                yield sub
        else:
            yield el


def Ysum(fhs, fh3):
    """
    sum maps with each other and store the results in a new map.
   
    Parameters
    ----------
    fhs : list of maps to sum
    fh3 : str
        Filehandle indicating where to store the results.
    """
    FH3 = OpenAsArray(fhs[0], nan_values = True) * 0
    for fh in fhs:
        FH3 = np.nansum((FH3, OpenAsArray(fh, nan_values = True)),0)
   
    if not os.path.exists(os.path.split(fh3)[0]):
        os.makedirs(os.path.split(fh3)[0])
   
    driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(fhs[0])
    CreateGeoTiff(fh3, FH3, driver, NDV, xsize, ysize, GeoT, Projection)
    

def Aggregate(fhs, fh = None):
    """
    Calculate the sum of multiple geotiffs.
    
    Parameters
    ----------
    fhs : list
        List of filehandles to be aggregated.
    fh  : str, optional
        String specifying where to store output, default is None.
        
    Returns
    -------
    fh : str
        Filehandle specifying where the aggregated map has been stored.
    """
    AssertProjResNDV([fhs])
    if fh is None:
        temp_folder = tempfile.mkdtemp()
        fh = os.path.join(temp_folder,'temp.tif')
    DATA = OpenAsArray(fhs[0], nan_values = True)
    for i in range(1,len(fhs)):
        DATA += OpenAsArray(fhs[i], nan_values = True)
    driver, NDV, xsize, ysize, GeoT, Projection = GetGeoInfo(fhs[0])
    CreateGeoTiff(fh, DATA, driver, NDV, xsize, ysize, GeoT, Projection)
    return fh    

def ZeroesDictionary(dictionary):
    """
    Creates a dictionary with the same keys as the input dictionary, but
    all values are zero.
    
    Parameters
    ----------
    dictionary : dict
        dictionary to be copied.
    
    Returns
    -------
    null_dictionary : dict
        dictionary with zero values.
    """
    null_dictionary = dict()
    for key in dictionary.keys():
        null_dictionary[key] = 0.0
    return null_dictionary     