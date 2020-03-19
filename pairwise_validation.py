# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:32:47 2016

@author: Bert Coerver (b.coerver [at] un-ihe.org)
"""
from __future__ import print_function
from builtins import zip
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
import os
import WA_Hyperloop.becgis as becgis
from scipy import interpolate
import csv
import gdal
import datetime
from matplotlib.colors import LinearSegmentedColormap

def compare_rasters2stations(ds1_fhs, ds1_dates, station_dict, output_dir, station_names = None, quantity_unit = None, dataset_names = None, method = 'cubic', min_records = 1):
    """
    Compare a series of raster maps with station time series by computing
    the relative bias, RMAE, Pearson-correlation coefficient and 
    the Nash-Sutcliffe coefficient for each station.
    
    Parameters
    ----------
    ds1_fhs : 1dnarray
        List containing filehandles to georeferenced raster files.
    ds1_dates : 1dnarray
        List containing datetime.date or datetime.datetime objects corresponding
        to the filehandles in ds1_fhs. Lenght should be equal to ds1_fhs.
    station_dict : dictionary
        Dictionary containing coordinates of stations and timeseries. See examples
        below for an example
    output_dir : str, optional
        Directory to store several results, i.e. (1) a csv file to load in a GIS program, 
        (2) interpolated maps showing the various error indicators spatially and (3)
        scatter plots for all the stations.
    station_names : dictionary, optional
        Dictionary containing names of the respective stations which can be added to the csv-file, see
        Examples for more information.
    quantity_unit : list, optional
        List of two strings describing the quantity and unit of the data.
    dataset_name : list, optional
        List of strings describing the names of the datasets.
    method : str, optional
        Method used for interpolation of the error-indicators, i.e.: 'linear', 'nearest' or 'cubic' (default).
    
    Returns
    -------
    results : dictionary
        Dictionary containing several error indicators per station.

    Examples
    --------
    
    >>> station_dict = {(lat1, lon1): [(datetime.date(year, month, day), data_value), 
                                       (datetime.date(year, month, day), data_value), 
                                        etc.],
                        (lat2, lon2): [(datetime.date(year, month, day), data_value), 
                                       (datetime.date(year, month, day), data_value), 
                                        etc.],
                         etc.}
                    
    >>> station_names = {(lat1,lon1): 'stationname1', (lat2,lon2): 'stationname2', etc.}
    
    >>> results = compare_rasters2stations(ds1_fhs, ds1_dates, station_dict, output_dir = r"C:/Desktop",
                                station_names = None, quantity_unit = ["P", "mm/month"], 
                                dataset_names = ["CHIRPS", "Meteo Stations"], 
                                method = 'cubic')
    """
    results = dict()
    pixel_coordinates = list()
    
    if dataset_names is None:
        dataset_names = ['Spatial', 'Station']
    if quantity_unit is not None:
        quantity_unit[1] = r'[' + quantity_unit[1] + r']'
    else:
        quantity_unit = ['data', '']
        
    becgis.assert_proj_res_ndv([ds1_fhs])
    no_of_stations = len(list(station_dict.keys()))
    ds1_dates = becgis.convert_datetime_date(ds1_dates, out = 'datetime')

    for i, station in enumerate(station_dict.keys()):
        
        station_dates, station_values = unzip(station_dict[station])
        common_dates = becgis.common_dates([ds1_dates, station_dates])
        sample_size = common_dates.size
        
        if sample_size >= min_records:
            ds1_values = list()
            xpixel, ypixel = pixelcoordinates(station[0], station[1], ds1_fhs[0])
            
            if np.any([np.isnan(xpixel), np.isnan(ypixel)]):
                print("Skipping station ({0}), cause its not on the map".format(station))
                continue
            else:
                for date in common_dates:
                    ds1_values.append(becgis.open_as_array(ds1_fhs[ds1_dates == date][0], nan_values = True)[ypixel, xpixel])
                    
                common_station_values = [station_values[station_dates == date][0] for date in common_dates]
                
                results[station] = pairwise_validation(ds1_values, common_station_values)
                results[station] += (sample_size,)
                         
                pixel_coordinates.append((xpixel, ypixel))
                #m, b = np.polyfit(ds1_values, common_station_values, 1)  
                
                path_scatter = os.path.join(output_dir, 'scatter_plots')
                if not os.path.exists(path_scatter):
                    os.makedirs(path_scatter)
                    
                path_ts = os.path.join(output_dir, 'time_series')
                if not os.path.exists(path_ts):
                    os.makedirs(path_ts)
                    
                path_int = os.path.join(output_dir, 'interp_errors')
                if not os.path.exists(path_int):
                    os.makedirs(path_int)
                
                xlabel = '{0} {1} {2}'.format(dataset_names[0], quantity_unit[0], quantity_unit[1])
                ylabel = '{0} {1} {2}'.format(dataset_names[1], quantity_unit[0], quantity_unit[1])
                if station_names is not None:
                    title = station_names[station]
                    fn = os.path.join(path_scatter,'{0}_vs_{1}.png'.format(station_names[station], dataset_names[0]))
                    fnts = os.path.join(path_ts,'{0}_vs_{1}.png'.format(station_names[station], dataset_names[0]))
                else:
                    title = station
                    fn = os.path.join(path_scatter,'{0}_vs_station_{1}.png'.format(dataset_names[0],i))
                    fnts = os.path.join(path_ts,'{0}_vs_station_{1}.png'.format(dataset_names[0],i)) 
                suptitle = 'pearson: {0:.5f}, rmse: {1:.5f}, ns: {2:.5f}, bias: {3:.5f}, n: {4:.0f}'.format(results[station][0],results[station][1],results[station][2],results[station][3],results[station][4])
                plot_scatter_series(ds1_values, common_station_values, xlabel, ylabel, title, fn, suptitle = suptitle, dates = common_dates)
    
                xaxis_label = '{0} {1}'.format(quantity_unit[0], quantity_unit[1])
                xlabel = '{0}'.format(dataset_names[0])
                ylabel = '{0}'.format(dataset_names[1])
                plot_time_series(ds1_values,common_station_values,common_dates,xlabel,ylabel,xaxis_label, title, fnts, suptitle = suptitle)
                
                print("station {0} ({3}) of {1} finished ({2} matching records)".format(i+1, no_of_stations, sample_size, title))
        else:
            print("____station {0} of {1} skipped____ (less than {2} matching records)".format(i+1, no_of_stations, min_records))
            continue
    
    n = len(results)
    csv_filename = os.path.join(output_dir, '{0}stations_vs_{1}_indicators.csv'.format(n, dataset_names[0]))
    with open(csv_filename, 'wb') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(['longitude','latitude','station_id','pearson','rmse','nash_sutcliffe','bias', 'no_of_samples'])
        for station in list(results.keys()):
            writer.writerow([station[1], station[0], station_names[station], results[station][0],results[station][1],results[station][2],results[station][3],results[station][4]])

    rslt = {'Relative Bias':list(),'RMSE':list(),'Pearson Coefficient':list(),'Nash-Sutcliffe Coefficient':list(),'Number Of Samples':list()}

    for value in list(results.values()):
        rslt['Relative Bias'].append(value[3])
        rslt['RMSE'].append(value[1])
        rslt['Pearson Coefficient'].append(value[0])
        rslt['Nash-Sutcliffe Coefficient'].append(value[2])
        rslt['Number Of Samples'].append(value[4])

    for key, value in list(rslt.items()):
        title = '{0}'.format(key)
        print(title)
        if key is 'RMSE':
            xlabel = '{0} [mm/month]'.format(key)
        else:
            xlabel = key
        value = np.array(value)
        value = value[(~np.isnan(value)) & (~np.isinf(value))]
        suptitle = 'mean: {0:.5f}, std: {1:.5f}, n: {2}'.format(np.nanmean(value), np.nanstd(value), n)
        print(value)
        plot_histogram(value[(~np.isnan(value)) & (~np.isinf(value))], title, xlabel, output_dir, suptitle = suptitle)
    
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.get_geoinfo(ds1_fhs[0])
    dummy_map = becgis.open_as_array(ds1_fhs[0])
    grid = np.mgrid[0:ysize, 0:xsize]
    var_names = ['pearson', 'rmse', 'ns', 'bias', 'no_of_samples']

    for i, var in enumerate(unzip(list(results.values()))):
        xy = np.array(pixel_coordinates)[~np.isnan(var)]
        z = var[~np.isnan(var)]
        interpolation_field = interpolate.griddata(xy, z, (grid[1], grid[0]), method=method, fill_value = np.nanmean(z))
        interpolation_field[dummy_map == NDV] = NDV
        fh = os.path.join(path_int, '{0}_{1}stations_vs_{2}.tif'.format(var_names[i], len(xy), dataset_names[0]))
        becgis.create_geotiff(fh, interpolation_field, driver, NDV, xsize, ysize, GeoT, Projection)

    return results

def compare_rasters2rasters(ds1_fhs, ds1_dates, ds2_fhs, ds2_dates, output_dir = None, dataset_names = None, data_treshold = 0.75):
    """ 
    Compare two series of raster maps by computing
    the relative bias, RMAE, Pearson-correlation coefficient and
    the Nash-Sutcliffe coefficient per pixel.
    
    Parameters
    ----------
    ds1_fhs : list
        list pointing to georeferenced raster files of dataset 1.
    ds1_dates : list
        list corresponding to ds1_fhs specifying the dates.
    ds2_fhs : list
        list pointing to georeferenced raster files of dataset 2.
    ds2_dates : list
        list corresponding to ds2_fhs specifying the dates.
    quantity_unit  : list, optional
        list of two strings describing the quantity and unit of the data. e.g. ['Precipitation', 'mm/month'].
    dataset_names : list, optional
        list of strings describing the names of the datasets. e.g. ['CHIRPS', 'ERA-I'].
    output_dir : list, optional
        directory to store some results, i.e. (1) a graph of the spatially averaged datasets trough time and the
        bias and (2) 4 geotiffs showing the bias, nash-sutcliffe coefficient, pearson coefficient and rmae per pixel.
    data_treshold : float, optional
        pixels with less than data_treshold * total_number_of_samples actual values are set to no-data, i.e. pixels with
        too few data points are ignored.
        
    Returns
    -------
    results : dict
        dictionary with four keys (relative bias, RMAE, Pearson-correlation coefficient and 
        the Nash-Sutcliffe) with 2dnarrays of the values per pixel.
        
    Examples
    --------
    >>> results = compare_rasters2rasters(ds1_fhs, ds1_dates, ds2_fhs, ds2_dates, 
                                          output_dir = r"C:/Desktop/", quantity_unit = ["P", "mm/month"], 
                                          dataset_names = ["CHIRPS", "TRMM"])
    """
    becgis.assert_proj_res_ndv([ds1_fhs, ds2_fhs])
    
    if dataset_names is None:
        dataset_names = ['DS1','DS2']
    
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.get_geoinfo(ds1_fhs[0])
    
    common_dates = becgis.common_dates([ds1_dates, ds2_dates])
    
    diff_sum = np.zeros((ysize,xsize))
    non_nans = np.zeros((ysize,xsize))
    
    progress = 0 
    samples = len(common_dates)
    
    for date in common_dates:
        
        DS1 = becgis.open_as_array(ds1_fhs[ds1_dates == date][0], nan_values = True)
        DS2 = becgis.open_as_array(ds2_fhs[ds2_dates == date][0], nan_values = True)
        
        DS1[np.isnan(DS2)] = np.nan
        DS2[np.isnan(DS1)] = np.nan
        
        non_nans[~np.isnan(DS1)] += np.ones((ysize,xsize))[~np.isnan(DS1)]
        
        diff = (DS1 - DS2)**2
        diff_sum[~np.isnan(DS1)] += diff[~np.isnan(DS1)]
        
        progress += 1
        print("progress: {0} of {1} finished".format(progress, samples))

    diff_sum[non_nans <= data_treshold*samples] = np.nan
    results = dict()
    results['rmse'] = np.where(non_nans == 0., np.nan, np.sqrt(diff_sum / non_nans))
    
    startdate = common_dates[0].strftime('%Y%m%d')
    enddate = common_dates[-1].strftime('%Y%m%d')
    
    path = os.path.join(output_dir, 'spatial_errors')
    if not os.path.exists(path):
        os.makedirs(path)
        
    if output_dir is not None:
        for varname in list(results.keys()):
            fh = os.path.join(path, '{0}_{1}_vs_{2}_{3}_{4}.tif'.format(varname, dataset_names[0], dataset_names[1], startdate, enddate))
            becgis.create_geotiff(fh, results[varname], driver, NDV, xsize, ysize, GeoT, Projection)

    return results 

def compare_rasters2rasters_per_lu(ds1_fhs, ds1_dates, ds2_fhs, ds2_dates, lu_fh, output_dir, dataset_names = ["DS1", "DS2"], class_dictionary = None, no_of_classes = 6):
    """
    Compare two raster datasets with eachother per different landuse categories.
    
    Parameters
    ----------
    ds1_fhs : ndarray
        Array with strings pointing to maps of dataset 1.
    ds1_dates : ndarray
        Array with same shape as ds1_fhs, containing datetime.date objects.
    ds2_fhs : ndarray
        Array with strings pointing to maps of dataset 2.
    ds2_dates : ndarray
        Array with same shape as ds2_fhs, containing datetime.date objects.
    lu_fh : str
        Pointer to a landusemap.
    output_dir : str
        Map to save results.
    dataset_names : list, optional
        List with two strings describing the names of the two datasets.
    class_dictionary : dict
        Dictionary specifying all the landuse categories.
    no_of_classes : int
        The 'no_of_classes' most dominant classes in the the lu_fh are compared, the rest is ignored.
    
    """
    LUCS = becgis.open_as_array(lu_fh, nan_values = True)
    DS1 = becgis.open_as_array(ds1_fhs[0], nan_values = True)
    DS2 = becgis.open_as_array(ds2_fhs[0], nan_values = True)
    
    DS1[np.isnan(DS2)] = np.nan
    LUCS[np.isnan(DS1)] = np.nan
    
    classes, counts = np.unique(LUCS[~np.isnan(LUCS)], return_counts = True)
    counts_sorted = np.sort(counts)[-no_of_classes:]
    selected_lucs = [classes[counts == counter][0] for counter in counts_sorted]
    
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.get_geoinfo(lu_fh)
    becgis.create_geotiff(lu_fh.replace('.tif','_.tif'), LUCS, driver, NDV, xsize, ysize, GeoT, Projection)

    common_dates = becgis.common_dates([ds1_dates, ds2_dates])
    
    ds1_totals = np.array([])
    ds2_totals = np.array([])
    
    DS1_per_class = dict()
    DS2_per_class = dict()
    
    for date in common_dates:
        
        DS1 = becgis.open_as_array(ds1_fhs[ds1_dates == date][0], nan_values = True)
        DS2 = becgis.open_as_array(ds2_fhs[ds2_dates == date][0], nan_values = True)
        
        for clss in selected_lucs:
            
            if clss in list(DS1_per_class.keys()):
                DS1_per_class[clss] = np.append(DS1_per_class[clss], np.nanmean(DS1[LUCS == clss]))
            else:
                DS1_per_class[clss] = np.array([np.nanmean(DS1[LUCS == clss])])
                
            if clss in list(DS2_per_class.keys()):
                DS2_per_class[clss] = np.append(DS2_per_class[clss], np.nanmean(DS2[LUCS == clss]))
            else:
                DS2_per_class[clss] = np.array([np.nanmean(DS2[LUCS == clss])])

        ds1_totals = np.append(ds1_totals, np.nanmean(DS1))
        ds2_totals = np.append(ds2_totals, np.nanmean(DS2))
        
        print("Finished {0}, going to {1}".format(date, common_dates[-1]))
    
    for clss in selected_lucs:
        
        if class_dictionary is None:
            plot_scatter_series(DS1_per_class[clss], DS2_per_class[clss], dataset_names[0], dataset_names[1], clss, output_dir)
        else:
            cats = {v[0]: k for k, v in list(class_dictionary.items())}
            plot_scatter_series(DS1_per_class[clss], DS2_per_class[clss], dataset_names[0], dataset_names[1], cats[clss], output_dir)
            
    plot_scatter_series(ds1_totals, ds2_totals, dataset_names[0], dataset_names[1], "Total Area", output_dir)

    if class_dictionary is not None:
        output_fh = os.path.join(output_dir, 'landuse_percentages.png')
        driver, NDV, xsize, ysize, GeoT, Projection = becgis.get_geoinfo(lu_fh)
        becgis.create_geotiff(lu_fh.replace('.tif','_.tif'), LUCS, driver, NDV, xsize, ysize, GeoT, Projection)
        becgis.plot_category_areas(lu_fh.replace('.tif','_.tif'), class_dictionary, output_fh, area_treshold = 0.01)
        os.remove(lu_fh.replace('.tif','_.tif'))
        
def plot_scatter_series(x,y,xlabel,ylabel,title, output_dir, suptitle = None, dates = None):
    """
    Plot a scatter plot of two datasets with a fitted line trough it.
    
    Parameters
    ----------
    x : 1darray
        Array with values for dataset 1.
    y : 1darray
        Array with values for dataset 2.
    xlabel : str
        Label to put on the x-axis.
    ylabel : str
        Label to put on the y-axis.
    title : str
        Title to put above the graph.
    output_dir : str
        Folder or path to store graph.
    """
    maxi = np.nanmax([np.nanmax(x),np.nanmax(y)])*1.1
    mini = np.nanmin([np.nanmin(x),np.nanmin(y), 0.0])*1.1
    m, b = np.polyfit(x, y, 1)  
    if dates != None:
        C = np.array([date.month for date in dates])
        clrs = ['#6bb8cc','#87c5ad', '#9ad28d', '#acd27a', '#c3b683', '#d4988b', '#b98b89', '#868583', '#497e7c']
        cmap = LinearSegmentedColormap.from_list('LUC', clrs, N = 12)
    else:
        cmap = 'NaN'
        C = 'b'
    plt.figure(1, figsize = (10,10))
    plt.clf()
    plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(x, y, c = C, cmap = cmap, marker = '.', alpha = 1.0, lw = 0.0, s = 500, vmin = 0.5, vmax = 12.5)
    plt.plot([mini, maxi],[mini, maxi], '--k')
    plt.plot([mini, maxi], [m*mini + b, m*maxi + b], '-r', label = '{0:.2f} * x + {1:.2f}'.format(m,b))
    plt.ylim([mini, maxi])
    plt.xlim([mini, maxi])
    plt.legend(loc='upper left')
    if dates != None:
        cbar = plt.colorbar(label = 'Month')
        cbar.set_ticks(list(range(1,13)))
        cbar.set_ticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    if suptitle:
        plt.suptitle(suptitle)
    if output_dir.split('.')[-1] == 'png':
        plt.savefig(output_dir)
    else:
        plt.savefig(os.path.join(output_dir, '{0}.png'.format(title)))

def plot_time_series(x,y,dates,xlabel,ylabel,xaxis_label, title, output_dir, suptitle = None):
    plt.figure(2, figsize = (13,5))
    plt.clf()
    plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
    plt.plot(dates, x, '-k')
    plt.fill_between(dates,x, color = '#6bb8cc', label = xlabel)
    plt.plot(dates, y, color = '#c64345', label = ylabel)
    plt.scatter(dates,y,color= '#c64345')
    maxi = np.max([x,y]) * 1.1
    plt.xlim([np.min(dates),np.max(dates)])
    plt.ylim([0, maxi])
    plt.xlabel('Time')
    plt.ylabel(xaxis_label)
    plt.title(title)
    if suptitle:
        plt.suptitle(suptitle)
    plt.legend()
    if output_dir.split('.')[-1] == 'png':
        plt.savefig(output_dir)
    else:
        plt.savefig(os.path.join(output_dir, '{0}.png'.format(title)))

def plot_histogram(values, title, xlabel, output_dir, suptitle = None):
    values = np.array(values)
    
    #mini = np.nanmin(values)
    #maxi = np.nanmax(values)
    #bins = np.arange(mini, maxi, (maxi - mini)/(len(values) / 10.))
    
    plt.figure(3, figsize = (10,10))
    plt.clf()
    plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
    plt.hist(values[~np.isnan(values)], color = '#a3db76')
    plt.title(title)
    if suptitle:
        plt.suptitle(suptitle)
    plt.xlabel(xlabel)
    plt.ylabel('Number of Stations [-]')
    if output_dir.split('.')[-1] == 'png':
        plt.savefig(output_dir)
    else:
        plt.savefig(os.path.join(output_dir, '{0}_histogram.png'.format(title)))
        
def create_dict_entry(csv_fh):
    """Opens a CSV-file and return the station_name, a list with (datetime.datetime, value) 
    tuples and the coordinates of the station.

    Parameters
    ----------
    csv_fh : str
        filehandle pointing to a CSV-file with station data. See examples
        for the required CSV-format.
        
    Returns
    -------
    coordinates : tuple
        Tuple with the latitude and longitude of the station.
    data : list
        List with tuples containing a datetime.datetime object and a value.
    station_name : str
        Name of the station, derived from the CSV's filename.
        
    Examples
    --------
    The CSV-filename should be the station name and the file should 
    be formatted as follows:
    
    >>> lat:;<latitude>;lon:;<longitude>;<unit>
    datetime;year;month;day;data
    <datetime.datetime>;<year>;<month>;<day>;<value>
    <datetime.datetime>;<year>;<month>;<day>;<value>
    etc. 
    
    or
    
    >>> lat:;16.21666667;lon:;107.2833333;mm/month
    datetime;year;month;day;data
    1976-01-01 00:00:00;1976;1;1;89.89999999999999
    1976-02-01 00:00:00;1976;2;1;0.5
    etc.
    """
    fh = open(csv_fh)
    reader = csv.reader(fh, delimiter=';')
    data = list()
    for i, row in enumerate(reader):
        if i == 0:
            coordinates = (float(row[1]), float(row[3]))
            unit = row[4]
        elif i == 1:
            pass
        else:
            try:
                time = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            except:
                try:
                    time = datetime.datetime.strptime(row[0], "%d-%m-%Y %H:%M:%S")
                except:
                    print("date has wrong format for {0}".format(csv_fh))
            data.append((time, float(row[4])))
    fh.close()
    fn = os.path.split(csv_fh)[1]
    station_name = fn.split('.')[0]
    return coordinates, data, station_name, unit
    
def create_dictionary(csv_fhs):
    """
    Opens multiple CSV-files and returns dictionaries to be used by 
    compare_rasters2stations.
    
    Parameters
    ----------
    csv_fhs : list
        List containing filehandles pointing to CSV-files, where each file
        contains data for one station. See examples for the required CSV-
        format.
        
    Returns
    -------
    station_dict : dict
        Dictionary with the timeseries for all the stations.
        
    station_names : dict
        Dictionary with the names of all the stations.
        
    Examples
    --------
    >>> lat:;<latitude>;lon:;<longitude>;<unit>
    datetime;year;month;day;data
    <datetime.datetime>;<year>;<month>;<day>;<value>
    <datetime.datetime>;<year>;<month>;<day>;<value>
    etc. 
    
    or
    
    >>> lat:;16.21666667;lon:;107.2833333;mm/month
    datetime;year;month;day;data
    1976-01-01 00:00:00;1976;1;1;89.89999999999999
    1976-02-01 00:00:00;1976;2;1;0.5
    etc.
    """
    station_dict = dict()
    station_names = dict()
    names = list()
    for fh in csv_fhs:
        print(fh)
        coordinates, data, station_name, unit = create_dict_entry(fh)
        if station_name in names:
            print("WARNING: station with name {0} already present in dataset".format(station_name))
        if coordinates in list(station_dict.keys()):
            print("WARNING: station with coordinates {0} already present in dataset".format(coordinates))
        names.append(station_name)
        station_dict[coordinates] = data
        station_names[coordinates] = station_name
        
    return station_dict, station_names
    
def merge_dictionaries(list_of_dictionaries):
    """
    Merges multiple dictionaries into one, gives a warning if keys are 
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
        expected_length += len(list(dic.keys()))
        merged_dict = dict(list(merged_dict.items()) + list(dic.items()))
    if expected_length is not len(merged_dict):
        print("WARNING: It seems some station(s) with similar keys have been overwritten ({0} != {1}), keys: {2}".format(expected_length, len(merged_dict)))
    return merged_dict
    
def error(ds1,ds2):
    """
    Calculate the elementwise absolute errors between two series. 
    
    Parameters
    ----------
    ds1 : list
        List of values.
    ds2 : list
        List of values to compare with ds1, should be equal length.
        
    Returns
    -------
    errors : ndarray
        List of the elementwise absolute errors (i.e. ds1 - ds2).
    mean_error : float
        The mean of the elementwise absolute errors.
    std_error : float
        The standard deviation of the elementwise absolute errors.
        
    """
    station = np.array(ds1)
    satellite = np.array(ds2)
    station[np.isnan(ds2)] = np.nan
    satellite[np.isnan(ds1)] = np.nan
     
    errors = ds2 - ds1
    mean_error = np.nanmean(errors)
    std_error = np.nanstd(errors)
    return errors, mean_error, std_error
    
def pearson_correlation(ds1,ds2):
    """
    Calculate the pearson correlation coefficient for two series. 
    
    Parameters
    ----------
    ds1 : list
        List of values.
    ds2 : list
        List of values to compare with ds1, should be equal length.
        
    Returns
    -------
    pearson : float
        The pearson correlation coefficient.
        
    """
    ds1 = np.array(ds1)
    ds2 = np.array(ds2)
    ds1[np.isnan(ds2)] = np.nan
    ds2[np.isnan(ds1)] = np.nan
    
    ds1_min_mean = ds1 - np.nanmean(ds1)
    ds2_min_mean = ds2 - np.nanmean(ds2)
    pearson = np.nansum(ds1_min_mean * ds2_min_mean) / (np.sqrt(np.nansum(ds1_min_mean**2)) * np.sqrt(np.nansum(ds2_min_mean**2)))
    return pearson

def RMSE(ds1,ds2):
    """
    Calculate the RMSE for two series. 
    
    Parameters
    ----------
    ds1 : list
        List of values.
    ds2 : list
        List of values to compare with ds1, should be equal length.
        
    Returns
    -------
    rmse : float
        The RMSE.
        
    """
    ds1 = np.array(ds1)
    ds2 = np.array(ds2)
    ds1[np.isnan(ds2)] = np.nan
    ds2[np.isnan(ds1)] = np.nan
    
    mse = np.sqrt(np.nanmean((ds1 - ds2)**2))
    
    return mse
    
def RMAE(ds1, ds2):
    """
    Calculate the RMAE for two series. 
    
    Parameters
    ----------
    ds1 : list
        List of values.
    ds2 : list
        List of values to compare with ds1, should be equal length.
        
    Returns
    -------
    rmae : float
        The RMAE.
        
    """
    ds1 = np.array(ds1)
    ds2 = np.array(ds2)
    ds1[np.isnan(ds2)] = np.nan
    ds2[np.isnan(ds1)] = np.nan
 
    rmae = (1. / np.nansum(ds1)) * (np.nansum(abs(ds2-ds1))/np.nanmean(ds1))
    return rmae

def nash_sutcliffe(ds1, ds2):
    """
    Calculate the nash-sutcliffe coefficient for two series. 
    
    Parameters
    ----------
    ds1 : list
        List of values.
    ds2 : list
        List of values to compare with ds1, should be equal length.
        
    Returns
    -------
    ns : float
        The nash-sutcliffe coefficient.
        
    """
    ds1 = np.array(ds1)
    ds2 = np.array(ds2)
    ds1[np.isnan(ds2)] = np.nan
    ds2[np.isnan(ds1)] = np.nan
     
    ns = 1. - np.nansum((ds2 - ds1)**2) / np.nansum((ds1 - np.nanmean(ds1))**2)
    return ns
    
def bias(ds1,ds2):
    """
    Calculate the relative bias for two series. 
    
    Parameters
    ----------
    ds1 : list
        List of values.
    ds2 : list
        List of values to compare with ds1, should be equal length.
        
    Returns
    -------
    b : float
        The relative bias.
        
    """
    ds1 = np.array(ds1)
    ds2 = np.array(ds2)
    ds1[np.isnan(ds2)] = np.nan
    ds2[np.isnan(ds1)] = np.nan
     
    b = np.nansum(ds2)/np.nansum(ds1)
    return b
    
def pairwise_validation(ds1,ds2):
    """
    Calculate the relative bias, RMSE, Pearson-correlation coefficient and 
    the Nash-Sutcliffe coefficient for two series. 
    
    Parameters
    ----------
    ds1 : list
        List of values.
    ds2 : list
        List of values to compare with ds1, should be equal length.
        
    Returns
    -------
    pearson : float
        The pearson correlation coefficient.
    b : float
        The relative bias.
    ns : float
        The nash-sutcliffe coefficient.
    rmse : float
        The RMSE.
        
    """
    pearson = pearson_correlation(ds1,ds2)
    ns = nash_sutcliffe(ds1, ds2)
    b = bias(ds1,ds2)
    rmse = RMSE(ds1,ds2)
    return pearson, rmse, ns, b
    
def unzip(list_of_tuples):
    """
    Create lists for seperate entries in a list of tuples. 
    
    Parameters
    ----------
    list_of_tuples : list
        List of tuples, each tuple must be of the same length.
        
    Returns
    -------
    out : list
        List of the first value in each tuple up to a list containing the
        last value in each tuple.
        
    """    
    out = [np.array(list(t)) for t in zip(*list_of_tuples)]
    return out

def pixelcoordinates(lat,lon,rasterfile):
    """
    Function to find the corresponding pixel to a latitude and longitude.
    
    Parameters
    ----------
    lat : float or int
        Latitude in same unit as provided map, usually decimal degrees.
    lon : float or int
        Longitude in same unit as provided map, usually decimal degrees.
    rasterfile : str
        Filehandle pointing to georeferenced rasterfile.
        
    Returns
    -------
    xpixel : int
        The column in which the coordinate is situated.
    ypixel : int
        The row in which the coordinate is situated.
        
    Examples
    --------
    >>> xpixel, ypixel = pixelcoordinates(15.2, 120, r"C:/Desktop/map.tif")
    >>> xpixel
    40
    >>> ypixel
    24
    """
    SourceDS = gdal.Open(rasterfile, gdal.GA_ReadOnly)
    xsize = SourceDS.RasterXSize
    ysize = SourceDS.RasterYSize
    GeoT = SourceDS.GetGeoTransform()
    if np.all([lon >= GeoT[0], lon <= GeoT[0] + xsize * GeoT[1], lat <= GeoT[3], lat >= GeoT[3] + ysize * GeoT[5]]):
    #assert (lon >= GeoT[0]) & (lon <= GeoT[0] + xsize * GeoT[1]), 'longitude is not on the map {0}'.format((lat,lon))
    #assert (lat <= GeoT[3]) & (lat >= GeoT[3] + ysize * GeoT[5]), 'latitude is not on the map {0}'.format((lat,lon))
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
    else:
        print('longitude or latitude is not on the map {0}, returning NaNs'.format((lat,lon)))
        xpixel = np.nan
        ypixel = np.nan
    return xpixel, ypixel

def get_timeseries_raster(ds1_fhs, ds1_dates, coordinates, output_fh, unit = 'm3/s'):
    """
    Substract a timeseries from a set of raster files. Store results in a csv-file.
    
    Parameters
    ----------
    ds1_fhs : 1dnarray
        List containing filehandles to georeferenced raster files.
    ds1_dates : 1dnarray
        List containing datetime.date or datetime.datetime objects corresponding
        to the filehandles in ds1_fhs. Lenght should be equal to ds1_fhs.
    coordinates : tuple
        Tuple with the latitude and longitude, (lat, lon).
    output_fh : str
        Filehandle pointing to a csv-file.
    unit : str, optional
        String indicating the unit of the data, default is 'm3/s'.
    """
    ds1_values = list()   
    xpixel, ypixel = pixelcoordinates(coordinates[0], coordinates[1], ds1_fhs[0])
    
    if np.any([np.isnan(xpixel), np.isnan(ypixel)]):
        print("Coordinates ({0}) not on the map".format(coordinates))
    else:
        for date in ds1_dates:
            ds1_values.append(becgis.open_as_array(ds1_fhs[ds1_dates == date][0], nan_values = True)[ypixel, xpixel])
        
        ds1_values = np.array(ds1_values)    
        
        csv_file = open(output_fh, 'wb')
        writer = csv.writer(csv_file, delimiter=';')
        
        writer.writerow(['lat:',coordinates[0], 'lon:', coordinates[1], unit])
        writer.writerow(['datetime','year','month','day','data'])
        
        for date in ds1_dates:
            
            year = date.year
            month = date.month
            day = date.day
            
            dt = datetime.datetime(year, month, day, 0,0,0)
            data = ds1_values[ds1_dates == date][0]
            writer.writerow([dt, year, month, day, data])
        
        csv_file.close()
