# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:06:46 2016

@author: Bert Coerver (b.coerver[at]unesco-ihe.org)
"""
import WA_Hyperloop.becgis as becgis
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from scipy import ndimage
import WA_Hyperloop.get_dictionaries as gd
import wa

def create_sheet2(data, metadata, output_dir):
    
    output_dir = os.path.join(output_dir, metadata['name'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lulc_dict = gd.get_lulcs(lulc_version = '4.0')
    classes_dict = gd.get_sheet2_classes(version = '1.0')
    
    monthly_csvs, yearly_csvs = create_sheet2_csv(lulc_dict, classes_dict, metadata['lu'], data['et'][0], data['et'][1], data['t'][0], data['t'][1], data['i'][0], data['i'][1], output_dir, catchment_name = metadata['name'], full_years = True)

    for fh in yearly_csvs:
        output_fh = fh.replace('csv', 'pdf')
        year = str(fh[-8:-4])
        wa.Sheets.create_sheet2(metadata['name'], year, 'km3/year', fh, output_fh)
        
    for fh in monthly_csvs:
        output_fh = fh.replace('csv', 'pdf')
        month = str(fh[-6:-4])
        year = str(fh[-11:-7])
        wa.Sheets.create_sheet2(metadata['name'], '{0}-{1}'.format(year, month), 'km3/month', fh, output_fh)
        
    return data

def create_sheet2_csv(lulc_dict, classes_dict, lu_fh, et_fhs, et_dates, t_fhs, t_dates, i_fhs, i_dates, output_dir, catchment_name = None, full_years = True):
    """
    Create sheet 2 csv-files.
    
    Parameters
    ----------
    lulc_dict : dict 
        Describing the different land use classes, import using 'get_dictionaries'.
    classes_dict   : dict   
        Describing the sheet 2 specific aggregation of classes from lulc_dict.
    lu_fh : str        
        Filehandle pointing to a landuse map.
    et_fhs : ndarray       
        Array of filehandles pointing to ET maps.
    et_dates : ndarray     
        Array with datetime.date objects specifying the date of the ET maps.
    t_fhs : ndarray      
        Array of filehandles pointing to T maps.
    t_dates : ndarray        
        Array with datetime.date objects specifying the date of the T maps.
    i_fhs : ndarray
        Array of filehandles pointing to I maps.
    i_dates : ndarray
        Array with datetime.date objects specifying the date of the I maps.    
    output_dir : str  
        Filehandle to folder to save output.
    catchment_name : str, optional
        Name of the catchment, default is None.
    full_years : boolean, optional  
        Choose to also create yearly csv-files, default is True.
    
    Returns
    -------
    csv_fhs : ndarray      
        Array of filehandles pointing to monthly csv-files.
    csv_fhs_yearly : ndarray    
        Array of filehandles pointing to yearly csv-files.
    """
    
    # Check if all maps have the same projection, resolution and No-Data-Value.
    becgis.AssertProjResNDV([lu_fh, et_fhs, t_fhs, i_fhs])
    
    # Calculate the size of each pixel in km2.
    MapArea = becgis.MapPixelAreakm(lu_fh)
    
    # Create some constants.
    month_labels = {1:'01',2:'02',3:'03',4:'04',5:'05',6:'06',7:'07',8:'08',9:'09',10:'10',11:'11',12:'12'}
    first_row = ['LAND_USE', 'CLASS', 'TRANSPIRATION', 'WATER', 'SOIL', 'INTERCEPTION', 'AGRICULTURE', 'ENVIRONMENT', 'ECONOMY', 'ENERGY', 'LEISURE', 'NON_BENEFICIAL']
    
    # Check if output folder exists.
    directory_months = os.path.join(output_dir, "monthly_sheet2")
    if not os.path.exists(directory_months):
        os.makedirs(directory_months)
    
    # Check if the name of the catchment is known.
    if catchment_name is None:
        catchment_name = 'Unknown'
    
    # Check for which dates calculations can be made.
    common_dates = becgis.CommonDates([et_dates, t_dates, i_dates])
    
    # Open the landuse-map.
    LULC = becgis.OpenAsArray(lu_fh)
    
    # Create some variables needed for yearly sheets.
    complete_years = [None]
    year_count = [None]
    if full_years:
        # Check if output folder for yearly csv-files exists.
        directory_years = os.path.join(output_dir, "yearly_sheet2")
        if not os.path.exists(directory_years):
            os.makedirs(directory_years)
        # Check for which years data for 12 months is available.
        yrs, counts = np.unique([date.year for date in common_dates], return_counts = True)
        complete_years = [int(year) for year, count in zip(yrs, counts) if count == 12]
        year_count = 1
    
    # Start calculations.
    for date in common_dates:
        
        # Create csv-file.
        csv_filename = os.path.join(directory_months, '{0}_{1}_{2}.csv'.format(catchment_name, date.year, month_labels[date.month]))
        csv_file = open(csv_filename, 'wb')
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(first_row)
        
        # Open the T, ET and I maps and set NDV pixels to NaN.
        T = becgis.OpenAsArray(t_fhs[t_dates == date][0], nan_values = True)
        ET = becgis.OpenAsArray(et_fhs[et_dates == date][0], nan_values = True)
        I = becgis.OpenAsArray(i_fhs[i_dates == date][0], nan_values = True)
                
        # Convert units from [mm/month] to [km3/month].
        I = I * MapArea / 1000000
        T = T * MapArea / 1000000
        ET = ET * MapArea / 1000000        
        
        # Add monthly values to yearly totals.
        if np.all([full_years, (date.year in complete_years), (year_count is 1)]):
            Tyear = T
            ETyear = ET
            Iyear = I
            year_count += 1
        
        # Add monthly values to yearly totals.
        elif np.all([full_years, (date.year in complete_years), (year_count is not 1)]):
            Tyear += T
            ETyear += ET
            Iyear += I
            year_count += 1
        
        # Calculate evaporation.
        E = ET - T - I
        
        # Write data to csv-file.
        for LAND_USE in classes_dict.keys():
            for CLASS in classes_dict[LAND_USE].keys():
                write_sheet2_row(LAND_USE, CLASS, lulc_dict, classes_dict, LULC, T, I, E, writer)
        
        # Close the csv-file.
        csv_file.close()

        # Start creating a yearly csv-file.
        if np.all([full_years, (date.year in complete_years), (year_count is 13)]):
            # Calcultate evaporation for a whole year.
            Eyear = ETyear - Tyear - Iyear
            
            # Create csv-file for yearly data.
            csv_filename = os.path.join(directory_years, '{0}_{1}.csv'.format(catchment_name, date.year))
            csv_file_year = open(csv_filename, 'wb')
            writer_year = csv.writer(csv_file_year, delimiter=';')
            writer_year.writerow(first_row)
            
            # Write data to yearly csv-file.
            for LAND_USE in classes_dict.keys():
                for CLASS in classes_dict[LAND_USE].keys():
                    write_sheet2_row(LAND_USE, CLASS, lulc_dict, classes_dict, LULC, Tyear, Iyear, Eyear, writer_year)
            
            # Close csv-file.
            csv_file_year.close()
            
            # Set counter back to one.
            year_count = 1

    # Create list of created files.
    csv_fhs = becgis.ListFilesInFolder(directory_months, extension = 'csv')
    
    # Return list of filehandles.
    if full_years:
        csv_fhs_yearly = becgis.ListFilesInFolder(directory_years, extension = 'csv')
        return csv_fhs, csv_fhs_yearly
    else:
        return csv_fhs

def splitET_ITE(et_fhs, et_dates, lai_fhs, lai_dates, p_fhs, p_dates, n_fhs, n_dates, ndm_fhs, ndm_dates, output_dir, ndm_max_original = True, plot_graph = True, save_e = False):
    """
    Split evapotranspiration into transpiration and interception.
    
    Parameters
    ----------
    et_fhs : ndarray
        Array of filehandles pointing to ET maps.
    et_dates : ndarray
        Array with datetime.date objects specifying the date of the ET maps.
    lai_fhs : ndarray
        Array of filehandles pointing to LAI maps.
    lai_dates : ndarray
        Array with datetime.date objects specifying the date of the LAI maps.
    p_fhs : ndarray
        Array of filehandles pointing to P maps.
    p_dates : ndarray
        Array with datetime.date objects specifying the date of the P maps. 
    n_fhs : ndarray
        Array of filehandles pointing to rainy days maps.
    n_dates : ndarray
        Array with datetime.date objects specifying the date of the rainy days maps.
    ndm_fhs : ndarray
        Array of filehandles pointing to NDM maps.
    ndm_dates : ndarray
        Array with datetime.date objects specifying the date of the NDM maps. 
    output_dir : str
        Filehandle specifying the folder to save output.
    plot_graph : boolean, optional
        Plot a graph of the spatially averaged ET and the fractions of E, T and I, default is True.
    
    Returns
    -------
    i_fhs : ndarray
        Array of filehandles pointing to I maps.
    i_dates : ndarray
        Array with datetime.date objects specifying the date of the I maps.
    t_fhs : ndarray
        Array of filehandles pointing to T maps.
    t_dates : ndarray
        Array with datetime.date objects specifying the date of the T maps.
    """
    # Check if all maps have the same projection, resolution and No-Data-Value.
    becgis.AssertProjResNDV([et_fhs, lai_fhs, p_fhs, n_fhs, ndm_fhs])
    
    # Create some constants.
    month_labels = {1:'01',2:'02',3:'03',4:'04',5:'05',6:'06',7:'07',8:'08',9:'09',10:'10',11:'11',12:'12'}
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(et_fhs[0])
    
    # Check for which dates calculations can be made.
    common_dates = becgis.CommonDates([et_dates, lai_dates, p_dates, n_dates, ndm_dates])
    
    if not ndm_max_original:
        ndm_months = np.array([date.month for date in ndm_dates])
        
        ndm_max_folder = os.path.join(output_dir, "ndm_max")
        if not os.path.exists(ndm_max_folder):
                os.makedirs(ndm_max_folder)
                
        ndm_max_fhs = dict()

        footprint = np.ones((50,50), dtype = np.bool)
        
        for month in np.unique(ndm_months):
            std, mean = becgis.CalcMeanStd(ndm_fhs[ndm_months == month], None, None)
            ndm_temporal_max = mean + 2 * std
            ndm_spatial_max = ndimage.maximum_filter(ndm_temporal_max, footprint = footprint)
            output_fh = os.path.join(ndm_max_folder, 'ndm_max_{0}.tif'.format(month_labels[month]))
            becgis.CreateGeoTiff(output_fh, ndm_spatial_max, driver, NDV, xsize, ysize, GeoT, Projection)
            ndm_max_fhs[month] = output_fh
            del std, mean, ndm_spatial_max
    
    if ndm_max_original:
        # Create some variables to calculate the monthly maximum NDM.
        ndm_months = np.array([date.month for date in ndm_dates])
        unique_ndm_months, counts = np.unique(ndm_months, return_counts = True)
        NDMmax = dict()
        
        # Calculate the maximum average NDM value for each month.
        for month in unique_ndm_months:
            ndm_monthly_mean = np.zeros((ysize,xsize))
            for date in ndm_dates[ndm_months == month]:
                data = becgis.OpenAsArray(ndm_fhs[ndm_dates == date][0], nan_values = True)
                ndm_monthly_mean[:,:] += data
            ndm_monthly_mean[:,:] /= counts[unique_ndm_months == month]
            NDMmax[month] = np.nanmax(ndm_monthly_mean)
    
    # Create some variables needed to plot graphs.
    if plot_graph:
        et = np.array([])
        i = np.array([])
        t = np.array([])
        e = np.array([])
    
    # Start iterating over dates.
    for date in common_dates:
        # Open data to calculate I and set NDV pixels to NaN.
        LAI = becgis.OpenAsArray(lai_fhs[lai_dates == date][0], nan_values = True)
        P = becgis.OpenAsArray(p_fhs[p_dates == date][0], nan_values = True)
        n = becgis.OpenAsArray(n_fhs[n_dates == date][0], nan_values = True)
        
        # Calculate I.
        I = LAI * (1 - (1 + (P/n) * (1 - np.exp(-0.5 * LAI)) * (1/LAI))**-1) * n
        
        # Set boundary conditions.
        I[LAI == 0] = 0.
        I[n == 0] = 0.
        
        # Open ET and NDM maps and set NDV pixels to NaN.
        ET = becgis.OpenAsArray(et_fhs[et_dates == date][0], nan_values = True)
        NDM = becgis.OpenAsArray(ndm_fhs[ndm_dates == date][0], nan_values = True)
        
        if ndm_max_original:
            NDMMAX = 0.95 / NDMmax[date.month]
        
        if not ndm_max_original:
            NDMMAX = 1.00 / becgis.OpenAsArray(ndm_max_fhs[date.month], nan_values = True)
    
        # Calculate T.
        T = np.minimum((NDM * NDMMAX),np.ones(np.shape(NDM)) * 0.95) * (ET - I)
            
        # Create folder to store maps.
        directory_t = os.path.join(output_dir, "t")
        if not os.path.exists(directory_t):
            os.makedirs(directory_t)
        
        directory_i = os.path.join(output_dir, "i")
        if not os.path.exists(directory_i):
            os.makedirs(directory_i)

        if save_e:
            directory_e = os.path.join(output_dir, "e")
            if not os.path.exists(directory_e):
                os.makedirs(directory_e)
            E = ET - I - T
            output_fh = os.path.join(directory_e, 'E_{0}{1}.tif'.format(date.year,month_labels[date.month]))
            becgis.CreateGeoTiff(output_fh, E, driver, NDV, xsize, ysize, GeoT, Projection)
        
        # Store values to plot a graph.
        if plot_graph:
            et = np.append(et, np.nanmean(ET))
            i = np.append(i, np.nanmean(I))
            t = np.append(t, np.nanmean(T))
            e = np.append(e, np.nanmean(ET - I - T))
        
        # Save I map.
        output_fh = os.path.join(directory_i, 'I_{0}{1}.tif'.format(date.year,month_labels[date.month]))
        becgis.CreateGeoTiff(output_fh, I, driver, NDV, xsize, ysize, GeoT, Projection)

        # Save T map.
        output_fh = os.path.join(directory_t, 'T_{0}{1}.tif'.format(date.year,month_labels[date.month]))
        becgis.CreateGeoTiff(output_fh, T, driver, NDV, xsize, ysize, GeoT, Projection)
        
        print("Finished E,T,I for {0}".format(date))
    
    # Plot graph of ET and E, T and I fractions.
    if plot_graph:
        fig = plt.figure(figsize = (10,10))
        plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
        ax = fig.add_subplot(111)
        ax.plot(common_dates, et, color = 'k')
        ax.patch.set_visible(False)
        ax.set_title('Average ET and E, T and I fractions')
        ax.set_ylabel('ET [mm/month]')
        ax.patch.set_visible(True)
        ax.fill_between(common_dates, et, color = '#a3db76', label = 'Evaporation')
        ax.fill_between(common_dates, i + t , color = '#6bb8cc', label = 'Transpiration')
        ax.fill_between(common_dates, i , color = '#497e7c', label = 'Interception')
        ax.scatter(common_dates, et, color = 'k')
        ax.legend(loc = 'upper left',fancybox=True, shadow=True)
        fig.autofmt_xdate()
        ax.set_xlim([common_dates[0], common_dates[-1]])
        ax.set_ylim([0, max(et) *1.2])
        ax.set_xlabel('Time')
        [r.set_zorder(10) for r in ax.spines.itervalues()]
        plt.savefig(os.path.join(output_dir,'ETfractions_ITE.png'))

    # Create arrays with filehandles and datetime.date objects on the created maps.
    t_fhs, t_dates, t_years, t_months, t_days = becgis.SortFiles(directory_t, [-10,-6], month_position = [-6,-4])
    i_fhs, i_dates, i_years, i_months, i_days = becgis.SortFiles(directory_i, [-10,-6], month_position = [-6,-4])
    
    return t_fhs, t_dates, i_fhs, i_dates

def calc_footprint(max_radius, pixel_size):
    """
    Calculate a footprint to select local maxima.
    
    Parameters
    ----------
    max_radius : float
        Radius of circle around each pixel in which maximum should be found.
    pixel_size : float
        Size of one pixel.
    
    Return
    ------
    footprint : ndarray
        Boolean array with circle of max_radius.
    
    """
    pixels = int(max_radius / pixel_size) + 1
    X, Y = np.meshgrid(range(pixels), range(pixels))
    radius = np.sqrt(X**2 + Y**2)
    half_circle = np.concatenate((np.flipud(radius), radius[1:,:]))
    half_circle_2 = np.fliplr(np.concatenate((np.flipud(radius), radius[1:,:])))[:,:-1]
    test = np.hstack((half_circle_2, half_circle))
    footprint = test <= pixels
    return footprint
    
def write_sheet2_row(LAND_USE, CLASS, lulc_dict, classes_dict, lulc, T, I, E, writer):
    """
    Write a row with spatial aggregates to a sheet2 csv-file.
    
    Parameters
    ----------
    LAND_USE : str
        The landuse category of the row.
    CLASS : str
        The class of the landuse category of the row.
    lulc_dict : dict 
        Describing the different land use classes, import using 'get_dictionaries'.
    classes_dict   : dict   
        Describing the sheet 2 specific aggregation of classes from lulc_dict.
    T : ndarray
        The spatial transpiration data.
    I : ndarray
        The spatial interception data.        
    E : ndarray
        The spatial evaporation data.
    writer : object
        csv.writer object.
    """
    # Get a list of the different landuse classes to be aggregated.
    lulcs = classes_dict[LAND_USE][CLASS]
    
    # Create a mask to ignore non relevant pixels.
    mask=np.logical_or.reduce([lulc == value for value in lulcs])
    
    # Calculate the spatial sum of the different parameters.
    transpiration = np.nansum(T[mask])
    interception = np.nansum(I[mask])
    evaporation = np.nansum(E[mask])
    
    # Set special cases.
    if np.any([CLASS == 'Natural water bodies', CLASS == 'Managed water bodies']):
        soil_evaporation = 0
        water_evaporation = evaporation
    else:
        soil_evaporation = evaporation
        water_evaporation = 0            
        
    # Create some necessary variables.
    agriculture = 0
    environment = 0
    economy = 0
    energy = 0
    leisure = 0
    non_beneficial = 0
    
    # Calculate several landuse type specific variables.
    for lu_type in lulcs:
        
        # Get some constants for the landuse type.
        beneficial_percentages = np.array(lulc_dict[lu_type][3:6]) / 100
        service_contributions = np.array(lulc_dict[lu_type][6:11]) / 100         
          
        # Calculate the beneficial ET.
        benef_et = np.nansum([np.nansum(T[lulc == lu_type]) * beneficial_percentages[0],
               np.nansum(E[lulc == lu_type]) * beneficial_percentages[1],
               np.nansum(I[lulc == lu_type]) * beneficial_percentages[2]])
               
        # Determine the service contributions.
        agriculture += benef_et * service_contributions[0] 
        environment += benef_et * service_contributions[1] 
        economy += benef_et * service_contributions[2]
        energy += benef_et * service_contributions[3]
        leisure += benef_et * service_contributions[4]
       
        # Determine non-beneficial ET.
        non_beneficial += (np.nansum([np.nansum(T[lulc == lu_type]) * (1 - beneficial_percentages[0]),
               np.nansum(E[lulc == lu_type]) * (1 - beneficial_percentages[1]),
               np.nansum(I[lulc == lu_type]) * (1 - beneficial_percentages[2])]))
    
    # Create the row to be written
    row = [LAND_USE, CLASS, "{0:.2f}".format(np.nansum([0, transpiration])), "{0:.2f}".format(np.nansum([0, water_evaporation])), "{0:.2f}".format(np.nansum([0, soil_evaporation])), "{0:.2f}".format(np.nansum([0, interception])), "{0:.2f}".format(np.nansum([0, agriculture])), "{0:.2f}".format(np.nansum([0, environment])), "{0:.2f}".format(np.nansum([0, economy])), "{0:.2f}".format(np.nansum([0, energy])), "{0:.2f}".format(np.nansum([0, leisure])), "{0:.2f}".format(np.nansum([0, non_beneficial]))]
    
    # Write the row.
    writer.writerow(row)