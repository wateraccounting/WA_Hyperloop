# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:05:42 2017

@author: bec
"""
import os
import indicators
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import shapefile
import becgis as bg
from WA_Hyperloop import get_dictionaries as gd


def clean_name(string):
    """
    Replace underscores with spaces and make every first letter
    a capital letter, except for the ET abbreviation.
    """
    return string.replace('_', ' ').title().replace(' Et ', ' ET ')


def get_frequency(date_list):
    """
    Check a list of datetime.date objects and estimate if the frequency
    of the dates is yearly or monthly.
    """
    if len(np.unique([d.month for d in date_list])) == 1:
        frequency = 'yearly'
    elif len(np.unique([d.month for d in date_list])) == 12:
        frequency = 'monthly'
    else:
        frequency = ''
    return frequency


def get_definitions(freq):
    """
    Define some definition strings as described in P. Karimi et al. (2013).
    """
    eq_5 = r'$^{Exploitable\;Water_{%s}}/_{Net\;Inflow_{%s}}$' % (freq, freq)
    eq_6 = (r'$^{(\Delta S_{fw})_{%s}}/'
            r'_{Exploitable\;Water_{%s}}$') % (freq, freq)
    eq_7 = (r'$^{Available\;Water_{%s}}/'
            r'_{Exploitable\;Inflow_{%s}}$') % (freq, freq)
    eq_8 = r'$^{Utilised\;Flow_{%s}}/_{Available\;Water_{%s}}$' % (freq, freq)
    eq_9 = (r'$^{Reserved\;Outflows_{%s}}/'
            r'_{(Q_{out}^{SW} + Q_{out}^{GW})_{%s}}$') % (freq, freq)

    eq_10 = r'$^{T_{%s}}/_{ET_{%s}}$' % (freq, freq)
    eq_11 = (r'$^{(E_{beneficial} + '
             r'T_{beneficial})_{%s}}/_{ET_{%s}}$') % (freq, freq)
    eq_12 = r'$^{(ET_{managed})_{%s}}/_{ET_{%s}}$' % (freq, freq)
    eq_13 = r'$^{(ET_{agricultural})_{%s}}/_{ET_{%s}}$' % (freq, freq)
    eq_14 = r'$^{(ET_{irrigated})_{%s}}/_{ET_{%s}}$' % (freq, freq)

    eq_20 = (r'$^{(Q_{w}^{GW})_{%s}}/_{(Q_{w}^{SW}'
             r' + Q_{w}^{GW})_{%s}}$') % (freq, freq)
    eq_21 = r'$^{(ET_{Q})_{%s}}/_{(Q_{w})_{%s}}$' % (freq, freq)
    eq_22 = r'$^{(Q_{R})_{%s}}/_{(Q_{w})_{%s}}$' % (freq, freq)

    definitions = {'expl._wat.':    eq_5,
                   'strg_chng.':    eq_6,
                   'avlb._wat.':    eq_7,
                   'bsn._clsr.':    eq_8,
                   'rsrvd._of.':    eq_9,
                   't_fraction':    eq_10,
                   'benefi_ET':     eq_11,
                   'mngd_ET':       eq_12,
                   'agr_ET':        eq_13,
                   'irr_agr_ET':    eq_14,
                   'gw_wthdrwl':    eq_20,
                   'irr._fcncy':    eq_21,
                   'recovarble':    eq_22}

    return definitions


def get_def_longname(short_name):
    """
    """
    long_definitions = {'expl._wat.':    'Exploitable Water',
                        'strg_chng.':    'Storage Change',
                        'avlb._wat.':    'Available Water',
                        'bsn._clsr.':    'Basin Closure',
                        'rsrvd._of.':    'Reserved Outflow',
                        't_fraction':    'Transpiration',
                        'benefi_ET':     'Beneficial ET',
                        'mngd_ET':       'Managed ET',
                        'agr_ET':        'Agricultural ET',
                        'irr_agr_ET':    'Irrigated Agricultural ET',
                        'gw_wthdrwl':    'Groundwater Withdrawal',
                        'irr._fcncy':    'Irrigation Efficiency',
                        'recovarble':    'Recoverable'}

    return long_definitions[short_name]


def plot_indicator(path, *args):
    """
    Plot histograms for different Water Accounting indicators and
    save the plots in the path folder.
    """
    for arg in args:

        freq = get_frequency(arg['dates'])
        defs = get_definitions(freq)
        dates = arg.pop('dates', None)

        for indicator, values in arg.items():

            values = values[~np.isnan(values)]

            out_file = 'idc_{0}_{1}.png'.format(indicator, freq)
            out_path = os.path.join(path, out_file)

            stats = len(values), np.mean(values), np.std(values)
            title = defs.get(indicator)

            plt.figure(1)
            plt.clf()
            plt.grid(b=True, which='Major', color='0.65', linestyle='--')
            plt.hist(values)
            if indicator == 'strg_chng.':
                plt.xlim([-1, 1])
            else:
                plt.xlim([0, 1])
            plt.xlabel(get_def_longname(indicator) + ' [-]')
            plt.ylabel('Frequency [-]')
            plt.suptitle(('n = {0}, mean = {1:.2f}'
                          ', std = {2:.2f}'.format(*stats)))
            plt.title(title, fontsize=18)
            plt.subplots_adjust(top=0.85)
            plt.savefig(out_path)
            plt.close(1)

        arg['dates'] = dates

#%%
plt.figure()

## the data
idc_means = dict()
idc_stds = dict()

for ID, basin in basins.items():

    idc_means[ID] = list()
    idc_stds[ID] = list()

    for indicator, values in basin['idcs'].items():

        if indicator != 'dates':
            idc_means[ID].append(np.nanmean(values))
            idc_stds[ID].append(np.nanstd(values))

no_of_indicators = len(basin1)
no_of_basins = len(basins.keys())

## necessary variables
ind = np.arange(N)                # the x locations for the groups
width = 0.8 / no_basins                   # the width of the bars

#colors = ['#6bb8cc','#87c5ad', '#9ad28d', '#acd27a', '#c3b683', '#d4988b', '#b98b89', '#868583', '#497e7c'] * 3
import matplotlib.colors as colors
import matplotlib.cm as cmx
jet = plt.get_cmap('tab20')
cNorm  = colors.Normalize(vmin=0, vmax=14)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

values = range(15)
rects = tuple()
i = 0
## the bars
for ID in basins.keys():
    rects1 = plt.bar(ind + i*width, idc_means[ID], width,
                    color= scalarMap.to_rgba(values[i]),
                    #yerr=idc_stds[ID],
                    error_kw=dict(elinewidth=2,ecolor='red'))
    
    i+= 1
    rects += (rects1, )

xticks = basin['idcs'].keys()
xticks.pop(xticks.index('dates'))
xticks = [get_def_longname(short_name) for short_name in xticks]
#rects2 = ax.bar(ind+width, basin2, width,
#                    color='red',
#                    yerr=basin2_std,
#                    error_kw=dict(elinewidth=2,ecolor='black'))


# axes and labels
plt.xlim(-width,len(ind)+width)
plt.ylim(0,1)
plt.ylabel('Scores')
#ax.set_title()
#xTickMarks = xticks
plt.xticks(ind+width)
ax = plt.gca()
xtickNames = ax.set_xticklabels(xticks)
plt.setp(xtickNames, rotation=45, fontsize=10)

## add a legend
names = [basin['name'] for basin in basins.values()]
ax.legend( rects, names)

plt.show()

#%%

def calc_mean(maps):
    """
    Calculate the mean and the standard deviation per pixel for a
    serie of maps.
    """
    fh = os.path.join(tempfile.mkdtemp(), 'temp.tif')
    geoinfo = bg.GetGeoInfo(maps[0])

    data_sum = np.zeros((geoinfo[3], geoinfo[2]))
    data_count = np.zeros((geoinfo[3], geoinfo[2]))

    for filename in maps:
        data = bg.OpenAsArray(filename, nan_values=True)
        data_sum = np.nansum([data_sum, data], axis=0)

        count = np.ones((geoinfo[3], geoinfo[2]))
        count[np.isnan(data)] = 0
        data_count += count

    mean = data_sum / data_count

    bg.CreateGeoTiff(fh, mean, *geoinfo)

    return fh


def calc_indicators(basins, output_dir):
    """
    Calculate indicators based on numbers on the Water Accounting sheets,
    plot histograms of them and fill in average values into a shapefile
    that contains all the basin outlines.
    """

    for basin in basins.values():

        print "Running Basin {0}".format(basin['id'])

        basin['idcs'] = dict()
        basin['stats'] = dict()

        dir1 = os.path.join(output_dir, basin['name'], 'csvs_monthly')
        sh1_indicators = indicators.sheet1_indicators(dir1)
        basin['idcs'] = merge_two_dicts(basin['idcs'], sh1_indicators)

        dir2 = os.path.join(output_dir, basin['name'], 'monthly_sheet2')
        sh2_indicators = indicators.sheet2_indicators(dir2)
        basin['idcs'] = merge_two_dicts(basin['idcs'], sh2_indicators)

        dir4 = os.path.join(output_dir, basin['name'], 'sheet4')
        sh4_indicators = indicators.sheet4_indicators(dir4)
        basin['idcs'] = merge_two_dicts(basin['idcs'], sh4_indicators)

        lu_areas = calc_lu_areas(basin['lu'])
        basin['stats'] = merge_two_dicts(basin['stats'], lu_areas)

        prcp_monthly = calc_monthly_p(basin, output_dir)
        basin['stats'] = merge_two_dicts(basin['stats'], prcp_monthly)

        eti = calc_avg_eti(basin['lu'], basin, output_dir)
        basin['stats'] = merge_two_dicts(basin['stats'], eti)

    return basins


def plot_indicators(basins, output_dir):
    """
    """
    basin_shp = os.path.join(output_dir, 'All_Basins.shp')

    for basin in basins.values():

        path = os.path.join(output_dir, basin['name'])
        plot_indicator(path, basin['idcs'])
        update_idc_shapefile(basin_shp, ('ID', basin['id']), basin['idcs'],
                             basin['stats'])


def calc_sb_indicators(basins, output_dir, pop_map):
    """
    """
    sb_shp = os.path.join(output_dir, 'All_Basins_Subbasins.shp')

    for basin in basins.values():

        print "Runnning Basin {0}".format(basin['id'])

        sb_masks = bg.ListFilesInFolder(basin['masks'])

        for sb_map in sb_masks:

            IDsb = str(basin['id']) + os.path.split(sb_map)[1].split('_')[0]

            pop = calc_mskd_mean(sb_map, pop_map, 'ppl/ha')
            et = calc_avg_flux(sb_map, basin, output_dir, 'et')
            precip = calc_avg_flux(sb_map, basin, output_dir, 'p')

            args = pop, et, precip
            update_idc_shapefile(sb_shp, ('IDsb', IDsb), *args)


def calc_avg_flux(mask, basin, output_dir, flux):
    """
    Calculate the temporally and spatially average of a timeseries
    of maps.
    """
    input_dir = os.path.join(output_dir, basin['name'], flux)
    ets = bg.SortFiles(input_dir, [-10, -6], month_position=[-6, -4])[0]
    mean_map = calc_mean(ets)
    et = calc_mskd_mean(mask, mean_map, flux)
    os.remove(mean_map)
    return et


def merge_two_dicts(x, y):
    """
    Given two dicts, merge them into a new dict as a shallow copy.
    """
    z = x.copy()
    z.update(y)
    return z


def calc_avg_eti(mask, basin, output_dir):
    """
    Calculate the average E, T and I values in a basin.
    """
    eti_idc = dict()
    for flux in ['i', 't', 'et']:
        flux_idc = calc_avg_flux(mask, basin, output_dir, flux)
        eti_idc = merge_two_dicts(eti_idc, flux_idc)
    eti_idc['e'] = eti_idc['et'] - eti_idc['t'] - eti_idc['i']
    return eti_idc


def calc_monthly_p(basin, output_dir):
    """
    Calculate the monthly average precipitation in a basin.
    """
    input_dir = os.path.join(output_dir, basin['name'], 'p')
    precip = bg.SortFiles(input_dir, [-10, -6], month_position=[-6, -4])

    prcp_idc = dict()

    for month in np.unique(precip[3]):

        p_mean = calc_mean(precip[0][precip[3] == month])
        prcp = calc_mskd_mean(basin['lu'], p_mean, 'P_{0}'.format(month))
        os.remove(p_mean)

        prcp_idc = merge_two_dicts(prcp_idc, prcp)

    return prcp_idc


def get_shp_field_names(shp_object):
    """
    Return a list of the field names in a shapefile.
    """
    return [field[0] for field in shp_object.fields[1:]]


def update_idc_shapefile(basin_shp, identifier, *args):
    """
    Update a shapefile by providing a colum name and value and
    an indentifier to indicate which feature should be updated.
    """
    orig = shapefile.Reader(basin_shp)
    new = shapefile.Writer()
    new.fields = list(orig.fields)
    fld_names = get_shp_field_names(orig)
    rrecords = orig.records()

    for arg in args:
        dates = arg.pop('dates', None)
        for indicator, values in arg.items():

            if len(indicator) >= 10:
                indicator = indicator[0:10]

            value = np.nanmean(values)

            if indicator not in fld_names:
                new.field(indicator, "F", 13, 3)
                fld_names = get_shp_field_names(new)
                for rec in rrecords:
                    fld_name_idx = fld_names.index(identifier[0])
                    if int(rec[fld_name_idx]) == int(identifier[1]):
                        rec.append(value)
                    else:
                        rec.append(None)

            else:
                index = fld_names.index(indicator)
                fld_name_idx = fld_names.index(identifier[0])
                for rec in rrecords:
                    if int(rec[fld_name_idx]) == int(identifier[1]):
                        rec[index] = value

        arg['dates'] = dates

    new.records = rrecords
    new.shapes().extend(orig.shapes())
    new.save(basin_shp)

# import VN_metadata
#
# basins = VN_metadata.define_VNbasin_metadata()
# pop_map = r"D:\Products\WorldPop\VNM-POP\VNM_pph_v2b_2009.tif"
# output_dir = r"D:\project_ADB\Catchments\Vietnam"
#
# basins = calc_indicators(basins, output_dir)
# calc_sb_indicators(basins, output_dir, pop_map)
# plot_indicators(basins, idc_values, output_dir)


def calc_mskd_mean(mask_map, pop_map, idc_name):
    """
    Open a map and calculate the average of a masked area.
    """
    target_maps = np.array([pop_map])
    temp_dir = os.path.split(mask_map)[0]

    pop_map = bg.MatchProjResNDV(mask_map, target_maps, temp_dir)[0]

    lu = bg.OpenAsArray(mask_map, nan_values=True)
    ppl_ha = bg.OpenAsArray(pop_map, nan_values=True)

    ppl_ha[np.isnan(lu)] = np.nan

    ppl = np.nanmean(ppl_ha)

    os.remove(pop_map)

    pop = {idc_name: ppl}

    return pop


def calc_lu_areas(lu_map):
    """
    Calculate the areas of the four different major WA+ landuse categories.
    """
    lu = bg.OpenAsArray(lu_map, nan_values=True)

    lu_types = gd.get_sheet1_classes()

    area_km2 = bg.MapPixelAreakm(lu_map)

    areas = dict()

    for typ, classes in lu_types.items():

        mask = np.logical_or.reduce([lu == value for value in classes])

        area = np.nansum(area_km2[mask])

        areas[typ] = area

    return areas
