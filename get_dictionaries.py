# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 13:36:07 2016

@author: Bert Coerver, b.coerver@unesco-ihe.org
"""

def get_sheet7_classes():   
    live_feed = {'Pasture':.5,
                 'Crop':.25}
# LULC class to Pasture or Crop
    feed_dict = {'Pasture':[2,3,12,13,14,15,16,17,20,29,34],
                 'Crop':[35,36,37,38,39,40,41,42,43,44,45,54,55,56,57,58,59,60,61,62]}
    
    # above ground biomass for tree and bush classes
    abv_grnd_biomass_ratio = {'1':.26, # protected forest
                              '8':.24, # closed deciduous forest
                              '9':.43, # open deciduous forest
                              '10':.23, # closed evergreen forest
                              '11':.46, # open evergreen forest
                              '12':.48, # closed savanna
                              '13':.48} # open savanna
    # C content as fraction dry matter for Carbon sequestration calculations
    c_fraction = {'default':.47
            }
    
    # 5% of above ground biomass for now
    fuel_dict ={'all':.05}
    # dict: class - lulc
    sheet7_lulc = {
    'PROTECTED':    {'Forest': [1],
                    'Shrubland': [2],
                    'Natural grasslands':[3],
                    'Natural water bodies':[4],
                    'Wetlands':[5],
                    'Glaciers':[6],
                    'Others':[7]
                    },
    'UTILIZED':     {'Forest':[8,9,10,11],
                    'Shrubland':[14],
                    'Natural grasslands':[12,13,15,16,20],
                    'Natural water bodies':[23,24,],
                    'Wetlands':[17,19,25,30,31],
                    'Others':[18,21,22,26,27,28,29,32]
                    },
    'MODIFIED':     {'Rainfed crops': [34,35,36,37,38,39,40,41,42,43,44],
                    'Forest plantations':[33],
                    'Settlements':[47,48,49,50,51],
                    'Others':[45,46]
                    },
    'MANAGED': {'Irrigated crops':[52,53,54,55,56,57,58,59,60,61,62],
                'Managed water bodies':[63,65,74],
                'Residential':[68,69,71,72],
                'Industry':[67,70,76],
                'Others':[75,78,77],
                'Indoor domestic':[66],
                'Indoor industry':[0],
                'Greenhouses':[64],
                'Livestock and husbandry':[73],
                'Power and energy':[79,80],
                }

    }

    sheet7_lulc_classes =dict()
    for k in sheet7_lulc.keys():
        l = []
        for k2 in sheet7_lulc[k].keys():
            l.append(sheet7_lulc[k][k2])
        sheet7_lulc_classes[k]=[item for sublist in l for item in sublist]
    return live_feed, feed_dict, abv_grnd_biomass_ratio,fuel_dict,sheet7_lulc_classes,c_fraction



def get_lulc_cmap():
    
    from matplotlib.colors import LinearSegmentedColormap
    
    cm_dict = {
    1: (0.0, 0.15686274509803921, 0.0), 
    2: (0.74509803921568629, 0.70588235294117652, 0.23529411764705882), 
    3: (0.69019607843137254, 1.0, 0.12941176470588237), 
    4: (0.32549019607843138, 0.55686274509803924, 0.83529411764705885), 
    5: (0.15686274509803921, 0.98039215686274506, 0.70588235294117652), 
    6: (1.0, 1.0, 1.0), 
    7: (0.85882352941176465, 0.83921568627450982, 0.0), 
    8: (0.0, 0.27450980392156865, 0.0), 
    9: (0.0, 0.48627450980392156, 0.0), 
    10: (0.0, 0.39215686274509803, 0.0), 
    11: (0.0, 0.5490196078431373, 0.0), 
    12: (0.60784313725490191, 0.58823529411764708, 0.19607843137254902), 
    13: (1.0, 0.74509803921568629, 0.35294117647058826), 
    14: (0.47058823529411764, 0.58823529411764708, 0.11764705882352941), 
    15: (0.35294117647058826, 0.45098039215686275, 0.098039215686274508), 
    16: (0.5490196078431373, 0.74509803921568629, 0.39215686274509803), 
    17: (0.11764705882352941, 0.74509803921568629, 0.66666666666666663), 
    18: (0.96078431372549022, 1.0, 0.90196078431372551), 
    19: (0.78431372549019607, 0.90196078431372551, 1.0), 
    20: (0.33725490196078434, 0.52549019607843139, 0.0), 
    21: (1.0, 0.82352941176470584, 0.43137254901960786), 
    22: (0.90196078431372551, 0.90196078431372551, 0.90196078431372551), 
    23: (0.0, 0.39215686274509803, 0.94117647058823528), 
    24: (0.0, 0.21568627450980393, 0.60392156862745094), 
    25: (0.6470588235294118, 0.90196078431372551, 0.39215686274509803), 
    26: (0.82352941176470584, 0.90196078431372551, 0.82352941176470584), 
    27: (0.94117647058823528, 0.6470588235294118, 0.078431372549019607), 
    28: (0.90196078431372551, 0.86274509803921573, 0.82352941176470584), 
    29: (0.74509803921568629, 0.62745098039215685, 0.5490196078431373), 
    30: (0.12941176470588237, 0.75686274509803919, 0.51764705882352946), 
    31: (0.10980392156862745, 0.64313725490196083, 0.4392156862745098), 
    32: (0.39215686274509803, 1.0, 0.58823529411764708), 
    33: (0.96078431372549022, 0.98039215686274506, 0.76078431372549016), 
    34: (0.92941176470588238, 0.96470588235294119, 0.59607843137254901), 
    35: (0.88627450980392153, 0.94117647058823528, 0.35294117647058826), 
    36: (0.81960784313725488, 0.89803921568627454, 0.082352941176470587), 
    37: (0.71372549019607845, 0.7803921568627451, 0.074509803921568626), 
    38: (0.59215686274509804, 0.6470588235294118, 0.058823529411764705), 
    39: (0.51764705882352946, 0.56470588235294117, 0.054901960784313725), 
    40: (0.4392156862745098, 0.47843137254901963, 0.047058823529411764), 
    41: (0.36078431372549019, 0.396078431372549, 0.043137254901960784), 
    42: (0.27843137254901962, 0.31372549019607843, 0.031372549019607843), 
    43: (0.20000000000000001, 0.22352941176470589, 0.019607843137254902), 
    44: (0.31372549019607843, 0.74509803921568629, 0.15686274509803921), 
    45: (0.70588235294117652, 0.62745098039215685, 0.70588235294117652), 
    46: (0.56862745098039214, 0.50980392156862742, 0.45098039215686275), 
    47: (0.47058823529411764, 0.019607843137254902, 0.098039215686274508), 
    48: (0.82352941176470584, 0.039215686274509803, 0.15686274509803921), 
    49: (1.0, 0.50980392156862742, 0.17647058823529413), 
    50: (0.98039215686274506, 0.396078431372549, 0.0), 
    51: (1.0, 0.58823529411764708, 0.58823529411764708), 
    52: (0.70196078431372544, 0.95294117647058818, 0.94509803921568625), 
    53: (0.61960784313725492, 0.94117647058823528, 0.93333333333333335), 
    54: (0.44313725490196076, 0.9137254901960784, 0.90196078431372551), 
    55: (0.32156862745098042, 0.89411764705882357, 0.88235294117647056),
     56: (0.20784313725490197, 0.87450980392156863, 0.85882352941176465), 
    57: (0.12941176470588237, 0.80392156862745101, 0.78823529411764703), 
    58: (0.11372549019607843, 0.70196078431372544, 0.68627450980392157), 
    59: (0.098039215686274508, 0.59215686274509804, 0.58039215686274515), 
    60: (0.082352941176470587, 0.49019607843137253, 0.4823529411764706), 
    61: (0.066666666666666666, 0.396078431372549, 0.38823529411764707), 
    62: (0.050980392156862744, 0.29411764705882354, 0.29019607843137257), 
    63: (0.0, 0.15686274509803921, 0.4392156862745098), 
    64: (1.0, 0.80000000000000004, 1.0), 
    65: (0.18431372549019609, 0.47450980392156861, 1.0), 
    66: (1.0, 0.23529411764705882, 0.039215686274509803), 
    67: (0.70588235294117652, 0.70588235294117652, 0.70588235294117652), 
    68: (1.0, 0.54509803921568623, 1.0), 69: (1.0, 0.29411764705882354, 1.0), 
    70: (0.5490196078431373, 0.5490196078431373, 0.5490196078431373), 
    71: (0.58823529411764708, 0.0, 0.80392156862745101), 
    72: (0.47058823529411764, 0.47058823529411764, 0.47058823529411764), 
    73: (0.70588235294117652, 0.50980392156862742, 0.50980392156862742), 
    74: (0.11764705882352941, 0.50980392156862742, 0.45098039215686275), 
    75: (0.078431372549019607, 0.58823529411764708, 0.50980392156862742), 
    76: (0.39215686274509803, 0.39215686274509803, 0.39215686274509803), 
    77: (0.11764705882352941, 0.35294117647058826, 0.50980392156862742), 
    78: (0.23529411764705882, 0.23529411764705882, 0.23529411764705882), 
    79: (0.15686274509803921, 0.15686274509803921, 0.15686274509803921), 
    80: (0.0, 0.0, 0.0)}

    cmap = LinearSegmentedColormap.from_list('WA_LULC', cm_dict.values(), N = 80)
    
    return cmap
    
    
def get_lulcs(lulc_version = '4.0'):
    
    lulc = dict()
        
    lulc['4.0'] = {
    'legend': ['Code', 'Landuse', 'Description', 'Beneficial T [%]', 'Beneficial E [%]', 'Beneficial I [%]', 'Agriculture [%]', 'Environment [%]', 'Economic [%]', 'Energy [%]', 'Leisure [%]'],
    0: ['X','X','X',0.,0.,0.,0.,0.,0.,0.,0.],
    1: ['PLU1', 'Protected', 'Protected forests', 100.0, 100.0, 0.0, 0.0, 85.0, 0.0, 0.0, 15.0], 
    2: ['PLU2', 'Protected', 'Protected shrubland', 100.0, 100.0, 0.0, 0.0, 85.0, 0.0, 0.0, 15.0], 
    3: ['PLU3', 'Protected', 'Protected natural grasslands', 100.0, 100.0, 0.0, 0.0, 85.0, 0.0, 0.0, 15.0], 
    4: ['PLU4', 'Protected', 'Protected natural waterbodies', 100.0, 100.0, 0.0, 0.0, 85.0, 0.0, 0.0, 15.0], 
    5: ['PLU5', 'Protected', 'Protected wetlands', 100.0, 100.0, 0.0, 0.0, 85.0, 0.0, 0.0, 15.0], 
    6: ['PLU6', 'Protected', 'Glaciers', 0.0, 100.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0], 
    7: ['PLU7', 'Protected', 'Protected other', 100.0, 100.0, 0.0, 0.0, 85.0, 0.0, 0.0, 15.0], 
    8: ['ULU1', 'Utilized', 'Closed deciduous forest', 100.0, 0.0, 0.0, 5.0, 90.0, 0.0, 0.0, 5.0], 
    9: ['ULU2', 'Utilized', 'Open deciduous forest', 100.0, 0.0, 0.0, 5.0, 90.0, 0.0, 0.0, 5.0], 
    10: ['ULU3', 'Utilized', 'Closed evergreen forest', 100.0, 0.0, 0.0, 5.0, 90.0, 0.0, 0.0, 5.0], 
    11: ['ULU4', 'Utilized', 'Open evergreen forest', 100.0, 0.0, 0.0, 5.0, 90.0, 0.0, 0.0, 5.0], 
    12: ['ULU5', 'Utilized', 'Closed savanna', 100.0, 0.0, 0.0, 5.0, 80.0, 0.0, 10.0, 5.0], 
    13: ['ULU6', 'Utilized', 'Open savanna', 100.0, 0.0, 0.0, 10.0, 80.0, 0.0, 5.0, 5.0], 
    14: ['ULU7', 'Utilized', 'Shrub land & mesquite', 100.0, 0.0, 0.0, 5.0, 85.0, 0.0, 10.0, 0.0], 
    15: ['ULU8', 'Utilized', ' Herbaceous cover', 100.0, 0.0, 0.0, 5.0, 95.0, 0.0, 0.0, 0.0], 
    16: ['ULU9', 'Utilized', 'Meadows & open grassland', 100.0, 0.0, 0.0, 60.0, 30.0, 0.0, 0.0, 10.0], 
    17: ['ULU10', 'Utilized', 'Riparian corridor', 100.0, 0.0, 0.0, 10.0, 60.0, 10.0, 0.0, 20.0], 
    18: ['ULU11', 'Utilized', 'Deserts', 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0], 
    19: ['ULU12', 'Utilized', 'Wadis', 100.0, 0.0, 0.0, 15.0, 80.0, 0.0, 0.0, 5.0], 
    20: ['ULU13', 'Utilized', 'Natural alpine pastures', 100.0, 0.0, 0.0, 70.0, 20.0, 0.0, 0.0, 10.0], 
    21: ['ULU14', 'Utilized', 'Rocks & gravel & stones & boulders', 100.0, 0.0, 0.0, 0.0, 95.0, 0.0, 0.0, 5.0], 
    22: ['ULU15', 'Utilized', 'Permafrosts', 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0], 
    23: ['ULU16', 'Utilized', 'Brooks & rivers & waterfalls', 0.0, 50.0, 0.0, 25.0, 55.0, 5.0, 0.0, 15.0], 
    24: ['ULU17', 'Utilized', 'Natural lakes\xa0', 0.0, 50.0, 0.0, 25.0, 40.0, 5.0, 0.0, 30.0], 
    25: ['ULU18', 'Utilized', 'Flood plains & mudflats', 100.0, 50.0, 0.0, 40.0, 60.0, 0.0, 0.0, 0.0], 
    26: ['ULU19', 'Utilized', 'Saline sinks & playas & salinized soil', 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0], 
    27: ['ULU20', 'Utilized', 'Bare soil', 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0], 
    28: ['ULU21', 'Utilized', 'Waste land', 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0], 
    29: ['ULU22', 'Utilized', 'Moorland', 100.0, 0.0, 0.0, 5.0, 80.0, 0.0, 0.0, 15.0], 
    30: ['ULU23', 'Utilized', 'Wetland', 100.0, 50.0, 0.0, 5.0, 80.0, 0.0, 5.0, 10.0], 
    31: ['ULU24', 'Utilized', 'Mangroves', 100.0, 50.0, 0.0, 5.0, 80.0, 0.0, 5.0, 10.0], 
    32: ['ULU25', 'Utilized', 'Alien invasive species', 0.0, 0.0, 0.0, 0.0, 60.0, 0.0, 10.0, 30.0], 
    33: ['MLU1', 'Modified', 'Forest plantations', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    34: ['MLU2', 'Modified', 'Rainfed production pastures', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    35: ['MLU3', 'Modified', 'Rainfed crops - cereals', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    36: ['MLU4', 'Modified', 'Rainfed crops - root/tuber', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    37: ['MLU5', 'Modified', 'Rainfed crops - legumious', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    38: ['MLU6', 'Modified', 'Rainfed crops - sugar', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    39: ['MLU7', 'Modified', 'Rainfed crops - fruit and nuts', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    40: ['MLU8', 'Modified', 'Rainfed crops - vegetables and melons', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    41: ['MLU9', 'Modified', 'Rainfed crops - oilseed', 100.0, 0.0, 0.0, 45.0, 0.0, 15.0, 40.0, 0.0], 
    42: ['MLU10', 'Modified', 'Rainfed crops - beverage and spice', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    43: ['MLU11', 'Modified', 'Rainfed crops - other ', 100.0, 0.0, 0.0, 80.0, 0.0, 20.0, 0.0, 0.0], 
    44: ['MLU12', 'Modified', 'Mixed species agro-forestry', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    45: ['MLU13', 'Modified', 'Fallow & idle land', 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0], 
    46: ['MLU14', 'Modified', 'Dump sites & deposits', 0.0, 0.0, 0.0, 0.0, 60.0, 40.0, 0.0, 0.0], 
    47: ['MLU15', 'Modified', 'Rainfed homesteads and gardens (urban cities) - outdoor', 100.0, 0.0, 0.0, 0.0, 0.0, 35.0, 0.0, 65.0], 
    48: ['MLU16', 'Modified', 'Rainfed homesteads and gardens (rural villages) - outdoor', 100.0, 0.0, 0.0, 0.0, 0.0, 35.0, 0.0, 65.0], 
    49: ['MLU17', 'Modified', 'Rainfed industry parks - outdoor', 100.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 50.0], 
    50: ['MLU18', 'Modified', 'Rainfed parks (leisure & sports)', 100.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 85.0], 
    51: ['MLU19', 'Modified', 'Rural paved surfaces (lots, roads, lanes)', 100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0], 
    52: ['MWU1', 'Managed', 'Irrigated forest plantations', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    53: ['MWU2', 'Managed', 'Irrigated production pastures', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    54: ['MWU3', 'Managed', 'Irrigated crops - cereals', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    55: ['MWU4', 'Managed', 'Irrigated crops - root/tubers', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    56: ['MWU5', 'Managed', 'Irrigated crops - legumious', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    57: ['MWU6', 'Managed', 'Irrigated crops - sugar', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    58: ['MWU7', 'Managed', 'Irrigated crops - fruit and nuts', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    59: ['MWU8', 'Managed', 'Irrigated crops - vegetables and melons', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    60: ['MWU9', 'Managed', 'Irrigated crops - Oilseed', 100.0, 0.0, 0.0, 65.0, 0.0, 10.0, 25.0, 0.0], 
    61: ['MWU10', 'Managed', 'Irrigated crops - beverage and spice', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    62: ['MWU11', 'Managed', 'Irrigated crops - other', 100.0, 0.0, 0.0, 80.0, 0.0, 20.0, 0.0, 0.0], 
    63: ['MWU12', 'Managed', 'Managed water bodies (reservoirs, canals, harbors, tanks)', 0.0, 100.0, 0.0, 35.0, 5.0, 30.0, 20.0, 10.0], 
    64: ['MWU13', 'Managed', 'Greenhouses - indoor', 100.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0], 
    65: ['MWU14', 'Managed', 'Aquaculture', 0.0, 100.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    66: ['MWU15', 'Managed', 'Domestic households - indoor (sanitation)', 0.0, 100.0, 0.0, 0.0, 0.0, 35.0, 0.0, 65.0], 
    67: ['MWU16', 'Managed', 'Manufacturing & commercial industry - indoor', 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0], 
    68: ['MWU17', 'Managed', 'Irrigated homesteads and gardens (urban cities) - outdoor', 100.0, 0.0, 0.0, 30.0, 5.0, 15.0, 0.0, 50.0], 
    69: ['MWU18', 'Managed', 'Irrigated homesteads and gardens (rural villages) - outdoor', 100.0, 0.0, 0.0, 30.0, 5.0, 15.0, 0.0, 50.0], 
    70: ['MWU19', 'Managed', 'Irrigated industry parks - outdoor', 100.0, 0.0, 0.0, 0.0, 15.0, 35.0, 0.0, 50.0], 
    71: ['MWU20', 'Managed', 'Irrigated parks (leisure, sports)', 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0], 
    72: ['MWU21', 'Managed', 'Urban paved Surface (lots, roads, lanes)', 100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0], 
    73: ['MWU22', 'Managed', 'Livestock and domestic husbandry', 100.0, 0.0, 0.0, 90.0, 0.0, 10.0, 0.0, 0.0], 
    74: ['MWU23', 'Managed', 'Managed wetlands & swamps', 100.0, 50.0, 0.0, 0.0, 65.0, 10.0, 0.0, 25.0], 
    75: ['MWU24', 'Managed', 'Managed other inundation areas', 100.0, 50.0, 0.0, 0.0, 55.0, 20.0, 0.0, 25.0], 
    76: ['MWU25', 'Managed', 'Mining/ quarry & shale exploiration', 100.0, 50.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0], 
    77: ['MWU26', 'Managed', 'Evaporation ponds', 0.0, 100.0, 0.0, 0.0, 75.0, 25.0, 0.0, 0.0], 
    78: ['MWU27', 'Managed', 'Waste water treatment plants', 0.0, 100.0, 0.0, 0.0, 55.0, 45.0, 0.0, 0.0], 
    79: ['MWU28', 'Managed', 'Hydropower plants', 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 97.5, 2.5], 
    80: ['MWU29', 'Managed', 'Thermal power plants', 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0]
    }
    
    return lulc[lulc_version]

def get_sheet2_classes(version = '1.0'):
    sheet2_classes =dict()
    sheet2_classes['1.0'] = {
    'PROTECTED':    {'Forest': [1],
                    'Shrubland': [2],
                    'Natural grasslands':[3],
                    'Natural water bodies':[4],
                    'Wetlands':[5],
                    'Glaciers':[6],
                    'Others':[7]
                    },
    'UTILIZED':     {'Forest':[8,9,10,11],
                    'Shrubland':[14],
                    'Natural grasslands':[12,13,15,16,20],
                    'Natural water bodies':[23,24,],
                    'Wetlands':[17,19,25,30,31],
                    'Others':[18,21,22,26,27,28,29,32]
                    },
    'MODIFIED':     {'Rainfed crops': [34,35,36,37,38,39,40,41,42,43,44],
                    'Forest plantations':[33],
                    'Settlements':[47,48,49,50,51],
                    'Others':[45,46]
                    },
    'MANAGED CONVENTIONAL': {'Irrigated crops':[52,53,54,55,56,57,58,59,60,61,62],
                            'Managed water bodies':[63,65,74],
                            'Residential':[68,69,71,72],
                            'Industry':[67,70,76],
                            'Others':[75,78]
                            },
    'MANAGED NON_CONVENTIONAL':     {'Indoor domestic':[66],
                                    'Indoor industry':[0],
                                    'Greenhouses':[64],
                                    'Livestock and husbandry':[73],
                                    'Power and energy':[79,80],
                                    'Others':[77]
                                    }

    }
    
    return sheet2_classes[version]
    
def get_sheet3_classes(version = '1.0'):
    sheet3_classes =dict()
    sheet3_classes['1.0'] =  { 'CROP':         {'Cereals':              {'-':                     {'RAIN': [35], 'IRRI': [54]}},
                     
                                             'Non-cereals':          {'Root/tuber crops':          {'RAIN': [36],'IRRI': [55]},
                                                                      'Leguminous crops':            {'RAIN': [37],'IRRI': [56]},
                                                                      'Sugar crops':                 {'RAIN': [38],'IRRI': [57]},
                                                                      'Merged':                      {'RAIN': [36, 37, 38],'IRRI': [55, 56, 57]}},
                                             'Fruit & vegetables':   {'Vegetables & melons':         {'RAIN': [40],'IRRI': [59]},
                                                                      'Fruits & nuts':               {'RAIN': [39],'IRRI': [58]},
                                                                      'Merged':                      {'RAIN': [39, 40],'IRRI': [58, 59]}},
                                             'Oilseeds':            {'-':                   {'RAIN': [41],'IRRI': [60]}},
                                             'Feed crops':           {'-':                  {'RAIN': [34],'IRRI': [53]}},
                                             'Beverage crops':       {'-':              {'RAIN':[42] ,'IRRI': [61]}},
                                             'Other crops':          {'-':                 {'RAIN': [43],'IRRI': [62]}},
                                             #'Timber':               {'-':                      {'RAIN': [33],'IRRI': [52]}}
                                             },
                            'NON-CROP':     {'Fish (Aquaculture)':                 {'-':                        {'RAIN': [-1234], 'IRRI':[-1234]}},
                                             'Timber':               {'-':                      {'RAIN': [-1234], 'IRRI':[-1234]}},
                                             'Livestock':            {'Meat':                        {'RAIN': [-1234], 'IRRI':[-1234]},
                                                                      'Milk':                        {'RAIN': [-1234], 'IRRI':[-1234]}}
                                            }
                            }
    
    return sheet3_classes[version]
    
def get_hi_and_ec():
    HIWC = {
    'Alfalfa': [None, None],
    'Banana': [0.6, 0.76],
    'Barley': [None, None],
    'Beans': [0.16, 0.33],
    'Cassava': [0.6, 0.65],
    'Cashew': [0.03, 0.3],
    'Chickpea' : [.34, .15],
    'Coconut': [0.244, 0.0],
    'Coffee': [0.012, 0.88],
    'Cotton': [0.13, 0.2],
    'Eucalypt':[0.5, 0.50],
    'Grapes':[0.22, 0.75],
    'Grass':[0.45, 0.60],
    'Lucerne':[0.6, None],
    'Maize - Rainfed': [0.32, 0.26],
    'Maize - Irrigated': [0.39, 0.26],
    'Mango': [0.14, 0.84],
    'Olives': [0.012, 0.20],
    'Onions': [0.55, 0.85],
    'Oranges': [0.22, 0.85],
    'Palm Oil': [0.185, 0.1],
    'Pineapple': [None, None],
    'Potato': [0.8, 0.80],
    'Rice - Rainfed': [0.33, 0.16],
    'Rice - Irrigated': [0.42, 0.16],
    'Rubber': [0.013, 0.63],
    'Sorghum': [0.25, None],
    'Soybean': [None, None],
    'Sugarbeet': [0.6, 0.80],
    'Sugar cane': [0.69, 0.65],
    'Tapioca': [None, None],
    'Tea': [0.12, 0.50],
    'Wheat': [0.37, 0.15],
    'Fodder': [0.45,0.6],
    'Peanut':[0.03, 0.3],
    'Almond':[0.03, 0.3],
    'Pepper':[0.1, 0.5],
    'Melon':[0.8, 0.6]
        }
    
    return HIWC

def get_sheet1_classes(lulc_version = '4.0'):
    lulc_dict = get_lulcs(lulc_version = lulc_version)
    categories = ['Protected', 'Utilized', 'Modified', 'Managed']
    sheet1_classes = dict()
    for cat in categories:
        sheet1_classes[cat] = [key for key, value in zip(lulc_dict.keys(), lulc_dict.values()) if value[1] == cat]

    return sheet1_classes

def get_bluegreen_classes(version = '1.0'):
    
    gb_cats = dict()
    mvg_avg_len = dict()
    
    gb_cats['1.0'] = {
    'crops':                [53,54,55,56,57,58,59,60,61,62, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 33, 44],
    'perennial crops':      [52],
    'savanna':              [12,13],
    'water':                [63,74,75,77,4, 19, 23, 24],
    'forests':              [1, 8, 9, 10, 11, 17],
    'grass':                [3, 16, 20, 2, 14, 15],
    'other':                [68,69,70,71,72,76,78,73,67,65,66,64,79,80,6, 7, 18, 21, 22, 26, 27, 28, 29, 32, 45, 46, 47, 48, 49, 50, 51, 5, 25, 30, 31],
    }

    mvg_avg_len['1.0'] = {
    'crops':                2,
    'perennial crops':      3,
    'savanna':              4,
    'water':                1,
    'forests':              5,
    'grass':                1,
    'other':                1,
    }
    
    return gb_cats[version], mvg_avg_len[version]
    
def get_sheet4_6_classes(version = '1.0'):
    lucs = dict()
    
    lucs['1.0'] = {
    'Forests':              [1, 8, 9, 10, 11, 17],
    'Shrubland':            [2, 12, 14, 15],
    'Rainfed Crops':        [34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    'Forest Plantations':   [33, 44],
    'Natural Water Bodies': [4, 19, 23, 24],
    'Wetlands':             [5, 25, 30, 31],
    'Natural Grasslands':   [3, 13, 16, 20],
    'Other (Non-Manmade)':  [6, 7, 18, 21, 22, 26, 27, 28, 29, 32, 45, 46, 48, 49, 50, 51],
    'Irrigated crops':      [52,53,54,55,56,57,58,59,60,61,62],
    'Managed water bodies': [63,74,75,77],
    'Aquaculture':          [65],
    'Residential':          [47, 66, 68, 72],
    'Greenhouses':          [64],
    'Other':                [68,69,70,71,76,78]}
    
    return lucs[version]

def get_sheet4_6_fractions(version = '1.0'):
    sw_supply_fractions = dict()
    
    sw_supply_fractions['1.0'] = {
    'Forests':              0.001,
    'Shrubland':            0.001,
    'Rainfed Crops':        0.001,
    'Forest Plantations':   0.001,
    'Natural Water Bodies': 0.95,
    'Wetlands':             0.95,
    'Natural Grasslands':   0.30,
    'Other (Non-Manmade)':  0.50,
    'Irrigated crops':      9999,
    'Managed water bodies': 0.95,
    'Other':                0.50,
    'Residential':          0.90,
    'Greenhouses':          0.50,
    'Aquaculture':          0.95} 
    
    return  sw_supply_fractions[version]

def get_sheet3_empties(): 
    wp_y_irrigated_dictionary = {
    'Cereals': {'-': None},
    'Non-cereals': {'Root/tuber crops':None, 'Leguminous crops':None, 'Sugar crops':None, 'Merged':None},
    'Fruit & vegetables': {'Vegetables & melons':None, 'Fruits & nuts':None, 'Merged':None},
    'Oilseeds': {'-': None},
    'Feed crops': {'-': None},
    'Beverage crops': {'-': None},
    'Other crops': {'-': None}}
    
    wp_y_rainfed_dictionary = {
    'Cereals': {'-':None},
    'Non-cereals': {'Root/tuber crops':None, 'Leguminous crops':None, 'Sugar crops':None, 'Merged':None},
    'Fruit & vegetables': {'Vegetables & melons':None, 'Fruits & nuts':None, 'Merged':None},
    'Oilseeds': {'-': None},
    'Feed crops': {'-': None},
    'Beverage crops': {'-': None},
    'Other crops': {'-': None}}
    
    wp_y_non_crop_dictionary = {
    'Livestock': {'Meat':None, 'Milk':None},
    'Fish (Aquaculture)': {'-':None},
    'Timber': {'-':None}}
    
    return wp_y_irrigated_dictionary, wp_y_rainfed_dictionary, wp_y_non_crop_dictionary
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    