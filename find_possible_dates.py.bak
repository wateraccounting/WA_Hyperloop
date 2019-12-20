# -*- coding: utf-8 -*-
"""
Created on Tue Oct 09 11:56:28 2018

@author: cmi001
"""
import os

def find_possible_dates(str_):
    """
    Finds index of possible year and month in a string if the date is of format 
    yyyymm or yyyy{char}mm for years between 1900 and 2020
    """
    basename = os.path.basename(str_)
    months =['{0:02d}'.format(i) for i in range(1,12)]
    years = ['{0}'.format(i) for i in range(1900, 2020)]
    options = {}
    i = 0
    for y in years:
        index = basename.find(y)
        if index > 0:
            if basename[index+4:index+6] in months:
                options[i] = ([index, index+4], [index+4, index+6])
                i +=1
            elif basename[index+5:index+7] in months:
                options[i] = ([index, index+4], [index+5, index+7])
                i+=1
            else:
                options[i] = [index, index+4]
    if len(options.keys()) == 0:
        print 'Could not find datestring'
    elif len(options.keys()) > 1:
        print 'Multiple possible locations for datestring'
    
    return options[0]

def find_possible_dates_negative(str_):
    """
    Finds index of possible year and month in a string if the date is of format 
    yyyymm or yyyy{char}mm for years between 1900 and 2020
    """
    basename = os.path.basename(str_)
    months =['{0:02d}'.format(i) for i in range(1,12)]
    years = ['{0}'.format(i) for i in range(1900, 2020)]
    options = {}
    i = 0
    for y in years:
        index1 = basename.find(y)
        index = index1 - len(basename)
        if index1 > 0:
            if basename[index+4:index+6] in months:
                options[i] = ([index, index+4], [index+4, index+6])
                i +=1
            elif basename[index+5:index+7] in months:
                options[i] = ([index, index+4], [index+5, index+7])
                i+=1
            else:
                options[i] = [index, index+4]
    if len(options.keys()) == 0:
        print 'Could not find datestring'
    elif len(options.keys()) > 1:
        print 'Multiple possible locations for datestring'
    
    return options[0]
