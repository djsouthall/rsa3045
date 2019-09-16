'''
This will make plots corresponding to the data taken in the
anechoic chamber.
'''
import visa
import time
from datetime import datetime
import pytz  # 3rd party: $ pip install pytz
import sys
import os
import numpy
sys.path.append('C:/Users/dsouthall/Desktop/RIGOL/python/rsa3045')
from display import getAttrDict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation
import h5py
import astropy
import astropy.time
import astropy.coordinates
import scipy
import scipy.signal
import scipy.interpolate


plt.ion()


if __name__ == '__main__':

    '''
    infiles = { '0ft'   :'./output/anechoic_test/RIGOL_CAPTURE_0ft_2019-09-10-17-51-55p767103+00-00.h5',
                '2ft'   :'./output/anechoic_test/RIGOL_CAPTURE_2ft_2019-09-10-17-18-41p815314+00-00.h5',
                '2.5ft' :'./output/anechoic_test/',
                '3ft'   :'./output/anechoic_test/'}
    '''
    '''
    infiles = { '2ft'   :'./output/anechoic_test/RIGOL_CAPTURE_2ft_2019-09-10-17-18-41p815314+00-00.h5',
                '2.5ft' :'./output/anechoic_test/RIGOL_CAPTURE_2.5ft_2019-09-10-18-28-19p557179+00-00.h5',
                '0ft'   :'./output/anechoic_test/RIGOL_CAPTURE_0ft_2019-09-10-17-51-55p767103+00-00.h5'}
    '''
    '''
    infiles = { 'Tx'   :'./output/anechoic_test/RIGOL_CAPTURE_Tx_2019-09-10-21-24-47p907323+00-00.h5',
                'terminator'   :'./output/anechoic_test/RIGOL_CAPTURE_terminator_2019-09-10-21-51-05p578881+00-00.h5'}
    '''
    '''    
    infiles = { 'Tx'   :'./output/anechoic_test/RIGOL_CAPTURE_day2_Tx_2019-09-11-17-05-42p310183+00-00.h5',
                'terminator'   :'./output/anechoic_test/RIGOL_CAPTURE_day2_terminator_2019-09-11-18-03-14p988401+00-00.h5'}
    '''
    '''
    infiles = { 'Tx_older'   :'./output/anechoic_test/RIGOL_CAPTURE_day2_Tx_2019-09-11-17-05-42p310183+00-00.h5',
                'Tx_newer'   :'./output/anechoic_test/RIGOL_CAPTURE_day2_passthrough-connector_2019-09-11-21-32-46p328445+00-00.h5',
                'Tx_newer_shelves':'./output/anechoic_test/RIGOL_CAPTURE_day2_passthrough-connector_2019-09-11-21-48-01p234572+00-00.h5',
                'terminator'   :'./output/anechoic_test/RIGOL_CAPTURE_day2_terminator_2019-09-11-19-13-35p989727+00-00.h5'}
    '''
    '''
    infiles = { 'Tx_OldSupply'      :'./output/anechoic_test/RIGOL_CAPTURE_day2_passthrough-connector_2019-09-11-21-48-01p234572+00-00.h5',
                'Tx_NewSupply'      :'./output/anechoic_test/RIGOL_CAPTURE_day2_new_power_supply_Tx_2019-09-11-23-16-26p532148+00-00.h5',
                'terminator'        :'./output/anechoic_test/RIGOL_CAPTURE_day2_new_power_supply_terminator_in_cage_2019-09-11-22-49-49p385204+00-00.h5'}
    '''
    '''
    infiles = { 'Tx_NewSupply_0ft_short_cable'      :'./output/anechoic_test/RIGOL_CAPTURE_day3_shorter_cable_Tx_0ft_2019-09-12-17-44-06p751369+00-00.h5',
                'Tx_NewSupply_2ft_short_cable'      :'./output/anechoic_test/RIGOL_CAPTURE_day3_shorter_cable_Tx_2019-09-12-17-18-20p011194+00-00.h5',
                'Tx_NewSupply_2ft_long_cable'      :'./output/anechoic_test/RIGOL_CAPTURE_day2_new_power_supply_Tx_2019-09-11-23-16-26p532148+00-00.h5',
                'terminator'        :'./output/anechoic_test/RIGOL_CAPTURE_day2_new_power_supply_terminator_in_cage_2019-09-11-22-49-49p385204+00-00.h5'}
    '''
    '''
    infiles = { 'Rx_NewSupply_2ft_short_cable'      :'./output/anechoic_test/RIGOL_CAPTURE_day3_shorter_cable_Rx_2ft_attempt2_2019-09-12-21-42-41p181070+00-00.h5',
                'Rx_NewSupply_0ft_short_cable'      :'./output/anechoic_test/RIGOL_CAPTURE_day3_shorter_cable_Rx_0ft_2019-09-12-20-49-09p908929+00-00.h5'}
    '''
    '''
    infiles = { 'Rx_NewSupply_2ft_short_cable_no_filter'      :'./output/anechoic_test/RIGOL_CAPTURE_day4_shorter_cable_Rx_2ft_no_filters_2019-09-13-18-06-31p818935+00-00.h5',
                'Rx_NewSupply_0ft_short_cable_no_filter'      :'./output/anechoic_test/RIGOL_CAPTURE_day4_shorter_cable_Rx_0ft_no_filters_2019-09-13-17-36-40p726519+00-00.h5'}
    '''
    '''
    infiles = { 'Rx_NewSupply_2ft_short_cable_no_filter_new_match1'      :'./output/anechoic_test/RIGOL_CAPTURE_day4_shorter_cable_Rx_2ft_no_filters_newmatch12019-09-13-20-17-09p272141+00-00.h5',
                'Rx_NewSupply_0ft_short_cable_no_filter_new_match1'      :'./output/anechoic_test/RIGOL_CAPTURE_day4_shorter_cable_Rx_0ft_no_filters_newmatch12019-09-13-20-49-10p552944+00-00.h5'}
    '''
    infiles = { 'Rx_NewSupply_2ft_short_cable_no_filter_new_match1'      :'./output/anechoic_test/RIGOL_CAPTURE_day5_shorter_cable_Rx_2ft_no_filters_newmatch12019-09-16-15-44-58p935908+00-00.h5',
                'Rx_NewSupply_0ft_short_cable_no_filter_new_match1'      :'./output/anechoic_test/RIGOL_CAPTURE_day5_shorter_cable_Rx_0ft_no_filters_newmatch12019-09-16-14-44-48p453236+00-00.h5'}
    
    baseline_keys = ['0ft','terminator','Rx_NewSupply_0ft_short_cable','Rx_NewSupply_0ft_short_cable_no_filter','Rx_NewSupply_0ft_short_cable_no_filter_new_match1']

    ###################################################

    fig = plt.figure()
    plt.subplot(2,1,1)
    ax1 = plt.gca()
    #Pretty up plot
    plt.grid(which='both', axis='both')
    ax1.minorticks_on()
    ax1.grid(b=True, which='major', color='k', linestyle='-')
    ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.ylabel('dBm',fontsize=16)
    plt.xlabel('Frequency (MHz)',fontsize=16)

    loop_index = 0
    all_powers = {}
    for key, path in infiles.items():
        with h5py.File(path, 'r') as file: 
            try:
                attrs = getAttrDict(file)
                frequencies = numpy.linspace(attrs['freq_L'],attrs['freq_R'],attrs['sweep_points'])
                df = numpy.diff(frequencies)[0]
                n_sweeps = numpy.shape(file['power'])[0]

                all_powers[key] = file['power'][...][[0,1,2,3,4],:]

                average_power = 10*numpy.log10(numpy.mean(10**(file['power'][...][[0,1,2,3,4],:]/10),axis=0))

                if key in baseline_keys:
                    average_power_0ft = average_power
                
                #Plot average spectra
                plt.plot(frequencies/1e6,average_power,label='Averaged Spectrum for %s'%key,alpha=0.8)
                
                file.close()
                loop_index += 1
            except Exception as e:
                print('Error in main() while file is open.  Closing file.')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                file.close()

   



    plt.legend(loc='upper right',fontsize=16)

    plt.subplot(2,1,2,sharex=ax1)
    ax = plt.gca()
    #Pretty up plot
    plt.grid(which='both', axis='both')
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.ylabel('dB',fontsize=16)
    plt.xlabel('Frequency (MHz)',fontsize=16)

    for key, path in infiles.items():
        if key in baseline_keys:
            continue
        with h5py.File(path, 'r') as file: 
            try:
                attrs = getAttrDict(file)
                frequencies = numpy.linspace(attrs['freq_L'],attrs['freq_R'],attrs['sweep_points'])
                df = numpy.diff(frequencies)[0]
                n_sweeps = numpy.shape(file['power'])[0]

                average_power = 10*numpy.log10(numpy.mean(10**(file['power'][...][[0,1,2,3,4],:]/10),axis=0)) - average_power_0ft
                
                #Plot average spectra
                plt.plot(frequencies/1e6,average_power,label='Averaged Spectrum for %s'%key,alpha=0.8)
                
                file.close()
                loop_index += 1
            except Exception as e:
                print('Error in main() while file is open.  Closing file.')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                file.close()


    plt.figure()

    sub_index = 1
    for key, values in all_powers.items():
        plt.subplot(len(list(all_powers.keys())),1,sub_index)
        for power in values:
            plt.plot(frequencies/1e6,power)
        ax = plt.gca()
        plt.title(key)
        #Pretty up plot
        plt.grid(which='both', axis='both')
        ax.minorticks_on()
        ax.grid(b=True, which='major', color='k', linestyle='-')
        ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.ylabel('dB',fontsize=16)
        plt.xlabel('Frequency (MHz)',fontsize=16)
        sub_index += 1