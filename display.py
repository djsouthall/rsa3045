'''
This will make plots corresponding to the test.py data.
'''
import visa
import time
from datetime import datetime
import pytz  # 3rd party: $ pip install pytz
import sys
import os
import numpy
import matplotlib.pyplot as plt
import h5py
import astropy
import astropy.time
import astropy.coordinates


plt.ion()

def getSagCoords(run_start_time_utc_timestamp, run_stop_time_utc_timestamp, n_points=1000, plot=False):
    '''
    Sagitarius A is functionally the same as centre of galaxy.

    Parameters
    ----------
    run_start_time_utc_timestamp : float
        The utc timestamp corresponding to the start of the run.
    run_stop_time_utc_timestamp : float
        The utc timestamp corresponding to the stop of the run.
    n_points : int
        The number of points between the start and stop time to calculate the altitude. 
    plot : bool
        Enables plotting if True.
    '''
    try:
        antenna_location = astropy.coordinates.EarthLocation(lat=41.791519,lon=-87.601641) #Site of test on ERC Roof, Chicago
        
        #Location of Sagitarius A in Right Ascension and Declination
        ra = 266.427 * astropy.units.deg # Could also use Angle
        dec = -29.007778 * astropy.units.deg  # Astropy Quantity
        sagA = astropy.coordinates.SkyCoord(ra, dec, frame='icrs') #ICRS=Internation Celestial Reference System
        
        #Setting up astropy time objects
        start_time = astropy.time.Time(run_start_time_utc_timestamp,format='unix')
        stop_time = astropy.time.Time(run_stop_time_utc_timestamp,format='unix')

        time_window_utc_timestamp = numpy.linspace(0,(run_stop_time_utc_timestamp-run_start_time_utc_timestamp),n_points)
        time_window = (time_window_utc_timestamp/3600.0)*astropy.units.hour

        #Setting up frame for Sagitarius A
        frame = astropy.coordinates.AltAz(obstime=start_time+time_window,location=antenna_location)
        sagAaltazs = sagA.transform_to(frame)

        if plot == True:
            plt.figure()
            plt.plot(time_window, sagAaltazs.alt)
            #plt.ylim(1, 4)
            plt.xlabel('Hours from Start of Run')
            plt.ylabel('Altitutude (Degrees)')

        return time_window_utc_timestamp, sagAaltazs.alt, sagAaltazs.az
    except Exception as e:
        print('Error in getSagCoords().')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def getAttrDict(file):
    '''
    This will produce a dictionary from the attributes in an h5py file.

    Parameters
    ----------
    file : h5py file
        The file you want the attr dict from.
    '''
    #Set range of measurement
    try:
        attrs = {}
        for attribute in list(file.attrs):
            attrs[attribute] = file.attrs[attribute]
        return attrs
    except Exception as e:
        print('Error in getAttrDict().')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == '__main__':
    filename = './output/RIGOL_CAPTURE_2019-08-27-23-29-32p501689+00-00.h5'
    
    freq_ROI = (0,200) #MHz, frequencies within this range will be considered for noise calculations.  This are inclusive

    with h5py.File(filename, 'r') as file:
        try:
            frequency_cut = numpy.logical_and(file['frequencies'][...] >= freq_ROI[0], file['frequencies'][...] <= freq_ROI[1])
            attrs = getAttrDict(file)
            n_sweeps = numpy.shape(file['power'])[0]
            utc_timestamps, sagA_alt, sagA_az = getSagCoords(file['utc_start_stamp'][0], file['utc_stop_stamp'][-1], plot=False)

            if True:
                #Plot Full Run Average Spectra
                plt.figure()
                average_power = 10*numpy.log10(numpy.mean(10**(file['power'][...]/10),axis=0))  
                plt.plot(file['frequencies'][...]/1e6,average_power,label='Averaged Spectrum')
                plt.axvline(freq_ROI[0], linestyle='--',color='r',label='Frequency ROI Bounds')
                plt.axvline(freq_ROI[1], linestyle='--',color='r')
                plt.grid(which='both', axis='both')
                ax = plt.gca()
                ax.grid(b=True, which='major', color='k', linestyle='-')
                ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.ylabel('dBm')
                plt.xlabel('Frequency (MHz)')
                plt.title('Average Spectum Over %0.2fh Run'%((utc_timestamps[-1]-utc_timestamps[0])/3600.0))
                plt.legend()

            if True:
                #PLot the noise RMS in ROI as a function of time. 
                


            file.close() #Open whenever writing to it
        except Exception as e:
            print('Error in main() while file is open.  Closing file.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            file.close()
