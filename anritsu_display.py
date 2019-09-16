'''
This will make plots corresponding to the data taken with the anritsu SA
as Steph has programmed it.

To use this script, ensure you have set the location of the antenna correctly
using the lat lon parameters.  Then put the filename you wish to examine in the
list filenames.

Then in an ipython terminal you can just type %run anritsu_display.py

Additional plotting parameters are available to allow you to highlight
subsets of the spectrum for tracking, or to exclude certain regions from any
measurements.  
'''
import visa
import time
from datetime import datetime
import pytz  # 3rd party: $ pip install pytz
import sys
import os
import numpy
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
import pandas as pd

plt.ion()



def getSagCoords(run_start_time_utc_timestamp, run_stop_time_utc_timestamp, antenna_latlon, n_points=1000, plot=False):
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
        lat = antenna_latlon[0]
        lon = antenna_latlon[1]
        antenna_location = astropy.coordinates.EarthLocation(lat=lat,lon=lon) #Barcroft Station
        
        #Location of Sagitarius A in Right Ascension and Declination
        ra = 266.427 * astropy.units.deg # Could also use Angle
        dec = -29.007778 * astropy.units.deg  # Astropy Quantity
        sagA = astropy.coordinates.SkyCoord(ra, dec, frame='icrs') #ICRS=Internation Celestial Reference System
        
        #Setting up astropy time objects
        time_window_utc_timestamp = numpy.linspace(0,(run_stop_time_utc_timestamp-run_start_time_utc_timestamp),n_points)
        
        start_time = astropy.time.Time(run_start_time_utc_timestamp,format='unix')
        stop_time = astropy.time.Time(run_stop_time_utc_timestamp,format='unix')
        time_window_object = astropy.time.Time(time_window_utc_timestamp,format='unix')


        time_window = (time_window_utc_timestamp/3600.0)*astropy.units.hour

        #Setting up frame for Sagitarius A
        frame = astropy.coordinates.AltAz(obstime=start_time+time_window,location=antenna_location)
        sagAaltazs = sagA.transform_to(frame)
        sun_loc = astropy.coordinates.get_sun(time_window_object).transform_to(frame)

        if plot == True:
            fig = plt.figure()
            fig.canvas.set_window_title('Sgr A* Alt')
            ax = plt.gca()
            #Make Landscape
            ax.axhspan(0, 90, alpha=0.2,label='Sky', color = 'blue')
            ax.axhspan(-90, 0, alpha=0.2,label='Ground', color = 'green')
            plt.plot(time_window, sagAaltazs.alt,label='Sgr A*')
            plt.plot(time_window, sun_loc.alt,label='Sun')

            #plt.ylim(1, 4)
            plt.xlabel('Hours from Start of Run',fontsize=16)
            plt.ylabel('Altitutude (Degrees)',fontsize=16)
            plt.ylim([-90,90])
            plt.legend(fontsize=16)

        return time_window_utc_timestamp, sagAaltazs.alt, sagAaltazs.az, sun_loc.alt, sun_loc.az
    except Exception as e:
        print('Error in getSagCoords().')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == '__main__':

    filenames = ['./output/anritsu/specanalyzer_run6_2019-09-12_11-16-40.hdf5',
                './output/anritsu/specanalyzer_run1_2019-09-13_16-55-31.hdf5']

    sag_plot = True
    df_multiplier = 2
    width = (0,4)
    prominence = 5
    lw = None #Linewidth None is default
    
    all_binnings = [None] #How many sweeps to put in each bin.  Put None in the list for no binning.
    freq_ROI= [[0,200]]
    ignore_range = []
    ignore_peaks = False
    ignore_min_max = False
    plot_total_averaged = True
    plot_stacked_spectra = True
    plot_comparison = True
    plot_comparison_mins = True
    plot_animated = True
    min_method = 3


    #Barcroft Station
    lat = 37.583342
    lon = -118.236484
    antenna_latlon = (lat,lon)

    timezone = pytz.timezone('US/Pacific')


    for filename in filenames:
        with h5py.File(filename, 'r') as file:
            try:
                ts = []
                powers = []

                dataframe = pd.read_hdf(filename,'spectra')
                for key in dataframe.keys():
                    time = datetime.strptime(key,'%Y-%m-%d-%W-%H-%M-%S')
                    ts.append(time.timestamp())
                    freq_MHz = numpy.array(dataframe[key]['freq_hz'])/1.0e6
                    freq_Hz = numpy.array(dataframe[key]['freq_hz'])
                    dBm = dataframe[key]['power_dBm']
                    powers.append(dBm)
                df = freq_Hz[1] - freq_Hz[0]
                powers = numpy.array(powers)
                ts = numpy.array(ts)
                file.close() #Open whenever writing to it
                run_start_time_utc_timestamp = min(ts)
                run_stop_time_utc_timestamp = max(ts)
                utc_timestamps, sagA_alt, sagA_az, sun_alt, sun_az = getSagCoords(run_start_time_utc_timestamp, run_stop_time_utc_timestamp,antenna_latlon, plot=sag_plot)
                
                average_power = 10*numpy.log10(numpy.mean(10**(powers/10),axis=0))
                peaks, _ = scipy.signal.find_peaks(average_power,width = width,prominence=prominence)
                if ignore_peaks == True:
                    for peak in peaks:
                        ignore_range.append([(freq_Hz[peak]-df_multiplier*df)/1e6,(freq_Hz[peak]+df_multiplier*df)/1e6])

                ignore_cut = numpy.ones_like(freq_Hz,dtype=bool)
                for i in ignore_range:
                    ignore_cut = numpy.multiply(~numpy.logical_and(freq_Hz/1e6 <= i[1],freq_Hz/1e6 >= i[0] ), ignore_cut) #Falses in this result in those freq_Hz not being considered in averages.

                if plot_total_averaged:
                    #Plot Full Run Average Spectra
                    fig = plt.figure()
                    fig.canvas.set_window_title('Total Average Spectra')
                    ax = plt.gca()
                    
                    #Plot regions of interest
                    ROI_cm = plt.get_cmap('gist_rainbow')
                    for ROI_index, ROI in enumerate(freq_ROI):
                        frequency_cut = numpy.logical_and(freq_Hz/1e6 >= ROI[0], freq_Hz/1e6 <= ROI[1])
                        if numpy.all(frequency_cut):
                            continue
                        ax.axvspan(ROI[0], ROI[1], alpha=0.5,label='Frequency ROI = %s'%str(ROI), color = ROI_cm(ROI_index/len(freq_ROI)))

                    for ROI_index, ROI in enumerate(ignore_range):
                        frequency_cut = numpy.logical_and(freq_Hz/1e6 >= ROI[0], freq_Hz/1e6 <= ROI[1])
                        ax.axvspan(ROI[0], ROI[1], alpha=0.8,hatch = 'x', color = 'gray')

                    #Plot average spectra
                    plt.plot(freq_Hz/1e6,average_power,label='Averaged Spectrum')
                    
                    #Pretty up plot
                    plt.grid(which='both', axis='both')
                    ax = plt.gca()
                    ax.minorticks_on()
                    ax.grid(b=True, which='major', color='k', linestyle='-')
                    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.ylabel('dBm',fontsize=16)
                    plt.xlabel('Frequency (MHz)',fontsize=16)
                    plt.title('Average Spectum Over %0.2fh Run'%((utc_timestamps[-1]-utc_timestamps[0])/3600.0))
                    plt.scatter(freq_Hz[peaks]/1e6,average_power[peaks],c='r',s=20,label='Identified Peaks')
                    plt.legend(loc='upper right',fontsize=16)

                start_tod = datetime.fromtimestamp(run_start_time_utc_timestamp).astimezone(timezone)
                stop_tod = datetime.fromtimestamp(run_stop_time_utc_timestamp).astimezone(timezone)

                print('Run Start:',start_tod)
                print('Run Stop:',stop_tod)

                for sweeps_per_bin in all_binnings:
                    if sweeps_per_bin is not None:

                        binned_power = numpy.zeros((numpy.ceil(numpy.shape(powers)[0] / sweeps_per_bin).astype(int),numpy.shape(powers)[1]))
                        binned_times = numpy.zeros((numpy.ceil(numpy.shape(powers)[0] / sweeps_per_bin).astype(int)))

                        for index in range((numpy.ceil(numpy.shape(powers)[0] / sweeps_per_bin).astype(int))):
                            binned_power[index] = 10*numpy.log10(numpy.mean(10**(powers[index*sweeps_per_bin:min((index+1)*sweeps_per_bin,len(ts))]/10),axis=0)) 
                            binned_times[index] = ((ts[min((index+1)*sweeps_per_bin - 1,len(ts)-1)] + ts[(index)*sweeps_per_bin])/2.0 - min(ts))/3600.
                        
                    else:
                        binned_power = powers
                        binned_times = (ts-min(ts))/3600.

                    interpolated_sun_alt = scipy.interpolate.interp1d(utc_timestamps.flatten()/3600,sun_alt.flatten())(binned_times)
                    
                    if plot_stacked_spectra:
                        #Plot All Spectra
                        fig = plt.figure()
                        fig.canvas.set_window_title('Stacked Spectra')
                        ax = plt.gca()
                        
                        #Plot regions of interest
                        ROI_cm = plt.get_cmap('gist_rainbow')
                        for ROI_index, ROI in enumerate(freq_ROI):
                            frequency_cut = numpy.logical_and(freq_Hz/1e6 >= ROI[0], freq_Hz/1e6 <= ROI[1])
                            if numpy.all(frequency_cut):
                                continue
                            ax.axvspan(ROI[0], ROI[1], alpha=0.5,label='Frequency ROI = %s'%str(ROI), color = ROI_cm(ROI_index/len(freq_ROI)))

                        for ROI_index, ROI in enumerate(ignore_range):
                            frequency_cut = numpy.logical_and(freq_Hz/1e6 >= ROI[0], freq_Hz/1e6 <= ROI[1])
                            #ax.axvspan(ROI[0], ROI[1], alpha=0.8,hatch = 'x',label='Frequency IGNORED = %s'%str(ROI), color = 'gray')
                            ax.axvspan(ROI[0], ROI[1], alpha=0.8,hatch = 'x', color = 'gray')
                        


                        #Plot average sepectra
                        cm = plt.get_cmap('gist_rainbow')
                        for index, row in enumerate(binned_power):
                            plt.plot(freq_Hz/1e6,row,alpha=0.5,color=cm(index/len(binned_times)))
                        
                        #Pretty up plot
                        plt.grid(which='both', axis='both')
                        ax = plt.gca()
                        ax.minorticks_on()
                        ax.grid(b=True, which='major', color='k', linestyle='-')
                        ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.ylabel('dBm',fontsize=16)
                        plt.xlabel('Frequency (MHz)',fontsize=16)



                    if plot_comparison:
                        #PLot the mean in ROI as a function of time. 
                        ROI_cm = plt.get_cmap('gist_rainbow')

                        fig = plt.figure()
                        fig.canvas.set_window_title('Mean Noise Alt Comparison')
                        plt.subplot(2,1,1)
                        ax = plt.gca()

                        #x = [datetime.datetime.strptime(datetime.fromtimestamp(d).replace(tzinfo=pytz.utc).astimezone(pytz.timezone('America/Chicago')),'%m/%d/%Y').date() for d in times]
                        for ROI_index, ROI in enumerate(freq_ROI):
                            frequency_cut = numpy.logical_and(freq_Hz/1e6 >= ROI[0], freq_Hz/1e6 <= ROI[1])
                            frequency_cut = numpy.logical_and(frequency_cut,ignore_cut)
                            if numpy.any(frequency_cut) == False:
                                continue
                            average_power = 10*numpy.log10(numpy.mean(10**(binned_power[:,frequency_cut]/10),axis=1))
                            
                            if ignore_min_max == True:
                                min_max_cut = ~numpy.isin(average_power, [min(average_power),max(average_power)])
                                average_power = average_power[min_max_cut]
                                x = binned_times[min_max_cut]
                            else:
                                x = binned_times

                            plt.plot(x, average_power,label='Frequency ROI = %s'%str(ROI), alpha=0.8, color = ROI_cm(ROI_index/len(freq_ROI)),linewidth=lw) 
                            plt.ylabel('Average Power in ROI (dBm)',fontsize=16)

                        #Pretty up plot
                        plt.grid(which='both', axis='both')
                        ax = plt.gca()
                        ax.minorticks_on()
                        ax.grid(b=True, which='major', color='k', linestyle='-')
                        ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.xlabel('Elapsed Time Since Beginning of Run (Hours)',fontsize=16)
                        plt.legend(loc='upper right',fontsize=16)

                        plt.subplot(2,1,2,sharex = ax)
                        ax = plt.gca()
                        
                        #Make Landscape
                        ax.axhspan(0, 90, alpha=0.2,label='Sky', color = 'blue')
                        ax.axhspan(-90, 0, alpha=0.2,label='Ground', color = 'green')
                        #ax.axvspan(0, 90, alpha=0.2,label='Sky', color = 'blue')


                        plt.plot(utc_timestamps/3600.0, sagA_alt, c='r',label='Sgr A*',linewidth=lw)
                        plt.plot(utc_timestamps/3600.0, sun_alt, c='k',label='Sun',linewidth=lw)

                        plt.xlabel('Hours from Start of Run',fontsize=16)
                        plt.ylabel('Altitutude (Degrees)',fontsize=16)
                        plt.grid(which='both', axis='both')
                        ax.minorticks_on()
                        ax.grid(b=True, which='major', color='k', linestyle='-')
                        ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.xlabel('Elapsed Time Since Beginning of Run (Hours)',fontsize=16)
                        plt.legend(loc='upper right',fontsize=16)
                        #plt.ylim((min(sagA_alt.degree)-5,max(sagA_alt.degree)+5))
                        plt.ylim((-90,90))

                    if plot_comparison_mins:
                        #PLot the mean in ROI as a function of time. 
                        ROI_cm = plt.get_cmap('gist_rainbow')

                        fig = plt.figure()
                        fig.canvas.set_window_title('Min Noise Alt Comparison')
                        plt.subplot(2,1,1)
                        ax = plt.gca()

                        #x = [datetime.datetime.strptime(datetime.fromtimestamp(d).replace(tzinfo=pytz.utc).astimezone(pytz.timezone('America/Chicago')),'%m/%d/%Y').date() for d in times]
                        for ROI_index, ROI in enumerate(freq_ROI):
                            frequency_cut = numpy.logical_and(freq_Hz/1e6 >= ROI[0], freq_Hz/1e6 <= ROI[1])
                            frequency_cut = numpy.logical_and(frequency_cut,ignore_cut)
                            if numpy.any(frequency_cut) == False:
                                continue

                            if min_method == 1:
                                min_power = numpy.min(binned_power[:,frequency_cut],axis=1)
                            elif min_method == 2:
                                #TRY GETTING MIN IN FREQ RANGE BEFORE BINNING IN TIME
                                min_powers = numpy.min(powers[:,frequency_cut],axis=1)
                                if sweeps_per_bin is not None:
                                    min_power = numpy.zeros_like(binned_times)
                                    for index in range((numpy.ceil(numpy.shape(powers)[0] / sweeps_per_bin).astype(int))):
                                        min_power[index] = 10.0*numpy.log10(numpy.mean(10.0**(min_powers[index*sweeps_per_bin:min((index+1)*sweeps_per_bin,len(ts))]/10.0))) 
                                else:
                                    min_power = min_powers
                            elif min_method == 3:
                                #TRY GETTING MIN IN FREQ RANGE BEFORE BINNING IN TIME
                                min_powers = numpy.min(powers[:,frequency_cut],axis=1)
                                if sweeps_per_bin is not None:
                                    min_power = numpy.zeros_like(binned_times)
                                    for index in range((numpy.ceil(numpy.shape(powers)[0] / sweeps_per_bin).astype(int))):
                                        min_power[index] = numpy.min(min_powers[index*sweeps_per_bin:min((index+1)*sweeps_per_bin,len(ts))]) 
                                else:
                                    min_power = min_powers

                            
                            x = binned_times

                            plt.plot(x, min_power,label='Frequency ROI = %s'%str(ROI), alpha=0.8, color = ROI_cm(ROI_index/len(freq_ROI)),linewidth=lw) 
                            plt.ylabel('Minimum Power in ROI (dBm)',fontsize=16)

                        #Pretty up plot
                        plt.grid(which='both', axis='both')
                        ax = plt.gca()
                        ax.minorticks_on()
                        ax.grid(b=True, which='major', color='k', linestyle='-')
                        ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.xlabel('Elapsed Time Since Beginning of Run (Hours)',fontsize=16)
                        plt.legend(loc='upper right',fontsize=16)

                        plt.subplot(2,1,2,sharex = ax)
                        ax = plt.gca()
                        
                        #Make Landscape
                        ax.axhspan(0, 90, alpha=0.2,label='Sky', color = 'blue')
                        ax.axhspan(-90, 0, alpha=0.2,label='Ground', color = 'green')
                        #ax.axvspan(0, 90, alpha=0.2,label='Sky', color = 'blue')


                        plt.plot(utc_timestamps/3600.0, sagA_alt, c='r',label='Sgr A*',linewidth=lw)
                        plt.plot(utc_timestamps/3600.0, sun_alt, c='k',label='Sun',linewidth=lw)

                        plt.xlabel('Hours from Start of Run',fontsize=16)
                        plt.ylabel('Altitutude (Degrees)',fontsize=16)
                        plt.grid(which='both', axis='both')
                        ax.minorticks_on()
                        ax.grid(b=True, which='major', color='k', linestyle='-')
                        ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.xlabel('Elapsed Time Since Beginning of Run (Hours)',fontsize=16)
                        plt.legend(loc='upper right',fontsize=16)
                        #plt.ylim((min(sagA_alt.degree)-5,max(sagA_alt.degree)+5))
                        plt.ylim((-90,90))

                    if plot_animated:
                        #Plot All Spectra
                        fig, ax = plt.subplots()
                        fig.canvas.set_window_title('Animated')
                        #Plot regions of interest
                        ROI_cm = plt.get_cmap('gist_rainbow')

                        for ROI_index, ROI in enumerate(freq_ROI):
                            frequency_cut = numpy.logical_and(freq_Hz/1e6 >= ROI[0], freq_Hz/1e6 <= ROI[1])
                            if numpy.all(frequency_cut):
                                continue
                            ax.axvspan(ROI[0], ROI[1], alpha=0.5,label='Frequency ROI = %s'%str(ROI), color = ROI_cm(ROI_index/len(freq_ROI)))

                        for ROI_index, ROI in enumerate(ignore_range):
                            frequency_cut = numpy.logical_and(freq_Hz/1e6 >= ROI[0], freq_Hz/1e6 <= ROI[1])
                            #ax.axvspan(ROI[0], ROI[1], alpha=0.8,hatch = 'x',label='Frequency IGNORED = %s'%str(ROI), color = 'gray')
                            ax.axvspan(ROI[0], ROI[1], alpha=0.8,hatch = 'x', color = 'gray')
                        

                        #Pretty up plot
                        plt.grid(which='both', axis='both')
                        ax = plt.gca()
                        ax.minorticks_on()
                        ax.grid(b=True, which='major', color='k', linestyle='-')
                        ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.ylabel('dBm',fontsize=16)
                        plt.xlabel('Frequency (MHz)',fontsize=16)
                        cm = plt.get_cmap('gist_rainbow')
                        line, = ax.plot(freq_Hz/1e6,binned_power[0],color='k',label='%.2f h Elapsed\n Sun Alt = %0.2f'%(binned_times[0],interpolated_sun_alt[0]))
                        leg = plt.legend(loc='upper right',fontsize=16)
                        def update(row):
                            line.set_ydata(binned_power[row])
                            leg.get_texts()[0].set_text('%.2f h Elapsed\n Sun Alt = %0.2f'%(binned_times[row],interpolated_sun_alt[row]))
                            return line, ax

                        anim = matplotlib.animation.FuncAnimation(fig, update, frames=numpy.arange(len(binned_times)), interval=10*1000 / len(binned_times))

            except Exception as e:
                print('Error in main() while file is open.  Closing file.')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                file.close()
