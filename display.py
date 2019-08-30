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
import matplotlib.dates as mdates
import matplotlib.animation
import h5py
import astropy
import astropy.time
import astropy.coordinates
import scipy
import scipy.signal


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
            plt.figure()
            plt.plot(time_window, sagAaltazs.alt,label='Sgr A*')
            plt.plot(time_window, sun_loc.alt,label='Sun')

            #plt.ylim(1, 4)
            plt.xlabel('Hours from Start of Run')
            plt.ylabel('Altitutude (Degrees)')


        return time_window_utc_timestamp, sagAaltazs.alt, sagAaltazs.az, sun_loc.alt, sun_loc.az
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
    plt.close('all')
    filename = './output/RIGOL_CAPTURE_2019-08-28-16-59-35p666232+00-00.h5'
    
    #freq_ROI = [[0,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,90],[90,100],[100,110]]#[[0,200],[30,80],[25,45],[60.5,62.5],[77,79.5]] #MHz, frequencies within this range will be considered for noise calculations.  This are inclusive
    #Alternating signal at 73 MHz (72.94 MHz)
    #step = 10
    #freq_ROI = list(zip(numpy.arange(20,90,step),numpy.arange(20,90,step)+step))
    #freq_ROI = [[0,200],[35,80],[25,45],[46.82,47.3],[60.5,62.5],[77,79.5]]
    freq_ROI = [[0.0,0.2],[0.3,200],[42,44],[60,62]]
    #freq_ROI = [[0,200],[35,80],[25,45],[46.82,47.3],[57,59],[60.5,62.5],[77,79.5]]
    ignore_range = [[0.2,28],[27,27.5],[35,37],[40.95,41.05],[49,55],[63.87,63.9],[72.8,73.1],[72.92,72.97],[88.4,88.5],[88.8,88.9],[86,200]]
    sweeps_per_bin = 40 #Put None to use capturing bins.
    all_binnings = [None, 5, 10, 20, 40, 80]

    ignore_min_max = False
    plot_norm = True
    sag_plot = False
    lw = None #Linewidth None is default
    ignore_peaks = True
    df_multiplier = 2.1

    plot_total_averaged = True
    plot_stacked_spectra = False
    plot_comparison = True
    plot_animated = False

    

    with h5py.File(filename, 'r') as file:
        try:
            attrs = getAttrDict(file)
            frequencies = numpy.linspace(attrs['freq_L'],attrs['freq_R'],attrs['sweep_points'])
            df = numpy.diff(frequencies)[0]
            n_sweeps = numpy.shape(file['power'])[0]

            run_start_time_utc_timestamp = file['utc_start_stamp'][0]
            start_aware = datetime.fromtimestamp(run_start_time_utc_timestamp).replace(tzinfo=pytz.utc)
            start_aware_chicago = start_aware.astimezone(pytz.timezone('America/Chicago'))
            run_stop_time_utc_timestamp = file['utc_stop_stamp'][-1]
            stop_aware = datetime.fromtimestamp(run_stop_time_utc_timestamp).replace(tzinfo=pytz.utc)
            stop_aware_chicago = stop_aware.astimezone(pytz.timezone('America/Chicago'))
            
            utc_timestamps, sagA_alt, sagA_az, sun_alt, sun_az = getSagCoords(run_start_time_utc_timestamp, run_stop_time_utc_timestamp, plot=sag_plot)

            average_power = 10*numpy.log10(numpy.mean(10**(file['power'][...]/10),axis=0))
            peaks, _ = scipy.signal.find_peaks(average_power,width = (0,4),prominence=2)
            if ignore_peaks == True:
                for peak in peaks:
                    ignore_range.append([(frequencies[peak]-df_multiplier*df)/1e6,(frequencies[peak]+df_multiplier*df)/1e6])

            ignore_cut = numpy.ones_like(frequencies,dtype=bool)
            for i in ignore_range:
                ignore_cut = numpy.multiply(~numpy.logical_and(frequencies/1e6 <= i[1],frequencies/1e6 >= i[0] ), ignore_cut) #Falses in this result in those frequencies not being considered in averages.

            if plot_total_averaged:
                #Plot Full Run Average Spectra
                plt.figure()
                ax = plt.gca()
                
                #Plot regions of interest
                ROI_cm = plt.get_cmap('gist_rainbow')
                for ROI_index, ROI in enumerate(freq_ROI):
                    frequency_cut = numpy.logical_and(frequencies/1e6 >= ROI[0], frequencies/1e6 <= ROI[1])
                    if numpy.all(frequency_cut):
                        continue
                    ax.axvspan(ROI[0], ROI[1], alpha=0.5,label='Frequency ROI = %s'%str(ROI), color = ROI_cm(ROI_index/len(freq_ROI)))

                for ROI_index, ROI in enumerate(ignore_range):
                    frequency_cut = numpy.logical_and(frequencies/1e6 >= ROI[0], frequencies/1e6 <= ROI[1])
                    #ax.axvspan(ROI[0], ROI[1], alpha=0.8,hatch = 'x',label='Frequency IGNORED = %s'%str(ROI), color = 'gray')
                    ax.axvspan(ROI[0], ROI[1], alpha=0.8,hatch = 'x', color = 'gray')

                #Plot average spectra
                plt.plot(frequencies/1e6,average_power,label='Averaged Spectrum')
                
                #Pretty up plot
                plt.grid(which='both', axis='both')
                ax = plt.gca()
                ax.minorticks_on()
                ax.grid(b=True, which='major', color='k', linestyle='-')
                ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.ylabel('dBm')
                plt.xlabel('Frequency (MHz)')
                plt.title('Average Spectum Over %0.2fh Run'%((utc_timestamps[-1]-utc_timestamps[0])/3600.0))
                plt.scatter(frequencies[peaks]/1e6,average_power[peaks],c='r',s=20,label='Identified Peaks')
                plt.legend(loc='upper right')


            for sweeps_per_bin in all_binnings:

                if sweeps_per_bin is not None:
                    power_data = file['power'][...]

                    binned_power = numpy.zeros((numpy.ceil(numpy.shape(power_data)[0] / sweeps_per_bin).astype(int),numpy.shape(power_data)[1]))
                    binned_times = numpy.zeros((numpy.ceil(numpy.shape(power_data)[0] / sweeps_per_bin).astype(int)))

                    starts = file['utc_start_stamp'][:,0]
                    stops = file['utc_stop_stamp'][:,0]

                    for index in range((numpy.ceil(numpy.shape(power_data)[0] / sweeps_per_bin).astype(int))):
                        binned_power[index] = 10*numpy.log10(numpy.mean(10**(power_data[index*sweeps_per_bin:min((index+1)*sweeps_per_bin,len(stops))]/10),axis=0)) 
                        binned_times[index] = ((stops[min((index+1)*sweeps_per_bin - 1,len(stops)-1)] + stops[(index)*sweeps_per_bin])/2.0 - file['utc_start_stamp'][0])/3600.
                else:
                    binned_power = file['power'][...]
                    binned_times = ((file['utc_stop_stamp'][:,0] + file['utc_start_stamp'][:,0])/2.0 - file['utc_start_stamp'][0])/3600.

                if plot_stacked_spectra:
                    #Plot All Spectra
                    plt.figure()
                    ax = plt.gca()
                    
                    #Plot regions of interest
                    ROI_cm = plt.get_cmap('gist_rainbow')
                    for ROI_index, ROI in enumerate(freq_ROI):
                        frequency_cut = numpy.logical_and(frequencies/1e6 >= ROI[0], frequencies/1e6 <= ROI[1])
                        if numpy.all(frequency_cut):
                            continue
                        ax.axvspan(ROI[0], ROI[1], alpha=0.5,label='Frequency ROI = %s'%str(ROI), color = ROI_cm(ROI_index/len(freq_ROI)))

                    for ROI_index, ROI in enumerate(ignore_range):
                        frequency_cut = numpy.logical_and(frequencies/1e6 >= ROI[0], frequencies/1e6 <= ROI[1])
                        #ax.axvspan(ROI[0], ROI[1], alpha=0.8,hatch = 'x',label='Frequency IGNORED = %s'%str(ROI), color = 'gray')
                        ax.axvspan(ROI[0], ROI[1], alpha=0.8,hatch = 'x', color = 'gray')
                    


                    #Plot average sepectra
                    cm = plt.get_cmap('gist_rainbow')
                    for index, row in enumerate(binned_power):
                        plt.plot(frequencies/1e6,row,alpha=0.5,color=cm(index/len(binned_times)))
                    
                    #Pretty up plot
                    plt.grid(which='both', axis='both')
                    ax = plt.gca()
                    ax.minorticks_on()
                    ax.grid(b=True, which='major', color='k', linestyle='-')
                    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.ylabel('dBm')
                    plt.xlabel('Frequency (MHz)')



                if plot_comparison:
                    #PLot the mean in ROI as a function of time. 
                    ROI_cm = plt.get_cmap('gist_rainbow')

                    plt.figure()
                    plt.subplot(2,1,1)
                    ax = plt.gca()

                    #x = [datetime.datetime.strptime(datetime.fromtimestamp(d).replace(tzinfo=pytz.utc).astimezone(pytz.timezone('America/Chicago')),'%m/%d/%Y').date() for d in times]
                    for ROI_index, ROI in enumerate(freq_ROI):
                        frequency_cut = numpy.logical_and(frequencies/1e6 >= ROI[0], frequencies/1e6 <= ROI[1])
                        frequency_cut = numpy.logical_and(frequency_cut,ignore_cut)
                        average_power = 10*numpy.log10(numpy.mean(10**(binned_power[:,frequency_cut]/10),axis=1))
                        
                        if ignore_min_max == True:
                            min_max_cut = ~numpy.isin(average_power, [min(average_power),max(average_power)])
                            average_power = average_power[min_max_cut]
                            x = binned_times[min_max_cut]
                        else:
                            x = binned_times

                        if plot_norm:
                            normalized_power = (average_power - min(average_power))/(max(average_power) - min(average_power)) #+ ROI_index
                            plt.plot(x, normalized_power,label='Frequency ROI = %s'%str(ROI), alpha=0.8, color = ROI_cm(ROI_index/len(freq_ROI)),linewidth=lw) 
                            plt.ylabel('Average Power in ROI (Normalized in dB Units For Each Curve)')
                        else:
                            plt.plot(x, average_power,label='Frequency ROI = %s'%str(ROI), alpha=0.8, color = ROI_cm(ROI_index/len(freq_ROI)),linewidth=lw) 
                            plt.ylabel('Average Power in ROI (dBm)')

                    #Pretty up plot
                    plt.grid(which='both', axis='both')
                    ax = plt.gca()
                    ax.minorticks_on()
                    ax.grid(b=True, which='major', color='k', linestyle='-')
                    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.xlabel('Elapsed Time Since Beginning of Run (Hours)')
                    plt.legend(loc='upper right')

                    plt.subplot(2,1,2,sharex = ax)
                    ax = plt.gca()
                    
                    #Make Landscape
                    ax.axhspan(0, 90, alpha=0.2,label='Sky', color = 'blue')
                    ax.axhspan(-90, 0, alpha=0.2,label='Ground', color = 'green')
                    day_cut = (sun_alt > 0)[:,0]
                    interest_points = numpy.where(numpy.diff(day_cut.astype(int)) != 0)[0]
                    #numpy.where(day_cut == 0)[0][0]
                    #ax.axvspan(0, 90, alpha=0.2,label='Sky', color = 'blue')


                    plt.plot(utc_timestamps/3600.0, sagA_alt, c='r',label='Sgr A*',linewidth=lw)
                    plt.plot(utc_timestamps/3600.0, sun_alt, c='k',label='Sun',linewidth=lw)

                    plt.xlabel('Hours from Start of Run')
                    plt.ylabel('Altitutude (Degrees)')
                    plt.grid(which='both', axis='both')
                    ax.minorticks_on()
                    ax.grid(b=True, which='major', color='k', linestyle='-')
                    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.xlabel('Elapsed Time Since Beginning of Run (Hours)')
                    plt.legend(loc='upper right')
                    #plt.ylim((min(sagA_alt.degree)-5,max(sagA_alt.degree)+5))
                    plt.ylim((-90,90))

                if plot_animated:
                    #Plot All Spectra
                    fig, ax = plt.subplots()
                    #Plot regions of interest
                    ROI_cm = plt.get_cmap('gist_rainbow')

                    for ROI_index, ROI in enumerate(freq_ROI):
                        frequency_cut = numpy.logical_and(frequencies/1e6 >= ROI[0], frequencies/1e6 <= ROI[1])
                        if numpy.all(frequency_cut):
                            continue
                        ax.axvspan(ROI[0], ROI[1], alpha=0.5,label='Frequency ROI = %s'%str(ROI), color = ROI_cm(ROI_index/len(freq_ROI)))

                    for ROI_index, ROI in enumerate(ignore_range):
                        frequency_cut = numpy.logical_and(frequencies/1e6 >= ROI[0], frequencies/1e6 <= ROI[1])
                        #ax.axvspan(ROI[0], ROI[1], alpha=0.8,hatch = 'x',label='Frequency IGNORED = %s'%str(ROI), color = 'gray')
                        ax.axvspan(ROI[0], ROI[1], alpha=0.8,hatch = 'x', color = 'gray')
                    


                    #Pretty up plot
                    plt.grid(which='both', axis='both')
                    ax = plt.gca()
                    ax.minorticks_on()
                    ax.grid(b=True, which='major', color='k', linestyle='-')
                    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.ylabel('dBm')
                    plt.xlabel('Frequency (MHz)')
                    cm = plt.get_cmap('gist_rainbow')
                    line, = ax.plot(frequencies/1e6,binned_power[0],color='k',label='%.2f h Elapsed'%binned_times[0])
                    leg = plt.legend(loc='upper right')
                    def update(row):
                        line.set_ydata(binned_power[row])
                        leg.get_texts()[0].set_text('%.2f h Elapsed'%binned_times[row])
                        return line, ax

                    anim = matplotlib.animation.FuncAnimation(fig, update, frames=numpy.arange(len(binned_times)), interval=10*1000 / len(binned_times))

            file.close() #Open whenever writing to it
        except Exception as e:
            print('Error in main() while file is open.  Closing file.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            file.close()
