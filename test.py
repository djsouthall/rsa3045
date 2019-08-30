'''
This is my test script for working with the RIGOL 3045.
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

plt.ion()

def getSA():
    '''
    Returns the instrument object that can be used to send commands to the spectrum analyzer.]

    Returns
    -------
    inst : pyvisa.resources.usb.USBInstrument
        The instrument as connected to by using pyvisa.
    '''
    try:
        rm = visa.ResourceManager()
        rsrc = rm.list_resources()
        inst = rm.open_resource(rsrc[0])
        return inst
    except Exception as e:
        print('Error in getSA().')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def calibrateSA(inst, wait=10):
    '''
    This will run the self calibration on the oscilloscope, and then wait for an amount of time 
    to ensure that it has completed, before returning.

    Parameters
    ----------
    inst : pyvisa.resources.usb.USBInstrument
        The instrument as connected to by using pyvisa.
    wait : int
        How long the function will wait in seconds for the calibration to run.
    '''
    #Calibrate the system: System->Alignment->Align Now
    try:
        print('Executing - :CALibration:ALL')
        val = inst.write(':CALibration:ALL')
        print('\t',val)
        time.sleep(wait)
    except Exception as e:
        print('Error in calibrateSA().')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)



def setUpSA(inst, freq_L=0, freq_R=200e6, sweep_time=10, sweep_points=3000, reference_level=-20, frontend_attenuation=0, rbw=1000, vbw=1000, calibrate=False, single=True):
    '''
    This function will prepare the SA (spectrum analyzer) for a measurement.  This currently
    assumes the SA is already in GPSA Swept SA mode.

    Parameters
    ----------
    inst : pyvisa.resources.usb.USBInstrument
        The instrument as connected to by using pyvisa.
    freq_L : int
        The lower edge of the measurement.  Given in Hz.  Default Value = 0
    freq_R : int
        The upper edge of the measurement.  Given in Hz.  Default Value = 200e6
    sweep_time : int
        How long the sweep time is for the measurement.  Given in seconds.  Default Value = 10
    sweep_points : int
        Number of points in the sweep.
        Default Value = 3000
    reference_level : int
        This will depend on the measurement and noise floor (effected by rbw).  Given in dBm.
        Default Value = -20
    frontend_attenuation : int
        Sets the front end attenuation of the SA.  Given in dBm.  Default Value = 0
    rbw : int
        Sets the capturing resolution BW (RBW) of the SA.  Should be ~<= the size of features you
        wish to be able to see seperately.  This will effect the apparent noise floor.  Given in Hz.
        Default Value = 1000
    vbw : int
        Sets the display video BW of the SA.  This will not effect the apparent location
        of the noise floor, but should smooth out features to look less noisy.  Given in Hz.
        Default Value = 1000
    '''
    #Set range of measurement
    try:
        print('Executing - [:SENSe]:FREQuency:START %i'%freq_L)
        val = inst.write('[:SENSe]:FREQuency:START %i'%freq_L)
        print('\t',val)

        print('Executing - [:SENSe]:FREQuency:STOP %i'%freq_R)
        val = inst.write('[:SENSe]:FREQuency:STOP %i'%freq_R)
        print('\t',val)

        #Set Attn to 0 dB
        print('Executing - :SENSe:POWer:RF:ATTenuation %i'%frontend_attenuation)
        val = inst.write(':SENSe:POWer:RF:ATTenuation %i'%frontend_attenuation)
        print('\t',val)

        #Set RBW to 10 KHz
        print('Executing - :SENSe:BANDwidth:RESolution %i'%rbw)
        val = inst.write(':SENSe:BANDwidth:RESolution %i'%rbw)
        print('\t',val)

        #Set VBW to 10 KHz
        print('Executing - :SENSe:BANDwidth:VIDeo %i'%vbw)
        val = inst.write(':SENSe:BANDwidth:VIDeo %i'%vbw)
        print('\t',val)

        #Set Swept Points to 3000
        print('Executing - [:SENSe]:SWEep:POINts %i'%sweep_points)
        val = inst.write('[:SENSe]:SWEep:POINts %i'%sweep_points)
        print('\t',val)

        #Set time rule to Accy
        print('Executing - :SENSe:SWEep:TIME:AUTO:RULes ACCuracy')
        val = inst.write(':SENSe:SWEep:TIME:AUTO:RULes ACCuracy')
        print('\t',val)

        #Set Time 300s NOT QUITE RIGHT I THINK
        print('Executing - [:SENSe]:SWEep:TIME  %i'%sweep_time)
        val = inst.write('[:SENSe]:SWEep:TIME  %i'%sweep_time)
        print('\t',val)

        #Set reference level
        print('Executing - :DISPlay:WINDow:TRACe:Y[:SCALe]:RLevel %i'%reference_level)
        val = inst.write(':DISPlay:WINDow:TRACe:Y[:SCALe]:RLevel %i'%reference_level)
        print('\t',val)

        #Set to single measurement mode (Rather than continuous)
        if single == True:
            print('Executing - :INITiate:CONTinuous OFF')
            val = inst.write(':INITiate:CONTinuous OFF')
            print('\t',val)
        else:
            print('Executing - :INITiate:CONTinuous ON')
            val = inst.write(':INITiate:CONTinuous ON')
            print('\t',val)
    except Exception as e:
        print('Error in setUpSA().')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == '__main__':
    script_start_time = datetime.utcnow()
    script_start_time = script_start_time.replace(tzinfo=pytz.utc) #N
    #PARAMETERS
    #----------
    freq_L = 0 #Hz The lower edge of the measurement.
    freq_R = 150e6 #Hz The upper edge of the measurement.
    sweep_time = 60 #s How long the sweep time is for the measurement.
    sweep_points = 10001 # Number of points in the sweep.
    sweep_buffer_time = 15 #s How much time ON TOP of sweep_time you want to wait for a sweep to occur.
                          #  I.e. this just gives the waiting for loop more time to ensure seep is complete.
    reference_level = -10 #dBm This will depend on the measurement and noise floor (effected by rbw)
    frontend_attenuation = 20 #dBm Sets the front end attenuation of the SA
    rbw = 1000 #Hz  Sets the capturing resolution BW of the SA.  Should be ~<= the size of features you
               #    wish to be able to see seperately.  This will effect the apparent noise floor. 
    vbw = 1000 #Hz   Sets the display video BW of the SA.  This will not effect the apparent location
               #     of the noise floor, but should smooth out features to look less noisy.
    frequencies = numpy.linspace(freq_L,freq_R,sweep_points) #Hz
    plot_data = False
    output_path = './output/'
    save_data = True
    total_runtime = 24*60*60 #s This is a lower bound.  Can be off by one cycle.  While loop runs until total time is reached and loop is called.

    if save_data == True:
        #PREPARE OUTPUT FILE
        output_file_created = False
        while output_file_created == False:
            try:
                outname = output_path + 'RIGOL_CAPTURE_' + str(script_start_time).replace(':','-').replace(' ','-').replace('.','p') + '.h5'
                with h5py.File(outname, 'w') as outfile:
                    try:
                        outfile.attrs['freq_L'] = freq_L
                        outfile.attrs['freq_R'] = freq_R
                        outfile.attrs['sweep_time'] = sweep_time
                        outfile.attrs['sweep_points'] = sweep_points
                        outfile.attrs['sweep_buffer_time'] = sweep_buffer_time
                        outfile.attrs['reference_level'] = reference_level 
                        outfile.attrs['frontend_attenuation'] = frontend_attenuation
                        outfile.attrs['rbw'] = rbw
                        outfile.attrs['vbw'] = vbw

                        outfile.create_dataset('frequencies', (len(frequencies),), dtype=float, compression='gzip', compression_opts=9)
                        outfile.create_dataset('power', (0,len(frequencies)), dtype=float, maxshape=(None,len(frequencies)))
                        outfile.create_dataset('utc_start_stamp', (0,1), dtype=float, maxshape=(None,1),chunks=True)
                        outfile.create_dataset('utc_stop_stamp', (0,1), dtype=float, maxshape=(None,1),chunks=True)
                        outfile['frequencies'] = frequencies

                        outfile.close() #Open whenever writing to it
                        output_file_created = True
                    except KeyboardInterrupt as e:
                        print(e)
                        print('KeyboardInterrupt detected while file is open.  Closing File.')
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        if 'outfile' in locals() or 'outfile' in globals():
                            outfile.close()
                            print('File closed.  Breaking scipt.')
                            sys.exit()
                        else:
                            print('outfile does not exist to close. ')
            except Exception as e:
                print('Error while trying to create output file.')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


    #Prep for measurement
    inst = getSA()
    setUpSA(inst, freq_L=freq_L, freq_R=freq_R, sweep_time=sweep_time, sweep_points=sweep_points, reference_level=reference_level, frontend_attenuation=frontend_attenuation, rbw=rbw, vbw=vbw)

    calibrateSA(inst)

    #Perform Measurement
    measurement_row = 0
    loop_start_time = time.time()
    while time.time() - loop_start_time < total_runtime:
        try:
            #Initiate Sweep
            start_time = time.time()
            utc_sweep_start_time = datetime.utcnow()
            utc_sweep_start_time = utc_sweep_start_time.replace(tzinfo=pytz.utc) #N

            print('Executing - :INITiate[:IMMediate]')
            val = inst.write(':INITiate[:IMMediate]')
            print('\t',val)
            #Should wait at least the sweep time now
            # setup wait
            wait_width = 40
            sys.stdout.write("[%s]" % (" " * (wait_width+1)))
            sys.stdout.flush()
            sys.stdout.write("\b" * (wait_width+2)) # return to start of line, after '['

            elapsed_time = 0
            while elapsed_time <= sweep_time:
                elapsed_time = time.time() - start_time
                time.sleep((sweep_time)/wait_width)
                sys.stdout.write("-")
                sys.stdout.flush()
            utc_sweep_stop_time = datetime.utcnow()
            utc_sweep_stop_time = utc_sweep_stop_time.replace(tzinfo=pytz.utc) #N
            time.sleep(sweep_buffer_time)

            sys.stdout.write("]\n") # this ends the progress bar
            stop_time = time.time() #SHOULD GET IN UTC

            #READ OUT MEASDATA TRACE FROM SA (ASCII?)

            power_values = numpy.array(inst.query(':FETCh:SANalyzer1?').split(',')).astype(float)[numpy.arange(sweep_points)*2+1] #dBm

            if plot_data == True:
                plt.figure()
                ax = plt.gca()
                plt.plot(frequencies/1e6,power_values)
                plt.grid(which='both', axis='both')
                ax.grid(b=True, which='major', color='k', linestyle='-')
                ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            if save_data == True:
                try:
                    with h5py.File(outname, 'a') as outfile:
                        try:
                            dataset = outfile['power']
                            dataset.resize(measurement_row+1,axis=0)
                            dataset[-1,:] = power_values

                            dataset = outfile['utc_start_stamp']
                            dataset.resize(measurement_row+1,axis=0)
                            dataset[-1] = utc_sweep_start_time.timestamp()

                            dataset = outfile['utc_stop_stamp']
                            dataset.resize(measurement_row+1,axis=0)
                            dataset[-1] = utc_sweep_stop_time.timestamp()

                            outfile.close() #Open whenever writing to it
                        except KeyboardInterrupt as e:
                            print(e)
                            print('KeyboardInterrupt detected while file is open.  Closing File.')
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)
                            if 'outfile' in locals() or 'outfile' in globals():
                                outfile.close()
                                print('File closed.  Breaking scipt.')
                                sys.exit()
                            else:
                                print('outfile does not exist to close. ')
                except Exception as e:
                    print('Error while saveing to output file.')
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

            measurement_row += 1
            print('Elapsed Time = %0.2f s / %0.2f s'%(time.time()-loop_start_time,total_runtime))
            
        except Exception as e:
            print('Error while sweeping.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)