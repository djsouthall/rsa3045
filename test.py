'''
This is my test script for working with the RIGOL 3045.
'''
import visa

def getSA():
    '''
    Returns the instrument object that can be used to send commands to the spectrum analyzer.]

    Returns
    -------
    inst : pyvisa.resources.usb.USBInstrument
        The instrument as connected to by using pyvisa.
    '''
    rm = visa.ResourceManager()
    rsrc = rm.list_resources()
    inst = rm.open_resource(rsrc[0])
    return inst

def setUpSA(inst, freq_L=0, freq_R=200e6, sweep_time=10, sweep_points=3000, reference_level=-20, frontend_attenuation=0, rbw=1000, vbw=1000):
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
    vbw
        Sets the display video BW of the SA.  This will not effect the apparent location
        of the noise floor, but should smooth out features to look less noisy.  Given in Hz.
        Default Value = 1000
    '''
    #Set range of measurement
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

    #Calibrate the system: System->Alignment->Align Now
    print('Executing - :CALibration:ALL')
    val = inst.write(':CALibration:ALL')
    print('\t',val)

    #Set reference level
    print('Executing - :DISPlay:WINDow:TRACe:Y[:SCALe]:RLevel %i'%reference_level)
    val = inst.write(':DISPlay:WINDow:TRACe:Y[:SCALe]:RLevel %i'%reference_level)
    print('\t',val)


if __name__ == '__main__':
    #PARAMETERS
    #----------
    freq_L = 0 #Hz The lower edge of the measurement.
    freq_R = 200e6 #Hz The upper edge of the measurement.
    sweep_time = 10 #s How long the sweep time is for the measurement.
    sweep_points = 3000 # Number of points in the sweep.
    reference_level = -20 #dBm This will depend on the measurement and noise floor (effected by rbw)
    frontend_attenuation = 0 #dBm Sets the front end attenuation of the SA
    rbw = 1000 #Hz  Sets the capturing resolution BW of the SA.  Should be ~<= the size of features you
               #    wish to be able to see seperately.  This will effect the apparent noise floor. 
    vbw = 1000 #Hz   Sets the display video BW of the SA.  This will not effect the apparent location
               #     of the noise floor, but should smooth out features to look less noisy.


    #Prep for measurement
    inst = getSA()
    setUpSA(inst, freq_L=freq_L, freq_R=freq_R, sweep_time=sweep_time, sweep_points=sweep_points, reference_level=reference_level, frontend_attenuation=frontend_attenuation, rbw=rbw, vbw=vbw)

    #Perform Measurement

    #PSEUDOCODE

    #SET SINGLE MEASUREMENT

    #GET TIME START TIME STAMP

    #START SINGLE MEASUREMENT

    #GET CHECK WHEN SINGLE MEASUREMENT COMPLETE

    #GET TIME STOP TIME STAMP

    #READ OUT MEASDATA TRACE FROM SA (ASCII?)

    #SAVE SA TRACE AND WRITE TO DISK ON PC WITH TIME STAMPS


