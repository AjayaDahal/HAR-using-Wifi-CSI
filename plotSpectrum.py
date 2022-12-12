'''
Author: Ajaya Dahal, ad2323@msstate.edu
Project: HAR using Wifi based technique compared with radar
'''

import time
import importlib
import config
import numpy as np
from scipy.fftpack import fft, ifft,fftfreq,fftshift
from plotters.AmpPhaPlotter import Plotter # Amplitude and Phase plotter
decoder = importlib.import_module(f'decoders.{config.decoder}') # This is also an import
from matplotlib import pyplot as plt
def string_is_int(s):
    '''
    Check if a string is an integer
    '''
    try:
        int(s)
        return True
    except ValueError:
        return False


my_samples = []


if __name__ == "__main__":
    pcap_filename = 'Mobile_Walking'#input('Pcap file name: ')

    if '.pcap' not in pcap_filename:
        pcap_filename += '.pcap'
    pcap_filepath = '/'.join([config.pcap_fileroot, pcap_filename])

    try:
        samples = decoder.read_pcap(pcap_filepath)
       
    except FileNotFoundError:
        print(f'File {pcap_filepath} not found.')
        exit(-1)

    try:
        i=0
        while (i<samples.nsamples):
            csi = samples.get_csi(
                        i,
                        config.remove_null_subcarriers,
                        config.remove_pilot_subcarriers
                    )
            my_samples.append(csi)
            i+=1
    except:
        print("Problem")        
  
    ######################################
    #regular fft plot
    my_samps = np.asarray(my_samples)
    my_index_arr = np.zeros(my_samps.shape[0], dtype=np.complex_)
    for i in range(0, len(my_samps)-1):
        my_index_arr[i] = my_samps[i][42]
    
    my_fft = np.fft.fft(my_index_arr, int(512))
    ps = np.fft.fftshift(abs(my_fft))
    ps = ps * (np.conjugate(ps)/512)
    
    #psd = np.reshape(ps, (int(512)*1, 1))
    psd = 20*np.log10(ps)
    x_1D = np.arange(0, len(psd)-(1/512))
    # plt.plot(x_1D,  psd)
    # plt.show()
    
    
    ######################################
    #STFT waterfall plot
    window = 8
    overlap = 4
    fftsize = 256
    ind_num = 12
    segment = int((2*(len(my_samples[ind_num])/window))-1) #### segment number for overlapping
    # segment = int(((len(sample_data_iq)/window)))

    
    y=[]


    for i in range(segment):
        # index1 = i*window 
        # index2 = ((i*window)+window)
        
        index1 = i*window - (i*overlap)
        index2 = ((i*window) - (i*overlap)) + window
        tmp_wave = []
        tmp_wave = np.asarray(my_samples[ind_num][index1:index2,])
        ###########
        #tmp_wave = tmp_wave - np.mean(tmp_wave)
        ###########
        temp = fftshift(fft(tmp_wave,n=fftsize))
        
        temp = np.abs(temp)
        y.append(temp)

    y=np.array(y)
    y=np.transpose(y)
    # y_db1 = 20*np.log10(y/np.amax(y))
    # plt.figure()
    # plt.pcolormesh(y_db1,cmap='jet',vmax=0,vmin=-50)
    #y_db = 20*np.log10(y/np.amax(y))
    plt.figure()
    plt.imshow(y, origin='lower', cmap='jet',  aspect='auto',vmax=np.max(y),vmin=np.min(y))
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Human activity recognition using CSI')
    plt.colorbar(label='')
    plt.show()
