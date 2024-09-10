import numpy as np
import matplotlib.pyplot as plt
from dsp_utils import *
import pandas as pd
import seaborn as sns

def spectralDistortionBin(f, freqRange, y, y_hat):
    if len( y.shape ) == 1:
        y = np.expand_dims(y, 0)
        y_hat = np.expand_dims(y_hat, 0)
    sd = []
    freqBinIdxs = []
    for i in range( 1, len( freqRange ) ):
        freqBinIdxs.append( np.where( (f < freqRange[i]) * (f >= freqRange[i - 1]) ) )
    for ind in freqBinIdxs:
        sd.append( spectralDistortion(y[:, np.squeeze(ind)], y_hat[:, np.squeeze(ind)]) )
    return sd

def plotRangeDistribution(f, freq_range, y, y_hats, labels, models):
    SDs = []
    d = dict()
    for i in range( len(labels) ):
        d[i] = labels[i]
    for i in range( len(y_hats) ):
        SD = spectralDistortion(y, y_hats[i])
        print('Mean SD ', models[i], ': ', np.mean(SD))
        SD_bins = spectralDistortionBin(f, freq_range, y, y_hats[i])
        SD_bins.append(SD)
        SDdf = pd.DataFrame( np.array( SD_bins ).transpose() )
        SDdf['dataset'] = [models[i]] * y_hats[i].shape[0]
        SDs.append(SDdf)
    df = pd.concat( SDs )
    df = df.rename( columns = d )
    df = df.melt( id_vars = ['dataset'], value_vars = labels)
    plt.figure( figsize = (12, 7) )
    sns.boxplot(x = 'variable', y = 'value', data = df, hue = 'dataset')
    plt.ylabel('Amplitude (dB)', fontsize = 18)
    plt.xlabel('Frequency (Hz)', fontsize = 18)
    plt.grid()
    plt.savefig("plot_freq_ranges.svg")
    plt.show()