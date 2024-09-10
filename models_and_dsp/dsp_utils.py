import numpy as np
import control as ct
import scipy.signal as sig
import pyfar as pf

def spectralDistortion(y, y_hat):
    '''
    Computing full mean (over frequencies and subjects) between
    the two (group of) signals passed
    '''
    sd = 0
    if len( y.shape ) == 1:
        y = np.expand_dims(y, 0)
        y_hat = np.expand_dims(y_hat, 0)
    sd = np.sqrt( np.sum( (y - y_hat)**2, axis = 1 ) / y.shape[1] )
    return sd

def pow2db(x):
    return 10 * np.log( x )

def sig2spec(x, side):
    '''
    Take a time signal in input and return its complex spectrum
    x: input signal/s, can be a matrix
    side: char, either 's' or 'd'
        - s: single side spectrum
        - d: double side spectrum in ''DC - positive frequencies - negative frequencies'' format
    '''
    y = np.fft.fft(x)
    if side == 's':
        if len( x.shape ) == 1:
            y = y[ 0 : x.shape[0]//2 ]
        else:
            y = y[ : , 0 : x.shape[1]//2 ]
    return np.squeeze( y )

def spec2mag(x, side):
    '''
    Take a frequency spectrum in input and returns its magnitude
    '''
    if side == 's':
        if len( x.shape ) == 1:
            y = np.abs( x / x.shape[0] )
            y[0] = y[0] / 2
        else:
            y = np.abs( x / x.shape[1] )
            y[:, 0] = y[:, 0] / 2
    else:
        if len( x.shape ) == 1:
            y = np.abs( x / x.shape[0] )
        else:
            y = np.abs( x / x.shape[1] )
    y = np.abs(x)
    return np.squeeze( y )

def sig2mag(x, side):
    '''
    Take a time signal in input and return its magnitude spectrum
    x: input signal/s, can be a matrix
    side: char, either 's' or 'd'
        - s: single side spectrum
        - d: double side spectrum in ''DC - positive frequencies - negative frequencies'' format
    '''
    y = sig2spec(x, side)
    y_mag = spec2mag(y, side)
    return y_mag
      
def RMS(x):
    '''
    Compute RMS of the signal/s passed
    '''
    if len( x.shape ) == 1:
        return np.sqrt( np.mean( np.square( x ) ) )
    else:
        return np.sqrt( np.mean( np.square( x ), 1 ) )

def RMSNormalization(x):
    '''
    Normalize all the signals in x by the RMS of the signal
    which has the maximum RMS in x itself
    '''
    max_rms = RMS(x).max()
    return (x / max_rms)

def minimumPhaseSpectrum(magnitude_spectrum, side):
    '''
    Reconstruct minimum phase spectrum from magnitude spectrum using real cepstrum
    From:
    https://it.mathworks.com/help/signal/ref/rceps.html#mw_55050d5f-703b-4d9c-a574-3b23056a721e
    magnitude_spectrum: magnitude spectrum
    side: either 's' or 'd'
        - s: magnitude_spectrum is a single side magnitude spectrum
        - d: magnitude_spectrum is a double side magniude spectrum
    '''
    if len( magnitude_spectrum.shape ) == 1:
        magnitude_spectrum = np.expand_dims(magnitude_spectrum, 0)
    if side == 's':
        #magnitude_spectrum = magnitude_spectrum * magnitude_spectrum.shape[1]
        # np.expand_dims(magnitude_spectrum[:, 0], 1) moves the 0 frequency at the center of the double sided spectrum
        magnitude_spectrum_d = np.concatenate( [magnitude_spectrum, np.expand_dims(magnitude_spectrum[:, 0], 1)*2, magnitude_spectrum[: , ::-1][:, 0 : -1]], 1)
    else:
        magnitude_spectrum_d = magnitude_spectrum
    
    real_cepstrum = np.real( np.fft.ifft( np.log( magnitude_spectrum_d ) ) ) 
    
    #Appropriate windowing in the cepstral domain forms the reconstructed minimum-phase signal:
    # repmat([1; 2*ones((nRows+odd)/2-1,1); ones(1-odd,1); zeros((nRows+odd)/2-1,1)],1,nCols);
    cols, rows = magnitude_spectrum_d.shape
    odd = np.remainder(rows, 2)
    w = np.concatenate(([1], 2*np.ones((rows + odd)//2-1), np.ones(1-odd), np.zeros((rows + odd)//2-1)))
    wn = np.tile(w , (cols, 1) )
    y = np.real( np.fft.ifft( np.exp( np.fft.fft( np.multiply(wn, real_cepstrum) ) ) ) )
    y_spec = sig2spec(y, 's')
    
    return y_spec

def fractionalOctaveSmoothing(one_sided_spectrum, f_axis, nth):
    '''
    From:
    https://it.mathworks.com/matlabcentral/fileexchange/55161-1-n-octave-smoothing
    '''
    
    if len(one_sided_spectrum.shape) == 1:
        one_sided_spectrum = np.expand_dims(one_sided_spectrum, 0)
    
    def gauss_f(f_axis, f, nth):
        sigma = (f / nth) / np.pi; # standard deviation
        g = np.exp( -((f_axis - f)**2) / (2 * (sigma**2)) ); # Gaussian
        g = g / np.sum(g) # normalize magnitude
        return g
    
    x_oct = np.copy(one_sided_spectrum)
    if nth > 0:
        for i in range(np.argwhere(f_axis == 0)[0][0] + 1, f_axis.shape[0]):
            g = gauss_f(f_axis, f_axis[i], nth)
            x_oct[:, i] = np.sum( np.multiply( np.tile(g, (one_sided_spectrum.shape[0] ,1)) , one_sided_spectrum[:, :] ), 1 )
        for i in range(one_sided_spectrum.shape[0]):
            if np.all(one_sided_spectrum[i, :] >= 0):
                x_oct[i, x_oct[i, :] < 0] = 0
    
    return np.squeeze( x_oct )

def inverseLMSRegularized(one_sided_spectrum, low_freq, high_freq, f, sr):
    '''
    Least Mean Square Regularized Inversion from:
    Automatic Regularization Parameter for Headphone Transfer Function Inversion - Bolanos et al.
    '''
    # sigma diverso per ogni soggetto/riga della matrice passata in input (se viene passata una matrice)
    # sigma
    minimum_phase = minimumPhaseSpectrum(one_sided_spectrum, 's')
    one_sided_smoothed = ct.db2mag( fractionalOctaveSmoothing( ct.mag2db( one_sided_spectrum ), f, 2) )
    sigma = np.subtract( np.abs(one_sided_spectrum) , np.abs(one_sided_smoothed) )
    sigma[sigma > 0] = 0
    
    # alpha
    highpass_n, highpass_d = sig.butter(4, low_freq, 'highpass', fs = sr)
    _, highpass_tf = sig.freqz(highpass_n, highpass_d, worN = f.shape[0], fs = sr)
    lowpass_n, lowpass_d = sig.butter(2, high_freq, 'lowpass', fs = sr)
    _, lowpass_tf = sig.freqz(lowpass_n, lowpass_d, worN = f.shape[0], fs = sr)
    
    bandpass_tf = highpass_tf * lowpass_tf
    alpha = (1 - np.abs(bandpass_tf)**2) / np.abs(bandpass_tf)**2 # (1 / |W|**2) - 1
    
    # beta
    # fare un np.tile su alpha per rendderlo: soggetti X bin frequenza
    if len(sigma.shape) > 1:
        alpha = np.tile(alpha, (sigma.shape[0], 1))
    beta = alpha + (sigma**2)
    
    # inverse
    inverse = np.divide( np.conjugate(minimum_phase) , np.add( (one_sided_spectrum)**2 , beta ) )
    
    return inverse, alpha, sigma

def peakError(spectrum, f, threshold, low_freq, high_freq):
    '''
    Peak error computation, from:
    COLORATION METRICS FOR HEADPHONE EQUALIZATION - Geronazzo, Boren, brinkmann, Choueiri
    '''
    if len( spectrum.shape ) == 1:
        spectrum = np.expand_dims(spectrum, 0)
    mean_spec = np.abs( np.mean( spectrum[ : , np.logical_and(f>200, f<400) ], 1 ) )
    spectrum = np.divide( spectrum, np.transpose( np.tile(mean_spec, (spectrum.shape[1], 1)) ) )
    #spectrum = spectrum / np.abs( np.mean( spectrum[ np.logical_and(f>200, f<400) ] ) )

    pow_spec = pow2db( spec2mag(spectrum, 's')**2 ) # power spectrum
    fine = fractionalOctaveSmoothing(pow_spec, f, 4)
    coarse = fractionalOctaveSmoothing(pow_spec, f, 1)
    diff_peaks = fine - coarse
    diff_notches = coarse - fine
    
    def findPeaks(diff, f, threshold, low_freq, high_freq):
        '''
        find peaks in a given spectrum in acertain frequency range delimited by
        low_ferq and high_freq
        copmute peak error
        return also frequencies at which peaks are found
        '''
        acceptable_freqs = np.logical_and(f >= low_freq, f <= high_freq)
        grad = np.gradient(diff, axis = -1)
        grad_sign = np.sign(grad)
        signchange = ((np.roll(grad_sign, 1) - grad_sign) != 0).astype(int).astype(bool) # detect sign changes in the gradient
        signchange[0] = False
        signchange[diff < threshold] = False # ignore changes in sign if corresponding power in power spectrum is less then 'threshold' dB
        signchange = np.logical_and(signchange, acceptable_freqs)
    
        # ignore peaks if in the range of 1/6th of octave there is a greater peak
        signchange_copy = np.copy(signchange)
        for i in range( len(np.argwhere(signchange)) ):
            freq = f[ np.argwhere(signchange_copy)[i] ] # current frequency
            freq_range = freq / 6 # 1/6th octave
            range_freq_idx =  np.logical_and( (f < freq + freq_range), (f > freq - freq_range) ) # frequency range of the search
            range_peak_values = diff[ np.logical_and( signchange, range_freq_idx ) ] # values of other peaks in the range
            freq_peak_value = diff[ np.argwhere(signchange_copy)[i] ] # peak of current frequency
            if np.any( np.less( freq_peak_value, range_peak_values ) ): # check if other peaks in range are greater then current one
                signchange[ np.argwhere(signchange_copy)[i] ] = False
    
        peaks = diff[signchange]
        peaks_freq = f[signchange]
        #peaks_idx = np.where(signchange)
        peak_error = np.sum(peaks) / (3 * np.log2( f[-1] / f[1] ))
        #if low_freq == 0:
        #    peak_error = np.sum(peaks) / (3 * np.log2( high_freq / f[1] ))
        #else:
        #    peak_error = np.sum(peaks) / (3 * np.log2( high_freq / low_freq ))
        
        
        return peak_error, peaks, peaks_freq
    
    peak_error = []
    peaks = []
    peaks_freq = []
    notch_error = []
    notches = []
    notches_freq = []
    for i in range(spectrum.shape[0]):
        p_err, p, p_freq = findPeaks(diff_peaks[i, :], f, threshold, low_freq, high_freq)
        n_err, n, n_freq = findPeaks(diff_notches[i, :], f, threshold, low_freq, high_freq)
        peak_error.append(p_err); peaks.append(p); peaks_freq.append(p_freq)
        notch_error.append(n_err); notches.append(-n); notches_freq.append(n_freq)
    
    #return peak_error, peaks, peaks_freq, pow_spec, diff
    return np.array(peak_error), np.array(notch_error), peaks, notches, peaks_freq, notches_freq, pow_spec, diff_peaks

def ERBError(spectrum, f, fs, low_freq, high_freq):
    if len( spectrum.shape ) == 1:
        spectrum = np.expand_dims(spectrum, 0)
        
    GFB = pf.dsp.filter.GammatoneBands([low_freq, high_freq])
    
    highpass_n, highpass_d = sig.butter(4, low_freq, 'highpass', fs = fs)
    _, highpass_tf = sig.freqz(highpass_n, highpass_d, worN = f.shape[0], fs = fs)
    lowpass_n, lowpass_d = sig.butter(2, high_freq, 'lowpass', fs = fs)
    _, lowpass_tf = sig.freqz(lowpass_n, lowpass_d, worN = f.shape[0], fs = fs)
    bandpass_tf = highpass_tf * lowpass_tf
    bandpass_tf = np.tile(bandpass_tf, (spectrum.shape[0], 1))
    
    fil = []
    for i in range(GFB.frequencies.shape[0]):
        b, a = sig.gammatone(GFB.frequencies[i], 'fir', fs = fs)
        _, fil_tf =  sig.freqz(b, a, worN = f.shape[0], fs = fs)
        fil.append( 20*ct.mag2db(fil_tf) )
    fil = np.array(fil)
    
    erb_sum = np.zeros( (spectrum.shape[0],) )
    for i in range(GFB.frequencies.shape[0]):
        erbeq = np.sum( (np.abs(spectrum)**2) * np.tile(fil[i], (spectrum.shape[0], 1)), 1)
        erbband = np.sum( (np.abs(bandpass_tf)**2) * np.tile(fil[i], (spectrum.shape[0], 1)), 1)
        erb_sum = np.add( np.abs( 10*np.log10( np.divide(erbeq, erbband) ) ), erb_sum)
    
    return erb_sum / GFB.frequencies.shape[0]

def broadbandError(erb_err, peak_err, notch_err):
    return erb_err - ( (peak_err + notch_err) / 2 )

def peaks_notches_match(seq_test_1, seq_predict_1, seq_test_2, seq_predict_2):
    match_count = 0
    missmatch_count = 0
    for i in range(len(seq_test_1)):
        for j in range(len(seq_test_1[i])):
            if seq_test_1[i][j] in seq_predict_1[i]:
                match_count += 1
            else:
                missmatch_count += 1
                
    for i in range(len(seq_test_2)):
        if len(seq_test_2[i]) != 0 and len(seq_predict_2[i]):
            for j in range(len(seq_test_2[i])):
                if seq_test_2[i][j] in seq_predict_2[i]:
                    match_count += 1
                else:
                    missmatch_count += 1
        else:
            continue
                
    tot = match_count + missmatch_count
    #print(seq1)
    #print(seq2)
    return ( match_count / tot ) * 100 # return match percentage