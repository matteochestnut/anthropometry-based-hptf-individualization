from GRNN import GRNN
from RBFNN import RBFNN
from MSVR import MSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from dsp_utils import *

from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import shapiro

import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import pickle

class Pipeline:
    
    def __init__(self, model_type):
        self.hyperparams = None
        self.model_type = model_type
        self.input_preprocessing_type = ''
        self.input_preprocessing_params = []
        self.output_preprocessing_type = ''
        self.output_preprocessing_params = []
        self.model = None
    
    def modelBuild(self, hyperparams, *args):
        self.hyperparams = hyperparams
        if self.model_type == 'kn':
            self.model = KNeighborsRegressor(
                n_neighbors = hyperparams[0],
                metric = hyperparams[1],
                algorithm = 'brute'
            )
        elif self.model_type == 'grnn':
            self.model = GRNN(hyperparams[0])
        elif self.model_type == 'rbfnn':
            # RBFNN(number_units, centers, output_dim, loss = 'custom', gammas = 1.0)
            #units = [x.shape[0]]
            #centers = [ x ]
            #hyperparams_rbfnn = list( itertools.product(units, centers) )
            self.model = RBFNN(hyperparams[0], centers = args[0], output_dim = args[1], loss = args[2], gammas = 1.0)
        elif self.model_type == 'forest':
            self.model = RandomForestRegressor(
                n_estimators = 128,
                max_depth = hyperparams[0],
                n_jobs = -1,
                bootstrap = hyperparams[1]
            )
        elif self.model_type == 'svr':
            self.model = MSVR(
                kernel = hyperparams[0]
            )
    
    def modelTrain(self, x, y):
        if self.model_type == 'kn':
            self.model.fit(x, y)
        elif self.model_type == 'grnn':
            self.model.fit(x, y)
        elif self.model_type == 'rbfnn':
            # keras callbacks
            # model checkpoint
            checkpoint_filepath = 'models/rbfnn/checkpoint.weights.h5'
            model_checkpoint_callback = ModelCheckpoint(
                verbose=0,
                filepath = checkpoint_filepath,
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True)
            # early stopping
            early_stopping_callback = EarlyStopping(
                verbose=0,
                monitor='loss',
                mode='min',
                patience=10,
                start_from_epoch=10)
            self.model.fit( x,
                           y,
                           verbose = 0,
                           batch_size = 1,
                           epochs = 50,
                           callbacks = [model_checkpoint_callback, early_stopping_callback])
        elif self.model_type == 'forest':
            self.model.fit(x, y)
        elif self.model_type == 'svr':
            self.model.fit(x, y)
    
    def modelPredict(self, x):
        if self.model_type == 'kn':
            predictions = self.model.predict(x)
        elif self.model_type == 'grnn':
            predictions = self.model.predict(x)
        elif self.model_type == 'rbfnn':
            predictions = self.model.predict(x, verbose = 0)
        elif self.model_type == 'forest':
            predictions = self.model.predict(x)
        elif self.model_type == 'svr':
            predictions = self.model.predict(x)
        return predictions
    
    def computeInputPreprocessingParams(self, preprocessing_type, x, *args):
        '''
        Compute input preprocessing parameters on the training set x
        when preprocessing_type == 'selection', *args needs to store
        x.shape[0]
        '''
        self.input_preprocessing_type = preprocessing_type
        # z-score
        self.input_preprocessing_params = []
        scaler = StandardScaler()
        scaler.fit( x )
        x_normalized = scaler.transform( x )
        self.input_preprocessing_params.append( scaler ) # 0
        if preprocessing_type == 'none':
            pass
        elif preprocessing_type == 'pca':
            # pca
            pca = PCA(n_components = 0.9, svd_solver = 'full')
            pca.fit( x_normalized )
            self.input_preprocessing_params.append( pca ) # 1
        elif preprocessing_type == 'selection':
            # fantini anthropometry selection
            dataset_length = args[0]
            dataset_indices = args[1]
            y = args[2]
            sd_matrix = SDmatrix( y )
            features = fantini2024(sd_matrix, dataset_indices, x, dataset_length)
            self.input_preprocessing_params.append( features ) # 1
    
    def inputPreprocessing(self, x):
        x_preprocessed = self.input_preprocessing_params[0].transform( x )
        if self.input_preprocessing_type == 'none':
            return x_preprocessed
        elif self.input_preprocessing_type == 'pca':
            return self.input_preprocessing_params[1].transform( x_preprocessed )
        elif self.input_preprocessing_type == 'selection':
            return x[:, self.input_preprocessing_params[1]]
    
    def computeOutputPreprocessingParams(self, preprocessing_type, y):
        self.output_preprocessing_type = preprocessing_type
        self.output_preprocessing_params = []
        if preprocessing_type == 'none':
            pass
        elif preprocessing_type == 'pca':
            # z-score
            scaler = StandardScaler()
            scaler.fit( y )
            y_normalized = scaler.transform( y )
            self.output_preprocessing_params.append( scaler ) # 0
            #pca
            pca = PCA(n_components = 0.9, svd_solver = 'full')
            pca.fit( y_normalized )
            self.output_preprocessing_params.append( pca ) # 1
    
    def outputPreprocessing(self, y):
        if self.output_preprocessing_type == 'none':
            return y
        elif self.output_preprocessing_type == 'pca':
            y_preprocessed = self.output_preprocessing_params[0].transform( y )
            return self.output_preprocessing_params[1].transform( y_preprocessed )
    
    def outputConvert(self, y):
        if self.output_preprocessing_type == 'none':
            return y
        elif self.output_preprocessing_type == 'pca':
            y_convert = self.output_preprocessing_params[1].inverse_transform( y )
            return self.output_preprocessing_params[0].inverse_transform( y_convert )
            
def SDmatrix(hptfs):
    matrix = np.full( (hptfs.shape[0], hptfs.shape[0]), np.nan )
    for i in range( hptfs.shape[0] ):
        for j in range( hptfs.shape[0] ):
            if i == j:
                continue
            matrix[i, j] = spectralDistortion(hptfs[i, :], hptfs[j, :])
    return matrix

def fantini2024(sd_matrix, subjects_idx, anthropometry, dataset_length):
    
    # Number of anthropometric parameters
    n_metrics = anthropometry.shape[1]
    # Minimum spectral distortion error
    min_err = np.inf
    # Initialize selected metrics
    selected_metrics = []
    # Initialize metric combination with all metrics
    metrics_comb = np.arange(0, n_metrics) # 1 x N_FEATURES
    metrics_comb = np.expand_dims(metrics_comb, 0)

    # Iterate over the number of metrics
    for m in range( n_metrics ):

        # Initialize the correlation coefficients for each metric combination
        err = np.zeros((metrics_comb.shape[0], 1))

        # Iterate over the metrics combinations
        for c in range( metrics_comb.shape[0] ):

            # Compute the distance between the anthropometry (select only
            # the current metrics combination) of each subject with any
            # other subject
            metric_dist = squareform(pdist(anthropometry[:, metrics_comb[c,:]]))

            # Set the anthropometric distances between left and right ears
            # of the same subject to NaN in order to ignore them
            # modifica del codice originale in modo che funzioni dopo l'estrazione e lo shuffle
            # dal dataset di esempi causali per fare il test set, scombinando quindi le coppie
            # sx/dx
            
            for i in range(subjects_idx.shape[0]):
                # gli esempi nel dataset originale (prima metà orecchio sx, seconda metà orecchio dx) 
                # prima dello split sono indicinzati con indici da 0 a dataset_length, quando poi viene
                # fatto lo shuffle e lo split (sia per gli indici che per il dataset), si fa un controllo
                # sugli indice del training set:
                # se per un dato indice i (da 0 a dataset_length//2) al'interno degli indici del training set
                # esiste, sempre tra questi indici, i + dataset_length//2, allora significa che per quello
                # esempio nel training set è presente si orecchio sx che dx
                # -> da questi indici "dei soggetti" risalgo agli indici della matrice quindi setto a nan
                # la distanza tra coppie di orecchie sx/dx per uno stesso soggetto
                idx = subjects_idx[i]
                if (idx < dataset_length//2) and np.isin(idx + dataset_length//2, subjects_idx):
                    metric_dist[ i, np.where( subjects_idx == (idx + dataset_length//2) )[0] ] = np.nan
                    metric_dist[ np.where( subjects_idx == (idx + dataset_length//2) )[0], i ] = np.nan

            # diagonale settata a nan
            np.fill_diagonal(metric_dist, np.nan)
            
            # per ogni orecchio trovo il corrispondente per il quale la differenza tra le antropometrie è minore
            min_idx = np.nanargmin(metric_dist, axis=0)

            # tra le due orecchie per le quali la distanza tra le antropometire è minore risalgo alla spectral
            # distortion tra le due, faccio questo per tutti i soggetti e faccio quindi la media
            err[c] = np.nanmean(sd_matrix[np.arange(sd_matrix.shape[0]), min_idx])

        min_err_c = np.min(err)
        min_erro_comb_idx = np.argmin(err)

        if min_err_c < min_err:
            min_err = min_err_c
            selected_metrics = metrics_comb[min_erro_comb_idx,:]

        else:
            # Otherwise stop the search
            break

        # Get the new metrics combinations to evaluate with one metric less
        selected_metrics = np.array(selected_metrics)
        metrics_comb = np.array( list( itertools.combinations( selected_metrics.tolist(), selected_metrics.size - 1 ) ) )

    return selected_metrics

def validateModel(validation_type, model_type, hyperparams_combinations, input_preprocessing_type, output_preprocessing_type, x_train, y_train, subject_indices, dataset_len):
    
    # choose the validation type, leave one out or 5-fold cross validation
    if validation_type == 'loo':
        val = LeaveOneOut()
    elif validation_type == 'kf':
        val = KFold(n_splits = 5)
            
    # examples / folder count
    fldCount = 0
    
    spectral_distortion = np.zeros(len(hyperparams_combinations))
        
    for _, (inner_train_indeces, validation_indeces) in enumerate( val.split( x_train ) ):
        fldCount += 1
        
        # splitting the trainin set from the test/trainig partition into the actual
        # innert training set of the cross validation and the validation set
        inner_subjects_indeces = subject_indices[inner_train_indeces]
        x_inner_train = x_train[inner_train_indeces, :]
        x_validation = x_train[validation_indeces, :]
        y_inner_train = y_train[inner_train_indeces, :]
        y_validation = y_train[validation_indeces, :]
        
        mdlCount = 0
        for hyperparams in hyperparams_combinations:
            model = Pipeline(model_type)
            
            # input preprocessing params computation
            if input_preprocessing_type == 'selection':
                try:
                    model.computeInputPreprocessingParams(input_preprocessing_type, x_inner_train, dataset_len, inner_subjects_indeces, y_inner_train)
                except:
                    print('When anthropometry selection is chosen as input preprocessing the total number of examples in the dataset has to be passed as arguments')
            else:
                model.computeInputPreprocessingParams(input_preprocessing_type, x_inner_train) 
            
            # input preprocessing
            x_train_preprocessed = model.inputPreprocessing( x_inner_train )
            x_validation_preprocessed = model.inputPreprocessing( x_validation )
            
            # output preprocessing params computation
            model.computeOutputPreprocessingParams( output_preprocessing_type, y_inner_train )
            
            # output preprocessing
            y_train_preprocessed = model.outputPreprocessing( y_inner_train )
            
            # build the model
            if output_preprocessing_type == 'pca':
                model.modelBuild(hyperparams, x_train_preprocessed, y_train_preprocessed.shape[1], 'mse')
            else:
                model.modelBuild(hyperparams, x_train_preprocessed, y_train_preprocessed.shape[1], 'custom')
            
            # training the model
            model.modelTrain(x_train_preprocessed, y_train_preprocessed)
            
            # in case of rbfnn, load the best weights saved during traning
            if model_type == 'rbfnn':
                model.model.load_weights('models/rbfnn/checkpoint.weights.h5')
            
            # prediction on validation set
            predictions = model.modelPredict( x_validation_preprocessed )
            
            # in case the model had to predict a preprocess output,
            # invert the processing to obtain the hptf
            predictions_converted = model.outputConvert( predictions )
            
            # compute spectral distortion error
            spectral_distortion[mdlCount] += np.mean( spectralDistortion(predictions_converted, y_validation) , 0)
            mdlCount += 1
        
    # average spectral distortion on the validation folds / examples from loo for each model
    spectral_distortion = spectral_distortion / fldCount
    bstModel = np.argmin(spectral_distortion)
    print('Spectral distortion of the best model on the validations set: ', np.min(spectral_distortion))
    print('Best model parameters: ', hyperparams_combinations[bstModel])
    return hyperparams_combinations[bstModel], spectral_distortion[bstModel]

def saveModel(path, model, model_type, *args):
    if model_type == 'rbfnn':
        with open(path, 'wb') as f:
            tosave = [model.hyperparams,
                    model.input_preprocessing_type,
                    model.input_preprocessing_params,
                    model.output_preprocessing_type,
                    model.output_preprocessing_params,
                    args[0],#x_train_preprocessed
                    args[1],#y_train_preprocessed.shape[1]
                    args[2]#loss
                    ]
            pickle.dump(tosave, f)
    else:
        with open(path, 'wb') as f:
            pickle.dump(model, f)

def loadModel(path, model_type):
    if model_type == 'rbfnn':
        with open(path, 'rb') as f:
            params = pickle.load(f)
            model = Pipeline('rbfnn')
            model.input_preprocessing_type = params[1]
            model.input_preprocessing_params = params[2]
            model.output_preprocessing_type = params[3]
            model.output_preprocessing_params = params[4]
            model.modelBuild(params[0], params[5], params[6], params[7])
    else:
        with open(path, 'rb') as f:
            model = pickle.load(f)
    return model

def testResults(model, x_test, y_test, y, x_df, test_idx, LOW_FREQ, HIGH_FREQ, f, sr):
    
    if x_test.shape[1] == 8:
        features = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7','Rotation']
    else:
        features = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'Rotation', 'Flare', 'Ear']
    
    # test the model
    x_test_preprocessed = model.inputPreprocessing( x_test )
    predictions = model.modelPredict( x_test_preprocessed )
    predictions = model.outputConvert( predictions )
    
    # spectral distortion
    sd = spectralDistortion(y_test, predictions)
    mean_sd = np.mean(sd)
    print('====================================================')
    print('Best hyperparams: ', model.hyperparams)
    print('Best input and output preprocessing: ', model.input_preprocessing_type, ', ', model.output_preprocessing_type)
    if model.input_preprocessing_type == 'selection':
        print('Selected features: ')
        for feature_index in model.input_preprocessing_params[1]:
            print(features[feature_index], ' ', end = '')
        print()
        
    print('----------------------------------------------------')
    print('Mean spectral distortion for the test set: ', mean_sd)
    
    # sonicom and hutubs mean sd
    y_test_dataset = np.array(x_df.iloc[test_idx]['Dataset'])
    count_sonicom = 0
    count_hutubs= 0
    sonicom_mean_sd = 0
    hutubs_mean_sd = 0
    hutubs_indices = []
    sonicom_indices = []
    for i in range(sd.shape[0]):
        if y_test_dataset[i] == 'SONICOM':
            sonicom_indices.append(i)
            sonicom_mean_sd += sd[i]
            count_sonicom += 1
        else:
            hutubs_indices.append(i)
            hutubs_mean_sd += sd[i]
            count_hutubs += 1
    sonicom_mean_sd /= count_sonicom
    hutubs_mean_sd /= count_hutubs
    print('Mean spectral distortion for HUTUBS examples in the test set: ', hutubs_mean_sd)
    print('Mean spectral distortion for SONICOM examples in the test set: ', sonicom_mean_sd)
    
    # equalizing the test set
    y_test_min_phase = y[test_idx, :]
    y_test_inverse, _, _ = inverseLMSRegularized( spec2mag(y_test_min_phase, 's'), LOW_FREQ, HIGH_FREQ, f, sr )
    y_test_equalized = np.multiply( y_test_inverse, y_test_min_phase)
    #equalizing the predictions
    pred_min_phase = minimumPhaseSpectrum(ct.db2mag(predictions), 's')
    predictions_inverse, _, _ = inverseLMSRegularized( spec2mag(pred_min_phase, 's'), LOW_FREQ, HIGH_FREQ, f, sr )
    predictions_equalized = np.multiply( predictions_inverse, pred_min_phase)
    
    
    # geronazzo metrics
    peak_error_test, notch_error_test, _, _, peaks_freq_test, notches_freq_test, _, _ = peakError( y_test_equalized, f, 1, LOW_FREQ, HIGH_FREQ)
    peak_error_pred, notch_error_pred, _, _, peaks_freq_pred, notches_freq_pred, _, _ = peakError( predictions_equalized, f, 1, LOW_FREQ, HIGH_FREQ)
    
    peak_error_test_hutubs, notch_error_test_hutubs, _, _, peaks_freq_test_hutubs, notches_freq_test_hutubs, _, _ = peakError( y_test_equalized[hutubs_indices], f, 1, LOW_FREQ, HIGH_FREQ)
    peak_error_pred_hutubs, notch_error_pred_hutubs, _, _, peaks_freq_pred_hutubs, notches_freq_pred_hutubs, _, _ = peakError( predictions_equalized[hutubs_indices], f, 1, LOW_FREQ, HIGH_FREQ)
    
    peak_error_test_sonicom, notch_error_test_sonicom, _, _, peaks_freq_test_sonicom, notches_freq_test_sonicom, _, _ = peakError( y_test_equalized[sonicom_indices], f, 1, LOW_FREQ, HIGH_FREQ)
    peak_error_pred_sonicom, notch_error_pred_sonicom, _, _, peaks_freq_pred_sonicom, notches_freq_pred_sonicom, _, _ = peakError( predictions_equalized[sonicom_indices], f, 1, LOW_FREQ, HIGH_FREQ)
    
    broadband_error_test = broadbandError(
        ERBError(y_test_equalized, f, sr, LOW_FREQ, HIGH_FREQ),
        peak_error_test,
        notch_error_test)
    
    broadband_error_pred = broadbandError(
        ERBError(predictions_equalized, f, sr, LOW_FREQ, HIGH_FREQ),
        peak_error_pred,
        notch_error_pred)
    
    broadband_error_test_hutubs = broadbandError(
        ERBError(y_test_equalized[hutubs_indices], f, sr, LOW_FREQ, HIGH_FREQ),
        peak_error_test_hutubs,
        notch_error_test_hutubs)
    
    broadband_error_pred_hutubs = broadbandError(
        ERBError(predictions_equalized[hutubs_indices], f, sr, LOW_FREQ, HIGH_FREQ),
        peak_error_pred_hutubs,
        notch_error_pred_hutubs)
    
    broadband_error_test_sonicom = broadbandError(
        ERBError(y_test_equalized[sonicom_indices], f, sr, LOW_FREQ, HIGH_FREQ),
        peak_error_test_sonicom,
        notch_error_test_sonicom)
    
    broadband_error_pred_sonicom = broadbandError(
        ERBError(predictions_equalized[sonicom_indices], f, sr, LOW_FREQ, HIGH_FREQ),
        peak_error_pred_sonicom,
        notch_error_pred_sonicom)
    
    print('----------------------------------------------------')
    print('Mean peak error difference: ', np.mean( np.abs( peak_error_test - peak_error_pred ) ) )
    print('Peaks match percentage (total): ', peaks_notches_match(peaks_freq_test, peaks_freq_pred, notches_freq_test, notches_freq_pred) )
    print('Mean peaks distance: ', peaks_distance(peaks_freq_test, peaks_freq_pred, notches_freq_test, notches_freq_pred))
    print('Mean peak error difference HUTUBS: ', np.mean( np.abs( peak_error_test_hutubs - peak_error_pred_hutubs ) ))
    print('Peaks match percentage HUTUBS: ', peaks_notches_match(peaks_freq_test_hutubs, peaks_freq_pred_hutubs, notches_freq_test_hutubs, notches_freq_pred_hutubs) )
    print('Mean peaks distance HUTUBS: ', peaks_distance(peaks_freq_test_hutubs, peaks_freq_pred_hutubs, notches_freq_test_hutubs, notches_freq_pred_hutubs))
    print('Mean peak error difference SONICOM: ', np.mean( np.abs( peak_error_test_sonicom - peak_error_pred_sonicom ) ))
    print('Peaks match percentage SONICOM: ', peaks_notches_match(peaks_freq_test_sonicom, peaks_freq_pred_sonicom, notches_freq_test_sonicom, notches_freq_pred_sonicom) )
    print('Mean peaks distance SONICOM: ', peaks_distance(peaks_freq_test_sonicom, peaks_freq_pred_sonicom, notches_freq_test_sonicom, notches_freq_pred_sonicom))
    
    print('----------------------------------------------------')
    print('Mean broadband error difference: ', np.mean( np.abs( broadband_error_test - broadband_error_pred ) ))
    print('Mean broadband error difference HUTUBS: ', np.mean( np.abs( broadband_error_test_hutubs - broadband_error_pred_hutubs ) ))
    print('Mean broadband error difference SONICOM: ', np.mean( np.abs( broadband_error_test_sonicom - broadband_error_pred_sonicom ) ))
    
    print('====================================================')
    
    return predictions, sd

def trainResults(model, x_test, y_test, y, x_df, test_idx, LOW_FREQ, HIGH_FREQ, f, sr):
    
    if x_test.shape[1] == 8:
        features = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7','Rotation']
    else:
        features = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'Rotation', 'Flare', 'Ear']
    
    # test the model
    x_test_preprocessed = model.inputPreprocessing( x_test )
    predictions = model.modelPredict( x_test_preprocessed )
    predictions = model.outputConvert( predictions )
    
    # spectral distortion
    sd = spectralDistortion(y_test, predictions)
    mean_sd = np.mean(sd)
    print('====================================================')
    print('ERROR METRICS ON TRAIN SET')
    print('Mean spectral distortion for the test set: ', mean_sd)
    
    # sonicom and hutubs mean sd
    y_test_dataset = np.array(x_df.iloc[test_idx]['Dataset'])
    count_sonicom = 0
    count_hutubs= 0
    sonicom_mean_sd = 0
    hutubs_mean_sd = 0
    hutubs_indices = []
    sonicom_indices = []
    for i in range(sd.shape[0]):
        if y_test_dataset[i] == 'SONICOM':
            sonicom_indices.append(i)
            sonicom_mean_sd += sd[i]
            count_sonicom += 1
        else:
            hutubs_indices.append(i)
            hutubs_mean_sd += sd[i]
            count_hutubs += 1
    sonicom_mean_sd /= count_sonicom
    hutubs_mean_sd /= count_hutubs
    print('Mean spectral distortion for HUTUBS examples in the test set: ', hutubs_mean_sd)
    print('Mean spectral distortion for SONICOM examples in the test set: ', sonicom_mean_sd)
    
    # equalizing the test set
    y_test_min_phase = y[test_idx, :]
    y_test_inverse, _, _ = inverseLMSRegularized( spec2mag(y_test_min_phase, 's'), LOW_FREQ, HIGH_FREQ, f, sr )
    y_test_equalized = np.multiply( y_test_inverse, y_test_min_phase)
    #equalizing the predictions
    pred_min_phase = minimumPhaseSpectrum(ct.db2mag(predictions), 's')
    predictions_inverse, _, _ = inverseLMSRegularized( spec2mag(pred_min_phase, 's'), LOW_FREQ, HIGH_FREQ, f, sr )
    predictions_equalized = np.multiply( predictions_inverse, pred_min_phase)
    
    
    # geronazzo metrics
    peak_error_test, notch_error_test, _, _, peaks_freq_test, notches_freq_test, _, _ = peakError( y_test_equalized, f, 1, LOW_FREQ, HIGH_FREQ)
    peak_error_pred, notch_error_pred, _, _, peaks_freq_pred, notches_freq_pred, _, _ = peakError( predictions_equalized, f, 1, LOW_FREQ, HIGH_FREQ)
    
    peak_error_test_hutubs, notch_error_test_hutubs, _, _, peaks_freq_test_hutubs, notches_freq_test_hutubs, _, _ = peakError( y_test_equalized[hutubs_indices], f, 1, LOW_FREQ, HIGH_FREQ)
    peak_error_pred_hutubs, notch_error_pred_hutubs, _, _, peaks_freq_pred_hutubs, notches_freq_pred_hutubs, _, _ = peakError( predictions_equalized[hutubs_indices], f, 1, LOW_FREQ, HIGH_FREQ)
    
    peak_error_test_sonicom, notch_error_test_sonicom, _, _, peaks_freq_test_sonicom, notches_freq_test_sonicom, _, _ = peakError( y_test_equalized[sonicom_indices], f, 1, LOW_FREQ, HIGH_FREQ)
    peak_error_pred_sonicom, notch_error_pred_sonicom, _, _, peaks_freq_pred_sonicom, notches_freq_pred_sonicom, _, _ = peakError( predictions_equalized[sonicom_indices], f, 1, LOW_FREQ, HIGH_FREQ)
    
    broadband_error_test = broadbandError(
        ERBError(y_test_equalized, f, sr, LOW_FREQ, HIGH_FREQ),
        peak_error_test,
        notch_error_test)
    
    broadband_error_pred = broadbandError(
        ERBError(predictions_equalized, f, sr, LOW_FREQ, HIGH_FREQ),
        peak_error_pred,
        notch_error_pred)
    
    broadband_error_test_hutubs = broadbandError(
        ERBError(y_test_equalized[hutubs_indices], f, sr, LOW_FREQ, HIGH_FREQ),
        peak_error_test_hutubs,
        notch_error_test_hutubs)
    
    broadband_error_pred_hutubs = broadbandError(
        ERBError(predictions_equalized[hutubs_indices], f, sr, LOW_FREQ, HIGH_FREQ),
        peak_error_pred_hutubs,
        notch_error_pred_hutubs)
    
    broadband_error_test_sonicom = broadbandError(
        ERBError(y_test_equalized[sonicom_indices], f, sr, LOW_FREQ, HIGH_FREQ),
        peak_error_test_sonicom,
        notch_error_test_sonicom)
    
    broadband_error_pred_sonicom = broadbandError(
        ERBError(predictions_equalized[sonicom_indices], f, sr, LOW_FREQ, HIGH_FREQ),
        peak_error_pred_sonicom,
        notch_error_pred_sonicom)
    
    print('----------------------------------------------------')
    print('Mean peak error difference: ', np.mean( np.abs( peak_error_test - peak_error_pred ) ) )
    #print('Peaks match percentage (total): ', peaks_notches_match(peaks_freq_test, peaks_freq_pred) )
    #print('Nothces match percentage (total): ', peaks_notches_match(notches_freq_test, notches_freq_pred) )
    print('Mean peak error difference HUTUBS: ', np.mean( np.abs( peak_error_test_hutubs - peak_error_pred_hutubs ) ))
    #print('Peaks match percentage HUTUBS: ', peaks_notches_match(peaks_freq_test_hutubs, peaks_freq_pred_hutubs) )
    #print('Notches match percentage HUTUBS: ', peaks_notches_match(notches_freq_test_hutubs, notches_freq_pred_hutubs) )
    print('Mean peak error difference SONICOM: ', np.mean( np.abs( peak_error_test_sonicom - peak_error_pred_sonicom ) ))
    #print('Peaks match percentage SONICOM: ', peaks_notches_match(peaks_freq_test_sonicom, peaks_freq_pred_sonicom) )
    #print('Notches match percentage SONICOM: ', peaks_notches_match(notches_freq_test_sonicom, notches_freq_pred_sonicom) )
    
    print('----------------------------------------------------')
    print('Mean broadband error difference: ', np.mean( np.abs( broadband_error_test - broadband_error_pred ) ))
    print('Mean broadband error difference HUTUBS: ', np.mean( np.abs( broadband_error_test_hutubs - broadband_error_pred_hutubs ) ))
    print('Mean broadband error difference SONICOM: ', np.mean( np.abs( broadband_error_test_sonicom - broadband_error_pred_sonicom ) ))
    
    print('====================================================')
    
    return predictions

def mannwhitneyu_test(model, x_test, y_test, y, x_df, test_idx):
    
    if x_test.shape[1] == 8:
        features = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7','Rotation']
    else:
        features = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'Rotation', 'Flare', 'Ear']
    
    # test the model
    x_test_preprocessed = model.inputPreprocessing( x_test )
    predictions = model.modelPredict( x_test_preprocessed )
    predictions = model.outputConvert( predictions )
    
    y_test_dataset = np.array(x_df.iloc[test_idx]['Dataset'])
    
    sd = spectralDistortion(y_test, predictions)
    hutubs = []
    sonicom = []
    for i in range(sd.shape[0]):
        if y_test_dataset[i] == 'SONICOM':
            sonicom.append(sd[i])
        else:
            hutubs.append(sd[i])
    
    print( mannwhitneyu(hutubs, sonicom) )
    
    return

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

def peaks_distance(seq_truth_1, seq_predict_1, seq_truth_2, seq_predict_2):
    tot = 0
    norm = 0
    for i in range(len(seq_truth_1)):
        for j in range(len(seq_truth_1[i])):
            tot += np.abs( (seq_truth_1[i][j] - find_nearest(seq_predict_1[i], seq_truth_1[i][j])) )
            norm += 1
    for i in range(len(seq_truth_2)):
        if len(seq_truth_2[i]) != 0 and len(seq_predict_2[i]):
            for j in range(len(seq_truth_2[i])):
                tot += np.abs( (seq_truth_2[i][j] - find_nearest(seq_predict_2[i], seq_truth_2[i][j])) )
                norm += 1
        else:
            continue
    return tot / norm