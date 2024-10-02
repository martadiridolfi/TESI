#!/usr/bin/env python3
#%% Import modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
# We configure TensorFlow to work in double precision 
tf.keras.backend.set_floatx('float64')

import utils
import optimization
import os

#%% Set some hyperparameters
dt = 2.27e-03
#dt_base = 0.44
#num_latent_states = 9

#%% Define problem
problem = {
    "space": {
        "dimension" : 3
    },
    #"input_parameters": [
    #    { "name": "diameter" }
    #],
    #"input_signals": [
    #    { "name": "impulse" }
    #],
    "output_fields": [
        { "name": "u" }
    ]
}

'''
normalization = {
'space': { 'min' : [0], 'max' : [+100.0]},
'time': { 'time_constant' : dt_base },
#'input_parameters': {
#    'diameter': { 'min': 1.0 , 'max': 10.0 },
#},
#'input_signals': {
#    'impulse': { 'min':   0.0 , 'max': 1.0},
#},
'output_fields': {
    'u': { 'min': 0.0, "max": 1.0 }
}
}
'''


#%% Dataset
data_set_path = '/home/martadr/Scrivania/TESI/LDNets/data/LA_healthy/'

import scipy.io
mat_1 = scipy.io.loadmat('/home/martadr/Scrivania/TESI/LDNets/data/LA_healthy/sample_1.mat')
#print(mat_1.keys())

'''
dataset_train = utils.LAhealthy_create_dataset(data_set_path, 0, 1)
dataset_valid = utils.LAhealthy_create_dataset(data_set_path, 0, 1)
#dataset_tests = utils.LAhealthy_create_dataset(data_set_path, 0, 1)

# For reproducibility (delete if you want to test other random initializations)
np.random.seed(0)
tf.random.set_seed(0)

n_subsample = 1500


# We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
utils.process_dataset(dataset_train, problem, normalization, dt = dt, num_points_subsample = n_subsample)
utils.process_dataset(dataset_valid, problem, normalization, dt = dt, num_points_subsample = n_subsample)
#utils.process_dataset(dataset_tests, problem, normalization, dt = dt)
''' 

import optuna
from tensorflow.keras.regularizers import  L2

# Funzione obiettivo per il tuning
def objective(trial):
    
    # Learning rate e pesi di regolarizzazione L2
    alpha_reg = trial.suggest_float('alpha_reg', 1e-5, 1e-2, log=True)

    #learning_rate = 5e-2
    #alpha_reg = 1.78e-4

    
    # Costante di normalizzazione temporale
    dt_base =  trial.suggest_float('dt_base', 0.1, 0.5)

    normalization = {
    'space': { 'min' : [0], 'max' : [+100.0]},
    'time': { 'time_constant' : dt_base },
    #'input_parameters': {
    #    'diameter': { 'min': 1.0 , 'max': 10.0 },
    #},
    #'input_signals': {
    #    'impulse': { 'min':   0.0 , 'max': 1.0},
    #},
    'output_fields': {
        'u': { 'min': 0.0, "max": 1.0 }
    }
    }

    dataset_train = utils.LAhealthy_create_dataset(data_set_path, 0, 1)
    dataset_valid = utils.LAhealthy_create_dataset(data_set_path, 0, 1)
    #dataset_tests = utils.LAhealthy_create_dataset(data_set_path, 0, 1)

    # For reproducibility (delete if you want to test other random initializations)
    np.random.seed(0)
    tf.random.set_seed(0)

    n_subsample = 2000

    # We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
    utils.process_dataset(dataset_train, problem, normalization, dt = dt, num_points_subsample = n_subsample)
    utils.process_dataset(dataset_valid, problem, normalization, dt = dt, num_points_subsample = n_subsample)    
    
    # Definisci il numero di stati latenti
    num_latent_states = trial.suggest_int('num_latent_state', 1, 12)  # Modifica se necessario
    '''
    input_shape = (num_latent_states,)# + len(problem['input_signals']) + len(problem['input_parameters']),) 
    NNdyn = tf.keras.Sequential([
                tf.keras.layers.Dense(24, activation = tf.nn.tanh, input_shape = input_shape),
                tf.keras.layers.Dense(num_latent_states)
            ])
    NNdyn.summary()

    # reconstruction network
    input_shape = (None, None, num_latent_states + problem['space']['dimension'])
    NNrec = tf.keras.Sequential([
                tf.keras.layers.Dense(25, activation = tf.nn.tanh, input_shape = input_shape),
                tf.keras.layers.Dense(25, activation = tf.nn.tanh),
                tf.keras.layers.Dense(25, activation = tf.nn.tanh),
                tf.keras.layers.Dense(25, activation = tf.nn.tanh),
                tf.keras.layers.Dense(len(problem['output_fields']))
            ])
    NNrec.summary()
    '''

    
    # Costruzione della rete dinamica NNdyn
    n_layers_dyn = trial.suggest_int('n_layers_dyn', 1, 10)
    units_dyn = trial.suggest_int('units_layer_dyn', 5, 30)
    
    input_shape_dyn = (num_latent_states,)  # Input per NNdyn
    NNdyn = tf.keras.Sequential()
    NNdyn.add(tf.keras.layers.Dense(units_dyn, activation=tf.nn.tanh, input_shape=input_shape_dyn))#, kernel_regularizer=l2(alpha_reg)))
    for i in range(1, n_layers_dyn):
        NNdyn.add(tf.keras.layers.Dense(units_dyn, activation=tf.nn.tanh))#, kernel_regularizer=l2(alpha_reg)))
    NNdyn.add(tf.keras.layers.Dense(num_latent_states))  # Output layer della rete dinamica

    # Costruzione della rete di ricostruzione NNrec
    n_layers_rec = trial.suggest_int('n_layers_rec', 1, 10)
    units_rec = trial.suggest_int('units_layer_rec', 5, 30)

    input_shape_rec = (None, None, num_latent_states + problem['space']['dimension'])  # Input per NNrec
    NNrec = tf.keras.Sequential()
    NNrec.add(tf.keras.layers.Dense(units_rec, activation=tf.nn.tanh, input_shape=input_shape_rec)) #, kernel_regularizer=l2(alpha_reg)))
    for i in range(1, n_layers_rec):
        NNrec.add(tf.keras.layers.Dense(units_rec, activation=tf.nn.tanh))#, kernel_regularizer=l2(alpha_reg)))
    NNrec.add(tf.keras.layers.Dense(len(problem['output_fields'])))  # Output layer della rete di ricostruzione
    
    def evolve_dynamics(dataset):
    # intial condition
        state = tf.zeros((dataset['num_samples'], num_latent_states), dtype=tf.float64) #Inizializzazione dello stato latente con un tensore di zeri
        state_history = tf.TensorArray(tf.float64, size = dataset['num_times']) #TensorArray che tiene traccia della storia degli stati latenti nel tempo. dim = n. passi temporali
        state_history = state_history.write(0, state) #Registrazione stato iniziale (t=0) nel TensorArray
        dt_ref = normalization['time']['time_constant'] #Riferimento temporale per la normalizzazione di dt
        
        # time integration
        for i in tf.range(dataset['num_times'] - 1):
            state = state + dt/dt_ref * NNdyn(tf.concat([state], axis = -1)) # dataset['inp_parameters'], dataset['inp_signals'][:,i,:]], axis = -1)) #Aggiornamento dello stato corrente mediante l'aggiunta della variazione
            state_history = state_history.write(i + 1, state) #Memorizzazione del nuovo stato nel TensorArray

        return tf.transpose(state_history.stack(), perm=(1,0,2)) #Restituzione dell'intera storia degli stati attraverso il tempo per tutti i campioni simulati.

    def reconstruct_output(dataset, states):
        states_expanded = tf.broadcast_to(tf.expand_dims(states, axis = 2), #expand_dims aggiunge una nuova dimensione all'indice 2 del tensore states
            [dataset['num_samples'], dataset['num_times'], dataset['num_points'], num_latent_states]) #Il broadcasting è un'operazione che espande un tensore lungo una dimensione senza dover replicare fisicamente i dati, consentendo di operare su tensori con forme compatibili.
        return NNrec(tf.concat([states_expanded, dataset['points_full']], axis = 3)) #Rete neurale applicata alla concatenazione di states_expanded con dataset[points_full] restituendo il campo fisico

    def LDNet(dataset):
        states = evolve_dynamics(dataset) #evoluzione temporale degli stati latenti
        return reconstruct_output(dataset, states) #ricostruzione dell'output

    # Definizione della funzione di perdita MSE (Mean Squared Error)
    def MSE(dataset):
        out_fields = LDNet(dataset)  # LDNet è la combinazione di NNdyn e NNrec
        error = out_fields - dataset['out_fields']
        return tf.reduce_mean(tf.square(error))
    
    def weights_reg(NN):
        return sum([tf.reduce_mean(tf.square(lay.kernel)) for lay in NN.layers])/len(NN.layers)

    # Funzione di allenamento (loss) che include la regolarizzazione
    def loss():
        return MSE(dataset_train) + alpha_reg * (weights_reg(NNdyn) + weights_reg(NNrec))
    
    def MSE_valid(): return MSE(dataset_valid)

    # Compilazione del modello
    trainable_variables = NNdyn.variables + NNrec.variables
    opt = optimization.OptimizationProblem(trainable_variables, loss, MSE_valid)

    num_epochs_Adam = 50
    #num_epochs_BFGS = 4000

    print('training (Adam)...')
    opt.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=1e-2))
    #print('training (BFGS)...')
    #opt.optimize_BFGS(num_epochs_BFGS)
    
    # Calcolo della perdita sul validation set
    val_loss = MSE(dataset_valid)
    return val_loss

# Crea uno studio per ottimizzare gli iperparametri
study = optuna.create_study(
    direction='minimize', 
    storage='sqlite:///optuna_trials.db',
    study_name='ldnet_study',
    load_if_exists=True
)
#study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300)

# Stampa i migliori iperparametri trovati
print("Migliori parametri:", study.best_params)

