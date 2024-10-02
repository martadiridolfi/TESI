#!/usr/bin/env python3
#%% Import modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
# We configure TensorFlow to work in double precision 
tf.keras.backend.set_floatx('float64')

from mpl_toolkits.mplot3d import Axes3D

import utils
import optimization

#%% Set some hyperparameters
dt = 5.56e-03
dt_base = 1.63e-1
num_latent_states = 9

#%% Define problem
problem = {
    "space": {
        "dimension" : 3
    },
    "input_parameters": [
        { "name": "diameter" }
    ],
    "input_signals": [
        { "name": "impulse" }
    ],
    "output_fields": [
        { "name": "u" }
    ]
}

normalization = {
    'space': { 'min' : [0], 'max' : [+100.0]},
    'time': { 'time_constant' : dt_base },
    'input_parameters': {
        'diameter': { 'min': 1.0 , 'max': 10.0 },
    },
    'input_signals': {
        'impulse': { 'min':   0.0 , 'max': 1.0},
    },
    'output_fields': {
        'u': { 'min': 0.0, "max": 1.0 }
    }
}

#%% Dataset
data_set_path = '/home/martadr/Scrivania/TESI/LDNets/data/LA_healthy/'

import scipy.io
mat_1 = scipy.io.loadmat('/home/martadr/Scrivania/TESI/LDNets/data/LA_healthy/sample_1.mat')
print(mat_1.keys())


dataset_train = utils.LAhealthy_create_dataset(data_set_path, 0, 1)
dataset_valid = utils.LAhealthy_create_dataset(data_set_path, 0, 1)
dataset_tests = utils.LAhealthy_create_dataset(data_set_path, 0, 1)

# For reproducibility (delete if you want to test other random initializations)
np.random.seed(0)
tf.random.set_seed(0)

# We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
utils.process_dataset(dataset_train, problem, normalization, dt = dt, num_points_subsample = 50)
utils.process_dataset(dataset_valid, problem, normalization, dt = dt, num_points_subsample = 50)
utils.process_dataset(dataset_tests, problem, normalization, dt = dt)

NNdyn = tf.keras.models.load_model('NNdyn_model.h5')
NNrec = tf.keras.models.load_model('NNrec_model.h5')

NNdyn.compile(optimizer='adam', loss='mse')
NNrec.compile(optimizer='adam', loss='mse')

def evolve_dynamics(dataset):
    # intial condition
    state = tf.zeros((dataset['num_samples'], num_latent_states), dtype=tf.float64)
    state_history = tf.TensorArray(tf.float64, size = dataset['num_times'])
    state_history = state_history.write(0, state)
    dt_ref = normalization['time']['time_constant']
    
    # time integration
    for i in tf.range(dataset['num_times'] - 1):
        state = state + dt/dt_ref * NNdyn(tf.concat([state, dataset['inp_parameters'], dataset['inp_signals'][:,i,:]], axis = -1))
        state_history = state_history.write(i + 1, state)

    return tf.transpose(state_history.stack(), perm=(1,0,2))

def reconstruct_output(dataset, states):
    states_expanded = tf.broadcast_to(tf.expand_dims(states, axis = 2), 
        [dataset['num_samples'], dataset['num_times'], dataset['num_points'], num_latent_states])
    return NNrec(tf.concat([states_expanded, dataset['points_full']], axis = 3))

def LDNet(dataset):
    states = evolve_dynamics(dataset)
    return reconstruct_output(dataset, states)

#%% Testing
# Compute predictions.
out_fields = LDNet(dataset_tests)
# Since the models work with normalized data, we map back the outputs into the original ranges.
out_fields_FOM = utils.denormalize_output(dataset_tests['out_fields'], problem, normalization).numpy()
out_fields_ROM = utils.denormalize_output(out_fields                 , problem, normalization).numpy()

NRMSE = np.sqrt(np.mean(np.square(out_fields_ROM - out_fields_FOM))) / (np.max(out_fields_FOM) - np.min(out_fields_FOM))

import scipy.stats
R_coeff = scipy.stats.pearsonr(np.reshape(out_fields_ROM, (-1,)), np.reshape(out_fields_FOM, (-1,)))

print('Normalized RMSE:       %1.3e' % NRMSE)
print('Pearson dissimilarity: %1.3e' % (1 - R_coeff[0]))

mesh_points = dataset_tests['points']
#print('Dimensione mesh_points: ', mesh_points.shape)

#%% Postprocessing
from scipy.interpolate import LinearNDInterpolator

for i_sample in range(min(3, out_fields_FOM.shape[0])):  # Limita il numero di campioni a quelli disponibili
    num_times = 4

    x = dataset_tests['points']

    states = evolve_dynamics(dataset_tests)
    x = mesh_points[:, 0]
    y = mesh_points[:, 1]
    z = mesh_points[:, 2]
    points = np.concatenate([[x], [y], [z]], axis=0).transpose()

    points_full = tf.convert_to_tensor(points, dtype=tf.float32)  # Converte i punti della mesh in tensor

    # Assumi che states_expanded sia creato correttamente come in precedenza
    num_samples = states.shape[0]
    num_times = len(dataset_tests['times'])
    num_points = len(mesh_points)

    # Assicurati che il broadcasting sia fatto correttamente
    states_expanded = tf.broadcast_to(tf.expand_dims(tf.expand_dims(states, axis=2), axis=2), [num_samples, len(dataset_tests['times']), num_points, num_latent_states])

    # Espandi le dimensioni di points_full per farlo coincidere con states_expanded
    points_full_expanded = tf.expand_dims(tf.expand_dims(points_full, axis=0), axis=1)
    points_full_expanded = tf.broadcast_to(points_full_expanded, [num_samples, len(dataset_tests['times']), num_points, 3])   

    """
    #points = np.concatenate([[mesh_points[:, 0]], [mesh_points[:, 1]], [mesh_points[:, 2]]], axis=0).transpose()
    points_full = tf.convert_to_tensor(points, dtype=tf.float64)
    #states_expanded = tf.broadcast_to(tf.expand_dims(tf.expand_dims(states[:, :, :], axis=0), axis=2), [1, len(dataset_tests['times']), len(points), num_latent_states])
    states_expanded = tf.broadcast_to(tf.expand_dims(tf.expand_dims(states[i_sample, :, :], axis=0), axis=2), [1, len(dataset_tests['times']), len(mesh_points), num_latent_states])
    """

    times = np.linspace(0, len(dataset_tests['times']) - 1, num=num_times, dtype=int)

    fig = plt.figure(figsize=(15, 8))

    for idxT, iT in enumerate(times):
        ax = fig.add_subplot(2, num_times, idxT + 1, projection='3d')
        ax.set_title(f'FOM t = {dataset_tests["times"][iT] * dt_base:.2f}')

        # Estrai i valori della soluzione FOM per il tempo corrente
        Z_FOM = out_fields_FOM[i_sample, iT, :, 0]

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=Z_FOM, cmap='magma')
        ax.set_zlim(np.min(Z_FOM), np.max(Z_FOM))

        ax = fig.add_subplot(2, num_times, idxT + 1 + num_times, projection='3d')
        ax.set_title(f'ROM t = {dataset_tests["times"][iT] * dt_base:.2f}')

        # Previsione ROM
        Z_ROM = np.reshape(NNrec(tf.concat([tf.expand_dims(states_expanded[:, iT, :, :], axis=1), points_full_expanded[:, iT, :, :]], axis=3)), (len(points)))

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=Z_ROM, cmap='magma')
        ax.set_zlim(np.min(Z_FOM), np.max(Z_FOM))

    fig.savefig('TESIStep1_sample%d.png' % i_sample)    

    
"""
for i_sample in range(min(3, out_fields_FOM.shape[0])):  # Limita il numero di campioni a quelli disponibili
    num_times = 8

    n_vis = 50
    x_vis = np.linspace(-1, 1, num=n_vis)
    y_vis = np.linspace(-1, 1, num=n_vis)
    z_vis = np.linspace(-1, 1, num=n_vis)
    X, Y, Z = np.meshgrid(x_vis, y_vis, z_vis)

    x = dataset_tests['points']

    # Verifica delle dimensioni di out_fields_FOM
    print(f"Processing sample {i_sample} with dimensions: {out_fields_FOM.shape}")

    v_min = np.min(out_fields_FOM[i_sample, :, :, :])
    v_max = np.max(out_fields_FOM[i_sample, :, :, :])

    states = evolve_dynamics(dataset_tests)
    points = np.concatenate([[X.reshape([-1])], [Y.reshape([-1])], [Z.reshape([-1])]], axis=0).transpose()
    points_full = np.broadcast_to(points[None, None, :, :], [1, 1, n_vis**3, 3])
    states_expanded = tf.broadcast_to(tf.expand_dims(tf.expand_dims(states[i_sample, :, :], axis=0), axis=2), [1, len(dataset_tests['times']), n_vis**3, num_latent_states])

    times = np.linspace(0, len(dataset_tests['times']) - 1, num=num_times, dtype=int)
    fig = plt.figure(figsize=(15, 8))
    for idxT, iT in enumerate(times):
        ax = fig.add_subplot(2, num_times, idxT + 1, projection='3d')
        ax.set_title('t = %.2f' % (dataset_tests['times'][iT] * dt_base))

        Z_FOM = LinearNDInterpolator(x, dataset_tests['out_fields'][i_sample, iT, :, 0])(X, Y, Z)
        Z_ROM = np.reshape(NNrec(tf.concat([tf.expand_dims(states_expanded[:, iT, :, :], axis=1), points_full], axis=3)), (n_vis, n_vis, n_vis))

        ax.scatter(X, Y, Z, c=Z_FOM.flatten(), cmap='magma')
        ax.set_zlim(v_min, v_max)

        ax = fig.add_subplot(2, num_times, idxT + 1 + num_times, projection='3d')
        ax.scatter(X, Y, Z, c=Z_ROM.flatten(), cmap='magma')
        ax.set_zlim(v_min, v_max)

    fig.savefig('TESIStep1_sample%d.png' % i_sample)
"""