import h5py
import numpy as np
from lxml import etree
from scipy.io import savemat
import os

# Percorso della directory dei file
input_directory = '/home/martadr/Scrivania/TESI/lifex_ep/results_la_healthy/'
output_directory = '/home/martadr/Scrivania/TESI/LDNets/data/LA_healthy'

h5_file_path_nodes = '/home/martadr/Scrivania/TESI/lifex_ep/results_la_healthy/solution_CRN_LA_healthy_000000.h5'

# Funzione per leggere i dati dal file .h5
def read_h5_data(h5_filename, dataset_path):
    if not os.path.exists(h5_filename):
        raise FileNotFoundError(f"File {h5_filename} non trovato.")
    with h5py.File(h5_filename, 'r') as h5file:
        if dataset_path not in h5file:
            raise KeyError(f"Dataset {dataset_path} non trovato nel file {h5_filename}.")
        data = h5file[dataset_path][:]
    return data

# Funzione per processare un singolo file .xdmf
def process_file(idx):

    xdmf_idx = idx * 200

    # Costruisci i percorsi dei file
    xdmf_file_name = f'solution_CRN_LA_healthy_{xdmf_idx:06d}.xdmf'
    h5_file_name = f'solution_CRN_LA_healthy_{xdmf_idx:06d}.h5'
    
    xdmf_file_path = os.path.join(input_directory, xdmf_file_name)
    h5_file_path = os.path.join(input_directory, h5_file_name)
    
    # Verifica che il file .xdmf esista
    if not os.path.exists(xdmf_file_path):
        raise FileNotFoundError(f"File {xdmf_file_path} non trovato.")
    
    # Leggere il file .xdmf
    tree = etree.parse(xdmf_file_path)
    root = tree.getroot()

    # Estrazione dei percorsi ai dataset
    domain = root.find('.//Domain')
    temporal_grid = domain.find('.//Grid[@Name="CellTime"]')
    grid = temporal_grid.find('.//Grid[@Name="mesh"]')

    geometry = grid.find('.//Geometry')
    attribute = grid.find('.//Attribute')

    # Percorso del dataset di geometria e attributi (coordinate e potenziale)
    geometry_data_item = geometry.find('.//DataItem').text.strip()
    attribute_data_item = attribute.find('.//DataItem').text.strip()

    # Estrarre il percorso del dataset dal file .h5
    geometry_dataset = geometry_data_item.split(':')[1].strip()
    attribute_dataset = attribute_data_item.split(':')[1].strip()

    # Estrazione delle coordinate (x, y, z) e del potenziale (v)
    coordinates = read_h5_data(h5_file_path_nodes, geometry_dataset)
    values = read_h5_data(h5_file_path, attribute_dataset)

    # Assicurarsi che le dimensioni corrispondano
    if coordinates.shape[0] != values.size:
        raise ValueError(f"Le dimensioni delle coordinate e dei valori non corrispondono per il file {xdmf_file_name}.")

    # Specificare il percorso di salvataggio del file .mat
    mat_file_name = f'sample_{idx}.mat'
    output_path = os.path.join(output_directory, mat_file_name)

    # Creare il dizionario da salvare nel file .mat
    data_dict = {
        'x': coordinates[:, 0],
        'y': coordinates[:, 1],
        'z': coordinates[:, 2],
        'v': values.flatten()  # Assicurarsi che il vettore dei valori sia unidimensionale
    }

    # Salvare i dati nel file .mat
    savemat(output_path, data_dict)
    print(f"Dati salvati in: {output_path}")

# Ciclo attraverso i file
for i in range(151):
    process_file(i)
