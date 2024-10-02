import h5py
import numpy as np
from lxml import etree
from scipy.io import savemat
import os

# Percorso della directory dei file
input_directory = '/home/martadr/Scrivania/TESI/lifex_ep/results_re_la_h_t044/'
output_directory = '/home/martadr/Scrivania/TESI/LDNets/data/LA_healthy/'

# Inizializzare le liste per accumulare i dati
all_potentials = []

# Funzione per leggere i dati dal file .h5
def read_h5_data(h5_filename, dataset_path):
    if not os.path.exists(h5_filename):
        raise FileNotFoundError(f"File {h5_filename} non trovato.")
    with h5py.File(h5_filename, 'r') as h5file:
        if dataset_path not in h5file:
            raise KeyError(f"Dataset {dataset_path} non trovato nel file {h5_filename}.")
        data = h5file[dataset_path][:]
    return data

# Costruisci i percorsi dei file
xdmf_file_name = f'solution_CRN_LA_healthy_000000.xdmf'
h5_file_name = f'solution_CRN_LA_healthy_000000.h5'

xdmf_file_path = os.path.join(input_directory, xdmf_file_name)
h5_file_path = os.path.join(input_directory, h5_file_name)

# Leggere il file .xdmf
tree = etree.parse(xdmf_file_path)
root = tree.getroot()

# Estrazione dei percorsi ai dataset
domain = root.find('.//Domain')
temporal_grid = domain.find('.//Grid[@Name="CellTime"]')
grid = temporal_grid.find('.//Grid[@Name="mesh"]')

geometry = grid.find('.//Geometry')

# Percorso del dataset di geometria (coordinate)
geometry_data_item = geometry.find('.//DataItem').text.strip()

# Estrarre il percorso del dataset dal file .h5
geometry_dataset = geometry_data_item.split(':')[1].strip()

# Estrazione delle coordinate (x, y, z)
coordinates = read_h5_data(h5_file_path, geometry_dataset)

# Funzione per processare un singolo file .xdmf
def process_file(idx):
    # Calcola l'indice per il file .xdmf
    xdmf_idx = idx * 20
    
    # Costruisci i percorsi dei file
    xdmf_file_name = f'solution_CRN_LA_healthy_{xdmf_idx:06d}.xdmf'
    h5_file_name = f'solution_CRN_LA_healthy_{xdmf_idx:06d}.h5'
    
    xdmf_file_path = os.path.join(input_directory, xdmf_file_name)
    h5_file_path = os.path.join(input_directory, h5_file_name)
    
    # Verifica che il file .xdmf esista
    if not os.path.exists(xdmf_file_path):
        print(f"File {xdmf_file_path} non trovato. Skipping...")
        return
    
    # Leggere il file .xdmf
    tree = etree.parse(xdmf_file_path)
    root = tree.getroot()

    # Estrazione dei percorsi ai dataset
    domain = root.find('.//Domain')
    temporal_grid = domain.find('.//Grid[@Name="CellTime"]')
    grid = temporal_grid.find('.//Grid[@Name="mesh"]')

    attribute = grid.find('.//Attribute')

    # Percorso del dataset di attributi (potenziale)
    attribute_data_item = attribute.find('.//DataItem').text.strip()

    # Estrarre il percorso del dataset dal file .h5
    attribute_dataset = attribute_data_item.split(':')[1].strip()

    # Estrazione del potenziale (v)
    values = read_h5_data(h5_file_path, attribute_dataset)

    # Aggiungere le coordinate e il potenziale alle liste
    all_potentials.append(values)

# Ciclo attraverso i file
for i in range(441):  # 151 file da 0 a 150 inclusi
    process_file(i)

# Verifica che tutte le matrici di potenziale abbiano la stessa dimensione
potential_shapes = {v.shape for v in all_potentials}
if len(potential_shapes) != 1:
    raise ValueError("Le dimensioni dei valori di potenziale non sono uniformi tra i file.")

# Creare una matrice di potenziale dove le righe rappresentano i punti di coordinate e le colonne rappresentano gli istanti di tempo
potential_matrix = np.hstack(all_potentials)

# Specificare il percorso di salvataggio del file .mat
mat_file_name = 'sample_1.mat'
output_path = os.path.join(output_directory, mat_file_name)

# Creare il dizionario da salvare nel file .mat
data_struct = {
    'data_train':{
        'vh': potential_matrix, # La matrice di potenziale, dove righe sono i punti e colonne sono gli istanti di tempo,
        'x': coordinates[:, 0],
        'y': coordinates[:, 1],
        'z': coordinates[:, 2]
    }
}

# Salvare i dati nel file .mat
savemat(output_path, data_struct)
print(f"Dati salvati in: {output_path}")
