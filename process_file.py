import h5py
import numpy as np
from lxml import etree
from scipy.io import savemat
import os

# Percorsi dei file
xdmf_file_path = '/home/martadr/Scrivania/TESI/lifex_ep/results_re_la_h_t15/solution_CRN_LA_healthy_000400.xdmf'
h5_file_path_nodes = '/home/martadr/Scrivania/TESI/lifex_ep/results_re_la_h_t15/solution_CRN_LA_healthy_000000.h5'
h5_file_path = '/home/martadr/Scrivania/TESI/lifex_ep/results_re_la_h_t15/solution_CRN_LA_healthy_000400.h5'

# Funzione per leggere i dati dal file .h5
def read_h5_data(h5_filename, dataset_path):
    if not os.path.exists(h5_filename):
        raise FileNotFoundError(f"File {h5_filename} non trovato.")
    with h5py.File(h5_filename, 'r') as h5file:
        data = h5file[dataset_path][:]
    return data

# Leggere il file .xdmf
tree = etree.parse(xdmf_file_path)
root = tree.getroot()

# Estrazione dei percorsi ai file .h5
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
    raise ValueError("Le dimensioni delle coordinate e dei valori non corrispondono.")

# Creare il dizionario da salvare nel file .mat
data_dict = {
    'x': coordinates[:, 0],
    'y': coordinates[:, 1],
    'z': coordinates[:, 2],
    'v': values.flatten()  # Assicurarsi che il vettore dei valori sia unidimensionale
}

output_directory = '/home/martadr/Scrivania/TESI/LDNets/data/LA_healthy'
output_file_name = 'sample000400.mat'
output_path = os.path.join(output_directory, output_file_name)

# Salvare i dati nel file .mat
savemat(output_path, data_dict)


