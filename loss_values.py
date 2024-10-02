import re
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# Leggi il contenuto del file di testo contenente i dati del terminale
with open('/home/martadr/Scrivania/TESI/LDNets/LDNets-main/src/Models/loss_values.txt', 'r') as file:
    lines = file.readlines()

# Inizializza liste vuote per la training loss e la validation loss
training_loss = []
validation_loss = []

# Definisci una regex per estrarre i valori della training e validation loss
loss_pattern = re.compile(r'training loss:\s([\d\.e\+\-]+)\s+-\s+validation loss\s([\d\.e\+\-]+)')

# Cicla attraverso le righe del file e trova le corrispondenze con la regex
for line in lines:
    match = loss_pattern.search(line)
    if match:
        training_loss.append(float(match.group(1)))
        validation_loss.append(float(match.group(2)))

epochs =  np.arange(0, 7101, 10)

tl = np.array(training_loss)
vl = np.array(validation_loss)

fig, axs = plt.subplots(1, 1)
axs.loglog(epochs, training_loss, 'o-', label = 'training loss')
axs.loglog(epochs, validation_loss, 'o-', label = 'validation loss')
axs.axvline(100)
axs.set_xlabel('epochs'), plt.ylabel('loss')
axs.legend()
fig.savefig('loss_values.png')


# Stampa i vettori estratti
#print("Training loss shape: ", tl.shape)
#print("Validation loss shape: ", vl.shape)
#print("Epochs shape: ", epochs.shape)
#print("Training Loss:", training_loss)
#print("Validation Loss:", validation_loss)
