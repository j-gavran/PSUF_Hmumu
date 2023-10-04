import os
import numpy as np

#################################################################################
# User defined

multiply = 100
data_path = 'DATA/generated_histograms'
histo_name = 'hist_range_110-160_nbin-25_'
labels = ['Background', 'Signal', 'Data']

#################################################################################
# Load data

pldict = {}
for label in labels:
    fileName = os.path.join(data_path, histo_name + label + '.npz')
    with np.load(fileName, 'rb') as data:
        bin_centers = data['bin_centers']
        bin_edges = data['bin_edges']
        bin_values = data['bin_values']
        bin_errors = data['bin_errors']

    pldict[label] = [bin_centers, bin_edges, bin_values, bin_errors]


#################################################################################
# Add multiple of signal to Data and Signal

pldict['Data'][2] += multiply * pldict['Signal'][2]
pldict['Data'][3] = np.sqrt(pldict['Data'][3]**2 + pldict['Signal'][3]**2)

pldict['Signal'][2] += multiply * pldict['Signal'][2]
pldict['Signal'][3] = multiply * pldict['Signal'][3]


#################################################################################
# Save Asimov data

for label in labels:
    fileName = histo_name + 'Asimov' + label + '.npz'
    np.savez(
        os.path.join(data_path, fileName),
        bin_centers=pldict[label][0],
        bin_edges=pldict[label][1],
        bin_values=pldict[label][2],
        bin_errors=pldict[label][3]
    )
