import sys
sys.path.append('/home/nathan.regev/software/git_repo/')
from END2END_optim import E2E
#from optimising_photon_yield import PH_yield
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import os
from NN_models import Model1, Model2,ScintProcessStochasticModel,ScintProcessStochasticModel_no_CNN

mod = '4'

#weights_path = r"C:\Users\nathan\OneDrive - Technion\Desktop\technion\7\projectB\ScintCityPython\20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model2_parameters.pth"
#model_type='chi'
model_type='MSE'
MSE_path=r'/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size_model4results/MSE_3/results_wieghts.pth'
chi_path=r'/home/nathan.regev/software/git_repo/20250212094403680822_Database_100000samples_nonperiodic_sim_100000_photons_10layers_PbOSiO2_varied_size_model4results/chi_3/results_wieghts.pth'
if model_type=='MSE':
    weights_path=MSE_path
    suffix='MSE'
else:
    weights_path=chi_path
    suffix='chi'
input_size=99
output_size=input_size*input_size
state_dict = torch.load(weights_path,map_location=torch.device('cpu'))

if mod=='1':
    model = Model1( input_size, input_size, output_size)
elif mod=='2':
    model = Model2( input_size, input_size, output_size)
elif mod=='3':
    model = ScintProcessStochasticModel( input_size, input_size, output_size)
elif mod=='4':
    model = ScintProcessStochasticModel_no_CNN( input_size, input_size, output_size)
        
model.load_state_dict(state_dict)


d_scint = 200
d_dielectric = 200



starts_with_scintilator = 0
num_of_layers=10
# Use torch optimizer to optimize layer_vec
#dist='exponential_optim'
loss_type='total photon'
#loss_type='FarField emission'
dist='optimisation'
thickness_scint=np.linspace(150,300,100)
thickness_dialecric=np.linspace(150,300,100)
loss = torch.zeros((thickness_scint.shape[0], thickness_dialecric.shape[0]))
theta_cutoff = 0.5
for i in range (loss.shape[0]):
    for j in range (loss.shape[1]):
        
            if starts_with_scintilator:
                layer_vec= [thickness_scint[i], thickness_dialecric[j]] * (num_of_layers//2)
            else:
                layer_vec= [thickness_dialecric[j],thickness_scint[i]] * (num_of_layers//2)
            E2E_model = E2E(dist, mod, model, layer_vec, starts_with_scintilator, num_of_layers)
            emission_distribution, emission_theta, emission_phi = E2E_model()
            selected_elements = emission_theta[1][(emission_theta[0] > -theta_cutoff) & (emission_theta[0] < theta_cutoff)]
            if loss_type=='FarField emission':
                loss[i,j]=torch.sum(selected_elements)
            else:
                loss[i,j]=torch.sum(torch.from_numpy(emission_distribution))
plt.figure()
plt.imshow(
    loss.detach().numpy(),
    extent=[thickness_scint[0], thickness_scint[-1], thickness_dialecric[0], thickness_dialecric[-1]],
    origin='lower',
    aspect='auto'
)
plt.colorbar(label='signal')
plt.xlabel('Scintillator Thickness (nm)')
plt.ylabel('Dielectric Thickness (nm)')
plt.show()
plt.savefig(loss_type+'_loss_map'+suffix+'.png')
plt.savefig(loss_type+'_loss_map'+suffix+'.svg')
with open(loss_type+'_loss_map'+suffix+'.pkl', 'wb') as f:
    pickle.dump(loss.detach().numpy(), f)