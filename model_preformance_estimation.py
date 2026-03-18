from ScintCityPython.TPMRS_single_emitter import calculating_emmision
import torch.optim as optim
from torch.optim import Optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LogNorm
import sys
sys.path.append('/home/nathan.regev/software/git_repo/')
import ScintCityPython.END2END_optim_v02 as opt_block
import G4_Nanophotonic_Scintillator.utils.generate_database as geant4
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from ScintCityPython.NN_models import CompleteAutoenconder_Model,PriorLayerModel,Model1_thick
from ScintCityPython.END2END_optim_v02 import E2E_continous,E2E_exponential
def create_boolean_width(z_bins, scint, dialect, starts_with_scintilator):
    total_list = np.empty(scint.size + dialect.size, dtype=scint.dtype)
    total_list[(1 - starts_with_scintilator)::2] = scint
    total_list[starts_with_scintilator::2] = dialect
    boolean_list = np.zeros(z_bins.size - 1)
    acumulated_bins_depth = 0
    layer_counter = 0
    partial_sum_of_layers = total_list[0]
    z_diff = z_bins[1] - z_bins[0]
    binned_layers_index = []  # indexing where layers start and end
    for i in range(boolean_list.size):
        if (z_diff + acumulated_bins_depth) < partial_sum_of_layers:  # next bin is within the layer
            boolean_list[i] = (layer_counter + starts_with_scintilator) % 2
        else:
            if layer_counter < (total_list.size - 1):
                layer_counter += 1
                partial_sum_of_layers += total_list[layer_counter]
           
            boolean_list[i] = -1  # interface between layers
        if np.abs(boolean_list[i]) - np.abs(boolean_list[i - 1]) != 0:  # if layers change then need to append the value
            binned_layers_index.append(i)
        acumulated_bins_depth += z_diff
    if binned_layers_index[-1] < i and np.array(binned_layers_index).size<8:
        binned_layers_index.append(i)
    binned_layers_index=binned_layers_index[1:]
    return boolean_list.astype(int), np.array(binned_layers_index)
def predict_output_model(model,layer_vec,layer_type,z_bins_num):

    emission_model = E2E_continous(model,torch.tensor(layer_vec),z_bins_num=z_bins_num) # assuming no inputs are needed
    emission_distribution, emission_theta, emission_phi = emission_model(torch.tensor(layer_type))
    return emission_distribution, emission_theta, emission_phi 
def predict_output_model_exponential(layer_vec,layer_type):
    emission_model = E2E_exponential(torch.tensor(layer_vec),layer_type[0]) # assuming no inputs are needed
    emission_distribution, emission_theta, emission_phi = emission_model()
    return emission_distribution, emission_theta, emission_phi 
    # Model loading and training setup
def simulate_emission_angular(layer_vec,scintillator_list,dialectric_list):
    simulation_scint_list=1e-4*scintillator_list
    simulation_dialectric_list=1e-4*dialectric_list

    propegation_scint_list=1e2*scintillator_list
    propegation_dialectric_list=1e2*dialectric_list
    N=1
    nLayersNS=10
    sim_photons=1_000_000
    bins=100
    max_rad=230e-5
    sample_size=np.sum(simulation_scint_list)+np.sum(simulation_dialectric_list)
    filename='simulation_sim_'+str(sim_photons)+'_photons_'+str(nLayersNS)+'layers'
    scenario=geant4.generate_scenario(simulation_scint_list,simulation_dialectric_list,sim_photons,startsWithScint=1,nLayersNS=nLayersNS)
    ph_x,ph_y,ph_z=geant4.get_denisty_of_scenario(scenario,filename,database=True,path_pre='G4_Nanophotonic_Scintillator')
    ph_r=np.sqrt(np.power(ph_x,2)+np.power(ph_y,2))
    z_bins=np.linspace(0,sample_size,bins)
    r_bins=np.linspace(0,max_rad,bins)
    binned_layesr,binned_layers_index=opt_block.create_boolean_width_numpy(z_bins,simulation_scint_list,simulation_dialectric_list,1)
    grid,edges_1,edes_2=np.histogram2d(ph_r,ph_z,bins=(r_bins,z_bins))
    saving_dict={'scintlator':[],'dialectric':[],'binned':[],'grid':[]}
    saving_dict['scintlator'].append(propegation_scint_list)
    saving_dict['dialectric'].append(propegation_dialectric_list)
    saving_dict['binned'].append(binned_layesr)
    saving_dict['grid'].append(grid)

    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(saving_dict, f)

    #simulate actual photon propegation 
    absorption_scint=3.95e4 # [nm]]
    absorption_other=4.22e6#[nm]
    control= {
            'is_Gz': 1,
            'dz_Nz': 10,
            'distribution':'optimisation',
            'plot_profile': 0,
            'load_data_from_dir': 0,
            'binary vector': torch.tensor(binned_layesr),
            'distribution_map': torch.tensor(grid),
            'layer_vector': 1e2*torch.tensor(layer_vec),
            'start_with_scint': 0,
            'binary_indexes':torch.tensor(binned_layers_index),
            'num_of_layers':10,
            'absorption_scint': absorption_scint,
		    'absorption_other': absorption_other,
        }
    
    sim_emission_ditribution,sim_emission_theta,sim_emission_phi= calculating_emmision(
        control, layer_struct='optimisation'
    )
    print(len(ph_x))
    sim_emission_theta[1]=(len(ph_x))*sim_emission_theta[1].detach().numpy()
    sim_emission_phi[1]=(len(ph_x))*sim_emission_phi[1].detach().numpy()
    os.remove(filename+'.pkl')
    return len(ph_x),sim_emission_theta,sim_emission_phi


if torch.cuda.is_available():
        # If a GPU is available, select it (typically 'cuda:0')
    device = torch.device('cuda')
else:
    # Otherwise, fall back to the CPU
    device = torch.device('cpu')
num_of_layers=10
z_bins=100
r_bins=100
autoencoder_model = CompleteAutoenconder_Model( nlayers=num_of_layers,kparam=10, z_dim=z_bins-1, r_dim=r_bins-1)
Predictor_Path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/run_autoencoder_mod4/optimization_model_nonolinear_12_11_2025/results_wieghts_network.pth'
#Predictor_Path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/run_autoencoder_mod4/251116134846/results_wieghts_network.pth'
predictor_state_dict = torch.load(Predictor_Path, map_location=device)
autoencoder_model.FCPred.load_state_dict(predictor_state_dict)
deconder_Path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/run_autoencoder_mod4/optimization_model_nonolinear_12_11_2025/results_wieghts_decoder.pth'
#deconder_Path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/run_autoencoder_mod4/251116134846/results_wieghts_decoder.pth'
decoder_state_dict = torch.load(deconder_Path, map_location=device)
autoencoder_model.decoder.load_state_dict(decoder_state_dict)


prior_layer_model = PriorLayerModel(num_of_layers,99, 99,0.000250)
prior_layer_model_path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/run_layer_exp_mod4/251117164407/results_wieghts.pth'
prior_layer_model_dict = torch.load(prior_layer_model_path)
prior_layer_model.load_state_dict(prior_layer_model_dict)



scint_layer_val=np.linspace(3,5,15)
dielectric_layer_val=np.linspace(3,5,15)
M1, M2 = np.meshgrid(scint_layer_val, dielectric_layer_val)
emission_yield_mat_autencoder=np.zeros(M1.shape)
farfield_signal_mat_autencoder=np.zeros(M1.shape)

emission_yield_mat_priorlayer=np.zeros(M1.shape)
farfield_signal_mat_priorlayer=np.zeros(M1.shape)

emission_yield_mat_simulation=np.zeros(M1.shape)
farfield_signal_mat_simulation=np.zeros(M1.shape)
emission_yield_mat_pure_exponential=np.zeros(M1.shape)
farfield_signal_mat_pure_exponential=np.zeros(M1.shape)
layer_type=[i%2 for i in range(num_of_layers)]
M1=np.expand_dims(M1, axis=0)
M2=np.expand_dims(M2, axis=0)
mat_cat=M1
for i in range(num_of_layers-1):
    if i%2==0:
        mat_cat=np.concatenate([mat_cat, M2], axis=0)
    else:
        mat_cat=np.concatenate([mat_cat, M1], axis=0)
for i in range(emission_yield_mat_priorlayer.shape[0]):
    for j in range(emission_yield_mat_priorlayer.shape[1]):
        emission_distribution, emission_theta, emission_phi =predict_output_model(autoencoder_model,mat_cat[:,i,j],layer_type,z_bins)
        emission_yield_mat_autencoder[i,j]=np.sum(emission_distribution.detach().numpy())
        farfield_signal_mat_autencoder[i,j]=torch.sum(torch.abs(emission_theta[1][torch.abs(emission_theta[0])<0.4])).detach().numpy()

        emission_distribution, emission_theta, emission_phi =predict_output_model(prior_layer_model,mat_cat[:,i,j],layer_type,z_bins)
        emission_yield_mat_priorlayer[i,j]=np.sum(emission_distribution.detach().numpy())
        farfield_signal_mat_priorlayer[i,j]=torch.sum(torch.abs(emission_theta[1][torch.abs(emission_theta[0])<0.4])).detach().numpy()

        emission_distribution, emission_theta, emission_phi =predict_output_model_exponential(mat_cat[:,i,j],layer_type)
        emission_yield_mat_pure_exponential[i,j]=emission_distribution
        farfield_signal_mat_pure_exponential[i,j]=torch.sum(torch.abs(emission_theta[1][torch.abs(emission_theta[0])<0.4])).detach().numpy()
        
        
        layer_vec_numpy=mat_cat[:,i,j]#.detach().numpy()
        yield_ph, emission_theta, emission_phi =simulate_emission_angular(layer_vec_numpy,layer_vec_numpy[1::2],layer_vec_numpy[::2])
        emission_yield_mat_simulation[i,j]=yield_ph
        farfield_signal_mat_simulation[i,j]=np.sum(np.abs(emission_theta[1][np.abs(emission_theta[0])<0.4]))


        print(f'finished epoch {i} iteration {j} ')

"""
plt.figure(figsize=(15, 10))
plt.subplot(2,3,1)
#plt.imshow(emission_yield_mat_autencoder)
im = plt.imshow(
    emission_yield_mat_autencoder, 
    aspect='auto', # Use 'auto' to fit the axes
    origin='lower', # Plot origin at bottom-left
    cmap='inferno',# Choose a suitable colormap
    extent=[scint_layer_val[0],scint_layer_val[-1],
        dielectric_layer_val[0],dielectric_layer_val[-1]]
)

# --- 3. Add Labels and Colorbar ---

plt.colorbar(im, label='Loss Metric (Log Scale)')
plt.title('emission_yield autoencoder')

plt.subplot(2,3,4)
#plt.imshow(farfield_signal_mat_autencoder)
im = plt.imshow(
    farfield_signal_mat_autencoder, 
    aspect='auto', # Use 'auto' to fit the axes
    origin='lower', # Plot origin at bottom-left
    cmap='viridis',# Choose a suitable colormap
    extent=[scint_layer_val[0],scint_layer_val[-1],
        dielectric_layer_val[0],dielectric_layer_val[-1]]
)
plt.colorbar(im, label='Loss Metric (Log Scale)')
plt.title('farfield autoencoder')

plt.subplot(2,3,2)
#plt.imshow(emission_yield_mat_priorlayer)
im = plt.imshow(
    emission_yield_mat_priorlayer, 
    aspect='auto', # Use 'auto' to fit the axes
    origin='lower', # Plot origin at bottom-left
    cmap='inferno',# Choose a suitable colormap
    extent=[scint_layer_val[0],scint_layer_val[-1],
        dielectric_layer_val[0],dielectric_layer_val[-1]]
)
# --- 3. Add Labels and Colorbar ---

plt.colorbar(im, label='Loss Metric (Log Scale)')

plt.title('emission_yield layerprior')

plt.subplot(2,3,5)
#plt.imshow(farfield_signal_mat_priorlayer)
im = plt.imshow(
    farfield_signal_mat_priorlayer, 
    aspect='auto', # Use 'auto' to fit the axes
    origin='lower', # Plot origin at bottom-left
    cmap='viridis',# Choose a suitable colormap
    extent=[scint_layer_val[0],scint_layer_val[-1],
        dielectric_layer_val[0],dielectric_layer_val[-1]]
)
plt.colorbar(im, label='Loss Metric (Log Scale)')
plt.title('farfield layerprior')


plt.subplot(2,3,3)
#plt.imshow(farfield_signal_mat_priorlayer)
im = plt.imshow(
    emission_yield_mat_simulation, 
    aspect='auto', # Use 'auto' to fit the axes
    origin='lower', # Plot origin at bottom-left
    cmap='inferno',# Choose a suitable colormap
    extent=[scint_layer_val[0],scint_layer_val[-1],
        dielectric_layer_val[0],dielectric_layer_val[-1]]
)
plt.colorbar(im, label='Loss Metric (Log Scale)')
plt.title('yield simulation')
plt.subplot(2,3,6)
#plt.imshow(farfield_signal_mat_priorlayer)
im = plt.imshow(
    farfield_signal_mat_simulation, 
    aspect='auto', # Use 'auto' to fit the axes
    origin='lower', # Plot origin at bottom-left
    cmap='viridis',# Choose a suitable colormap
    extent=[scint_layer_val[0],scint_layer_val[-1],
        dielectric_layer_val[0],dielectric_layer_val[-1]]
)
plt.colorbar(im, label='Loss Metric (Log Scale)')
plt.title('farfield simulation')
plt.savefig('model_thickness_sweep.svg')
"""
save_data={'autoencoder_':[emission_yield_mat_autencoder,farfield_signal_mat_autencoder],'pure_exponential':[emission_yield_mat_pure_exponential,farfield_signal_mat_pure_exponential],'layerprior':[emission_yield_mat_priorlayer,farfield_signal_mat_priorlayer],'simulation':[emission_yield_mat_simulation,farfield_signal_mat_simulation]}
with open('los_function_visualisation.pkl', 'wb') as file:
        pickle.dump(save_data, file)