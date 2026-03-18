from ScintCityPython.TPMRS_single_emitter import calculating_emmision
import torch.optim as optim
from torch.optim import Optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import median_filter
import os
from skimage.metrics import mean_squared_error
from ScintCityPython.NN_models import Model1, Model2
class PositiveOptimizer(Optimizer):
    def __init__(self, params, lr=1):
        defaults = dict(lr=0.001)
        super(PositiveOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        # Perform a single optimization step
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
                
                # Clamp the parameters to be positive
                p.data.clamp_(min=1.0)
        
        return loss
class E2E(nn.Module):
    def __init__(self,dist_model,mod, sim_model, layer_vec, starts_with_scintilator,num_of_layers):
        super(E2E, self).__init__()  # Initialize the parent class

        # Convert layer_vec to a torch.nn.Parameter
        self.layer_vec = nn.Parameter(torch.tensor(layer_vec, dtype=torch.float64))
        self.mod=mod
        self.dist=dist_model
        self.starts_with_dielectric = 1 - starts_with_scintilator
        self.z_bins = np.linspace(0, np.sum(layer_vec), 100)
        self.starts_with_scintilator = starts_with_scintilator
        self.bin_layer_vec = create_boolean_width(
            self.z_bins,
            self.layer_vec[self.starts_with_dielectric::2].detach().numpy(),
            self.layer_vec[self.starts_with_scintilator::2].detach().numpy(),
            self.starts_with_scintilator
        )
        self.model = sim_model
        self.num_of_layers=num_of_layers
    def set_binary_vec(self):
        self.bin_layer_vec = create_boolean_width(
            self.z_bins,
            self.layer_vec[self.starts_with_dielectric::2].detach().numpy(),
            self.layer_vec[self.starts_with_scintilator::2].detach().numpy(),
            self.starts_with_scintilator
        )

    def forward(self):
        bin_layer_vec,binned_layers_index=self.bin_layer_vec

        if self.dist != 'exponential_optim':
            if self.mod =='2':
                emmsion_density_profile=emmsion_density_profile[0,0,:,:]
            elif self.mod=='4':
                emmsion_density_profile = self.model(torch.tensor(bin_layer_vec).double(),cell_size=torch.tensor(torch.sum(self.layer_vec)/100).double())
            elif self.mod=='5':
                emmsion_density_profile = self.model(torch.tensor(bin_layer_vec).double(),thickness=torch.tensor(torch.sum(self.layer_vec)/100).double())
            emmsion_density_profile = np.abs(emmsion_density_profile.detach().numpy())
        else:
            emmsion_density_profile=0
        reg_layer_vec = self.layer_vec
        #bin_layer_vec, binned_layers_index = self.bin_layer_vec
        absorption_scint=3.95e4 # [nm]]
        absorption_other=4.22e6#[nm]
        #dist='exponential_optim'
        #dist='optimisation'
        if int(bin_layer_vec[0])<0:
            return
        control = {
            'is_Gz': 1,
            'dz_Nz': 10,
            'distribution':self.dist,
            'plot_profile': 0,
            'load_data_from_dir': 0,
            'binary vector': bin_layer_vec,
            'distribution_map': emmsion_density_profile,
            'layer_vector': reg_layer_vec,
            'start_with_scint': int(bin_layer_vec[0]),
            'binary_indexes':binned_layers_index[1:],
            'num_of_layers':self.num_of_layers,
            'absorption_scint': absorption_scint,
		    'absorption_other': absorption_other,
        }
       
    
        emission_distribution, emission_theta, emission_phi = calculating_emmision(
            control, layer_struct='optimisation'
        )
        emission_theta=[torch.tensor(np.real(emission_theta[0])),torch.real(emission_theta[1])]
        emission_phi=[torch.tensor(np.real(emission_phi[0])),torch.real(emission_phi[1])]
        return emmsion_density_profile, emission_theta,emission_phi
def generate_rand_length_vec(nLayersNS,sample_size,low=100,high=300):
    scintillator_list=[]
    dialectric_list=[]
    starts_with_scintilator=np.random.choice([0, 1], size=1, p=[0.5, 0.5])[0]
    #starts_with_scintilator=1
    for layer in range (nLayersNS):
        layer_width=np.random.uniform(low, high)
        if layer%2==starts_with_scintilator:
            dialectric_list.append(layer_width)
        else:
            scintillator_list.append(layer_width)
    dialectric_list=np.array(dialectric_list)
    scintillator_list=np.array(scintillator_list)
    thickness=np.zeros(nLayersNS)
    thickness[starts_with_scintilator::2]=dialectric_list
    thickness[1-starts_with_scintilator::2]=scintillator_list
    return thickness,starts_with_scintilator
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
def optimise(dist,mod,model,num_epochs,num_of_ini_points=1,optimisation_flag='constrained',plot_loss=False,num_of_layers=10,lr=1e-1,relative_wiehgt=1e-3,width=2000):
    if plot_loss:
        plt.figure()

    for ini_point in range(num_of_ini_points):
        min_err=10
        begin_flag=True
        ini_layer_vec,starts_with_scintilator=generate_rand_length_vec(num_of_layers,width)
        #ini_layer_vec=width*ini_layer_vec/np.sum(ini_layer_vec)
        ini_layer_vec=np.array(np.linspace(5,10,num_of_layers))
        ini_layer_vec=width*ini_layer_vec/np.sum(ini_layer_vec)
        starts_with_scintilator=0
        #dist='exponential_optim'
        #dist='optimisation'
        E2Eoptim = E2E(dist,mod,model, ini_layer_vec, starts_with_scintilator,num_of_layers)
        
        if optimisation_flag=='regular':
            optimizer = optim.Adam([E2Eoptim.layer_vec], lr=lr)
        elif optimisation_flag=='constrained':
            optimizer =PositiveOptimizer([E2Eoptim.layer_vec], lr=lr)
        loss_arr = []
        accuracy_arr = []
        opt_stat_scint=1
        theta_cutoff = 0.5

        try:
            for epoch in range(num_epochs):
                E2Eoptim.train()
                optimizer.zero_grad()
                emission_distribution, emission_theta, emission_phi = E2Eoptim()  # assuming no inputs are needed
                sample_size=torch.sum(E2Eoptim.layer_vec) 
                yield_ph=np.sum(emission_distribution)
                if (yield_ph==0):
                    print(yield_ph)
                print(f'photon yield is {yield_ph}')
                print(f'size error is {(width-sample_size)**2}')
                selected_elements = emission_theta[1][(emission_theta[0] > -theta_cutoff) & (emission_theta[0] < theta_cutoff)]
                #loss = -1e-3*yield_ph*(torch.sum(selected_elements)/torch.sum(emission_theta[1]))+(width-sample_size)**2
                loss = -relative_wiehgt*(torch.sum(selected_elements))+(width-sample_size)**2
                loss.backward()
                optimizer.step()
                loss_arr.append(loss.item())
                E2Eoptim.set_binary_vec()
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_arr[-1]:.4f}')
                print(f'layers: {E2Eoptim.layer_vec}')
            if plot_loss:
                plt.plot(loss_arr,label=f'interation number {ini_point}')
            if loss_arr[-1]<min_err or begin_flag:
                begin_flag=False
                min_err=loss_arr[-1]
                min_vec=E2Eoptim.layer_vec.detach().numpy()
                opt_stat_scint=starts_with_scintilator
        except:
            print ('final epoch is '+str(epoch))
    if plot_loss:
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.show()
    print(f'most optimal vec is {min_vec}')
    print(f'sws bit is {opt_stat_scint}')    
    print(f'min error is {min_err}')           
    print(f'most optimal vector is {min_vec} \n stat_with_scintilator {opt_stat_scint}')
    return opt_stat_scint,min_vec,min_err
    # Model loading and training setup
def temperature_schedule(epoch, T0=1.0, alpha=0.95):
    return T0 * np.exp(-epoch/1e4)
def simulated_annealing(dist,mod,model, starts_with_scintilator,num_of_layers, width, relative_weight, num_epochs, theta_cutoff=0.5, plot_loss=False):
    #ini_layer_vec,tmp=generate_rand_length_vec(num_of_layers,width)
    #ini_layer_vec=width*ini_layer_vec/np.sum(ini_layer_vec)
    ini_layer_vec=np.array(np.linspace(5,10,num_of_layers))
    ini_layer_vec=width*ini_layer_vec/np.sum(ini_layer_vec)
    starts_with_scintilator=0
    E2Eoptim = E2E(dist,mod,model, ini_layer_vec, starts_with_scintilator,num_of_layers)
    current_vec = E2Eoptim.layer_vec.clone()#.detach().requires_grad_(False)
    best_vec = current_vec.clone()
    min_loss = float('inf')
    loss_arr = []

    def compute_loss(layer_vec):
        E2Eoptim.layer_vec.data = layer_vec.clone().detach()
        emission_distribution, emission_theta, emission_phi = E2Eoptim()
        sample_size = torch.sum(layer_vec)
        selected_elements = emission_theta[1][(emission_theta[0] > -theta_cutoff) & (emission_theta[0] < theta_cutoff)]
        loss = -relative_weight * (torch.sum(selected_elements)) + (width - sample_size) **2 +1e-2/torch.min(layer_vec)
        return loss.item()

    current_loss = compute_loss(current_vec)

    for epoch in range(num_epochs):
        T = temperature_schedule(epoch)

        # Create a new candidate by perturbing the current vector
        perturbation = torch.randn_like(current_vec) * 10
        candidate_vec = current_vec + perturbation

        # Enforce positivity
        candidate_vec = torch.clamp(candidate_vec, min=1e-6)

        # Enforce constraint: normalize to keep total width
        candidate_vec = width * candidate_vec / torch.sum(candidate_vec)

        candidate_loss = compute_loss(candidate_vec)

        # Acceptance criteria
        delta_loss = candidate_loss - current_loss
        if delta_loss < 0 or np.random.rand() < np.exp(-delta_loss / T):
            current_vec = candidate_vec
            current_loss = candidate_loss

        if current_loss < min_loss:
            min_loss = current_loss
            best_vec = current_vec.clone()

        loss_arr.append(current_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {current_loss:.4f}, Temp: {T:.4f}")

    if plot_loss:
        plt.plot(loss_arr, label="Simulated Annealing Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SA Optimization Loss")
        plt.legend()
        plt.show()
   
    return E2Eoptim.starts_with_scintilator,best_vec.detach().numpy(), min_loss

def main():
    mod = '2'
    loss_arr = []
    #optimisation_flag='regular'
    optimisation_flag='constrained'
    #weights_path = r"C:\Users\nathan\OneDrive - Technion\Desktop\technion\7\projectB\ScintCityPython\20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model2_parameters.pth"
    #weights_path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model2results/240417105148/20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model2_parameters.pth'
    weights_path=r"/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2_model2results/250105130503/20250102154707541738Database_5000samples_nonpriodic_sim_100000_photons_10layers_PbOSiO2_model2_parameters.pth"
    state_dict = torch.load(weights_path)
    if mod == '1':
        model = Model1(state_dict['fc1.weight'].shape[1], state_dict['fc2.weight'].shape[1], state_dict['fc2.weight'].shape[0])
    elif mod == '2':
        model = Model2(99, 99, 99 * 99)
    model.load_state_dict(state_dict)

    d_scint = 200
    d_dielectric = 200


    ini_layer_vec = np.array([d_scint, d_dielectric, d_scint, d_dielectric, d_scint, d_dielectric])
    starts_with_scintilator = 0
    lr_list=[1e3]
    lr=1
    num_epochs = int(200)
    num_of_ini_points=1
    min_vec=np.zeros(6)

    # Use torch optimizer to optimize layer_vec
    #dist='exponential_optim'
    dist='optimisation'
    optimise(dist,mod,model,num_epochs,num_of_ini_points=1,optimisation_flag='constrained',plot_loss=False)




