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
                p.data.clamp_(min=0.1)
        
        return loss
    
class E2E_old(nn.Module):
    def __init__(self,dist_model,mod, sim_model, layer_vec, starts_with_scintilator,num_of_layers):
        super(E2E_old, self).__init__()  # Initialize the parent class

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
            emmsion_density_profile = self.model(torch.tensor(bin_layer_vec).double(),cell_size=torch.tensor(torch.sum(self.layer_vec)/100).double())
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
class E2E_exponential(nn.Module):
    def __init__(self, layer_vec, starts_with_scintilator):
        super(E2E_exponential, self).__init__()  # Initialize the parent class

        # Convert layer_vec to a torch.nn.Parameter
        self.layer_vec = nn.Parameter(torch.tensor(layer_vec, dtype=torch.float64))
        self.starts_with_dielectric = 1 - starts_with_scintilator
        self.starts_with_scintilator = starts_with_scintilator
        self.num_of_layers=layer_vec.size(0)
    def forward(self):
        #bin_layer_vec, binned_layers_index = self.bin_layer_vec
        absorption_scint=3.95e4 # [nm]]
        absorption_other=4.22e6#[nm]
        #dist='exponential_optim'
        #dist='optimisation'
        control = {
            'is_Gz': 1,
            'dz_Nz': 10,
            'distribution':'exponential_optim',
            'plot_profile': 0,
            'load_data_from_dir': 0,
            #'binary vector': bin_layer_vec,
            #'distribution_map': emmsion_density_profile,
            'layer_vector': 1e2*self.layer_vec,
            'start_with_scint':   self.starts_with_scintilator,
            #'binary_indexes':binned_layers_index[1:],
            'num_of_layers':self.num_of_layers,
            'absorption_scint': absorption_scint,
		    'absorption_other': absorption_other,
        }
       
    
        emission_distribution, emission_theta, emission_phi = calculating_emmision(
            control, layer_struct='optimisation'
        )
        emission_theta=[torch.tensor(np.real(emission_theta[0])),torch.real(emission_theta[1])]
        emission_phi=[torch.tensor(np.real(emission_phi[0])),torch.real(emission_phi[1])]
        return  emission_distribution,emission_theta,emission_phi





class E2E_continous(nn.Module):
    def __init__(self,sim_model,ini_layer_vec,z_bins_num=100):
        super(E2E_continous, self).__init__()  # Initialize the parent class
        self.model = sim_model
        self.layer_vec=nn.Parameter(ini_layer_vec)
        self.z_bins_num=z_bins_num
    def forward(self, layer_type,batch_size = 500):
        batched_layer_vec = self.layer_vec.unsqueeze(0).expand(batch_size, -1)
        batched_layer_type=layer_type.unsqueeze(0).expand(batch_size, -1)
        emmsion_density_profile_batch = self.model(batched_layer_vec, batched_layer_type).double()
        emmsion_density_profile = torch.mean(emmsion_density_profile_batch, dim=0)
        z_bins = torch.linspace(0, torch.sum(self.layer_vec), self.z_bins_num)
        """
        bin_layer_vec,binned_layers_index = create_boolean_width(
        z_bins,
        self.layer_vec[1-layer_type[0]::2],
        self.layer_vec[layer_type[0]::2],
        layer_type[0]
        )
        """
        
        bin_layer_vec,binned_layers_index = create_boolean_width_vectorized(
        z_bins,
        self.layer_vec[1-layer_type[0]::2],
        self.layer_vec[layer_type[0]::2],
        layer_type[0]
        )
        absorption_scint=3.95e4 # [nm]]
        absorption_other=4.22e6#[nm]
        control = {
            'is_Gz': 1,
            'dz_Nz': 10,
            'distribution':'optimisation',
            'plot_profile': 0,
            'load_data_from_dir': 0,
            'binary vector': bin_layer_vec,
            'distribution_map': emmsion_density_profile,
            'layer_vector': 1e2*self.layer_vec,
            'start_with_scint': int(layer_type[0]),
            'binary_indexes':binned_layers_index,
            'num_of_layers':layer_type.size()[0],
            'absorption_scint': absorption_scint,
		    'absorption_other': absorption_other,
        }
       
    
        emission_distribution, emission_theta, emission_phi = calculating_emmision(
            control, layer_struct='optimisation'
        )
        emission_theta=[torch.tensor(torch.real(emission_theta[0])),torch.sum(emmsion_density_profile)*torch.real(emission_theta[1])]
        emission_phi=[torch.tensor(torch.real(emission_phi[0])),torch.sum(emmsion_density_profile)*torch.real(emission_phi[1])]
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
    total_list = torch.empty(scint.size()[0] + scint.size()[0], dtype=scint.dtype)
    total_list[(1 - starts_with_scintilator)::2] = scint
    total_list[starts_with_scintilator::2] = dialect
    boolean_list = torch.zeros(z_bins.size()[0] - 1)
    acumulated_bins_depth = 0
    layer_counter = 0
    partial_sum_of_layers = total_list[0]
    z_diff = z_bins[1] - z_bins[0]
    binned_layers_index = []  # indexing where layers start and end
    for i in range(boolean_list.size()[0]):
        if (z_diff + acumulated_bins_depth) < partial_sum_of_layers:  # next bin is within the layer
            boolean_list[i] = (layer_counter + starts_with_scintilator) % 2
        else:
            if layer_counter < (total_list.size()[0] - 1):
                layer_counter += 1
                partial_sum_of_layers += total_list[layer_counter]
           
            boolean_list[i] = -1  # interface between layers
        if torch.abs(boolean_list[i]) - torch.abs(boolean_list[i - 1]) != 0:  # if layers change then need to append the value
            binned_layers_index.append(i)
        acumulated_bins_depth += z_diff
    if binned_layers_index[-1] < i and torch.tensor(binned_layers_index).size()[0]<8:
        binned_layers_index.append(i)
    binned_layers_index=binned_layers_index[1:]
    return boolean_list, torch.tensor(binned_layers_index)
def create_boolean_width_numpy(z_bins, scint, dialect, starts_with_scintilator):
    """
    Creates a boolean vector of layer type (0 or 1) for each z-bin
    and returns indices of layer boundaries (interfaces) using NumPy.

    Parameters:
    - z_bins: np.ndarray, (N_bins) Grid points (position)
    - scint: np.ndarray, (M) Thicknesses of scintillator layers
    - dialect: np.ndarray, (M) Thicknesses of dielectric layers
    - starts_with_scintilator: int (0 or 1)

    Returns:
    - boolean_list: np.ndarray (N_bins - 1) 0 for dielectric, 1 for scintillator, -1 for interface
    - binned_layers_index: np.ndarray (Indices of layer boundaries)
    """
    
    # Ensure inputs are float for calculation stability
    scint = scint.astype(np.float64)
    dialect = dialect.astype(np.float64)
    z_bins = z_bins.astype(np.float64)

    # 1. CONSTRUCT THE FULL LAYER VECTOR (Total Thicknesses)
    
    # Create the total list template (size 2M)
    total_list_scint = np.empty(scint.size + dialect.size, dtype=np.float64)
    
    # Place scintillator and dielectric thicknesses correctly based on the start flag
    total_list_scint[starts_with_scintilator::2] = scint
    total_list_scint[1 - starts_with_scintilator::2] = dialect

    # 2. DETERMINE THE GRID BOUNDARIES AND LAYER INDICES
    
    # Calculate the cumulative depth of the layers (layer boundaries)
    layer_boundaries = np.cumsum(total_list_scint)

    # Calculate the midpoints of the z-bins
    z_midpoints = (z_bins[:-1] + z_bins[1:]) / 2.0
    
    # Assign each z-midpoint to a layer index using np.searchsorted
    # The result (layer_indices) is the index of the layer boundary it falls short of.
    layer_indices = np.searchsorted(layer_boundaries, z_midpoints, side='right')

    # 3. CREATE THE BOOLEAN VECTOR (Type and Interfaces)
    
    # Calculate the layer type (0 or 1) based on the layer index and the start flag
    # layer_indices starts from 0 (index before first boundary), so (index + starts_with_scintilator) % 2 works
    boolean_list = (layer_indices + starts_with_scintilator) % 2
    
    # 4. IDENTIFY LAYER INTERFACES (Position where index changes)

    # Calculate the difference between adjacent layer indices.
    # We prepend -1 to ensure the first index change (from -1 to 0) is captured.
    index_diff = np.diff(layer_indices.astype(np.float64), prepend=-1.0)
    
    # The mask identifies where the layer index changed (i.e., boundary crossed)
    boundary_mask = (index_diff != 0)
    
    # Get the indices where the mask is True (the layer changes)
    binned_layers_index = np.where(boundary_mask)[0]

    # 5. MARK INTERFACE POINTS (Optional, but matches original logic)

    # The interface occurs at the bin *before* the index crossing (excluding the first index)
    interface_indices = binned_layers_index[1:] - 1 
    
    # Set the interface points to -1
    # We need to handle potential out-of-bounds access if interface_indices is empty.
    if interface_indices.size > 0:
        # Ensure boolean_list is float/int type to hold -1
        boolean_list = boolean_list.astype(np.int64) 
        boolean_list[interface_indices] = -1

    return boolean_list, binned_layers_index
def create_boolean_width_vectorized(z_bins, scint, dialect, starts_with_scintilator):
    """
    Creates a boolean vector of layer type (0 or 1) for each z-bin
    and returns indices of layer boundaries (interfaces).

    Parameters:
    - z_bins: torch.Tensor, (N_bins) Grid points (position)
    - scint: torch.Tensor, (M) Thicknesses of scintillator layers
    - dialect: torch.Tensor, (M) Thicknesses of dielectric layers
    - starts_with_scintilator: int (0 or 1)

    Returns:
    - boolean_list: torch.Tensor (N_bins - 1) 0 for dielectric, 1 for scintillator, -1 for interface
    - binned_layers_index: torch.Tensor (Indices of layer boundaries)
    """
    
    # 1. CONSTRUCT THE FULL LAYER VECTOR (Total Thicknesses)
    # Total vector size: 2M. Use torch.stack for efficient placement.
    total_list_scint = torch.empty(scint.size(0) + dialect.size(0), dtype=scint.dtype, device=scint.device)
    
    # Place scintillator and dielectric thicknesses correctly based on the start flag
    total_list_scint[starts_with_scintilator::2] = scint
    total_list_scint[1 - starts_with_scintilator::2] = dialect

    # 2. DETERMINE THE GRID BOUNDARIES AND LAYER INDICES
    
    # Calculate the cumulative depth of the layers (layer boundaries)
    # Shape: (2M)
    layer_boundaries = torch.cumsum(total_list_scint, dim=0)

    # Calculate the midpoints of the z-bins (where we determine the layer type)
    # Shape: (N_bins - 1)
    z_midpoints = (z_bins[:-1] + z_bins[1:]) / 2.0
    
    # Assign each z-midpoint to a layer index.
    # The result is a tensor (N_bins - 1) where each element holds the index of the
    # layer boundary it falls short of.
    # The complexity is reduced from O(N_bins * N_layers) to O(N_bins * log(N_layers)).
    # layer_indices = (N_layers + 1) layer 0, layer 1, layer 2...
    layer_indices = torch.searchsorted(layer_boundaries, z_midpoints, right=True)

    # 3. CREATE THE BOOLEAN VECTOR (Type and Interfaces)
    
    # Layer type (0 or 1): The layer index mod 2 gives the type
    # starts_with_scintilator=0 (dielectric) -> index 0=dialect (0), 1=scint (1), 2=dialect (0)
    # starts_with_scintilator=1 (scint) -> index 0=scint (1), 1=dialect (0), 2=scint (1)
    # The actual layer index is (layer_indices - 1)
    boolean_list = (layer_indices + starts_with_scintilator) % 2
    
    # 4. IDENTIFY LAYER INTERFACES (Position where index changes)

    # Shift the layer index vector to compare adjacent bins: (N_bins - 1)
    # We use padding to align the tensors for comparison.
    index_diff = torch.diff(layer_indices.float(), prepend=torch.tensor([-1.0], device=z_midpoints.device))
    
    # Where index_diff is non-zero, a layer boundary was crossed.
    # The indices of these boundary crossings are the binned_layers_index.
    boundary_mask = (index_diff != 0)
    
    # Get the indices where the mask is True (the layer changes)
    binned_layers_index = torch.where(boundary_mask)[0]
    
    # The interface should be marked as -1 (this step is optional but matches your original request)
    # The boundary occurs at the start of the new layer, so we check the index of the boundary_mask
    interface_indices = binned_layers_index[1:] - 1 # Indices of the interface points
    boolean_list[interface_indices] = -1
    last_index_value = z_bins.size(0)-1

    # The line to append the last index:
    binned_layers_index = torch.cat([
        binned_layers_index, 
        torch.tensor([last_index_value], dtype=binned_layers_index.dtype, device=binned_layers_index.device)
    ])
    return boolean_list, binned_layers_index
def optimise(model,num_epochs,ini_layer_vec,starts_with_scintilator,optimisation_flag='constrained',plot_loss=False,num_of_layers=10,lr=1e-1,optim_type_flag='yield',width=2,z_bins_num=100):
    model_name=model.get_name()
    min_err=10
    
    begin_flag=True

    layer_type=torch.tensor([(starts_with_scintilator+i)%2 for i in range(num_of_layers)])
    #dist='exponential_optim'
    #dist='optimisation'
    if model_name=='Model1_thick':
        E2Eoptim = E2E_exponential(torch.tensor(ini_layer_vec), starts_with_scintilator)
    else:
        E2Eoptim = E2E_continous(model,torch.tensor(ini_layer_vec),z_bins_num=z_bins_num)
    
    if optimisation_flag=='regular':
        optimizer = optim.Adam([E2Eoptim.layer_vec], lr=lr)
    elif optimisation_flag=='constrained':
        optimizer =PositiveOptimizer([E2Eoptim.layer_vec], lr=lr)
    loss_arr = []
    accuracy_arr = []
    opt_stat_scint=0
    theta_cutoff = 0.2
    for epoch in range(num_epochs):
        E2Eoptim.train()
        optimizer.zero_grad()
        if model_name=='Model1_thick' :
            emission_distribution, emission_theta, emission_phi = E2Eoptim()
        else:
            emission_distribution, emission_theta, emission_phi = E2Eoptim(layer_type)  
            yield_ph=torch.sum(emission_distribution)
        
        scint_sample_size=torch.sum(torch.abs(E2Eoptim.layer_vec*layer_type)) 
        sample_size=torch.sum(E2Eoptim.layer_vec) 

        threshold = 0.3  # C
        lambda_barrier = 10000.0 # Adjust this value (e.g., 10 to 1000)

        # Calculate the violation: C - x. This is positive only when x < C.
        violation = threshold - E2Eoptim.layer_vec

        # Apply ReLU to only keep the positive violations (when x is too small)
        positive_violations = torch.relu(violation)
        barrier_loss = lambda_barrier * torch.sum(positive_violations ** 2)
        broken_layers=torch.sum(E2Eoptim.layer_vec<0.2) 
        scint_size_error= (width/2-scint_sample_size)**2
        #sample_size_error=(width-sample_size)**2
        print(f'size error is {scint_size_error}')
        #print(f'size error is {sample_size_error}')

        
        selected_elements = emission_theta[1][(emission_theta[0] > -theta_cutoff) & (emission_theta[0] < theta_cutoff)]
        if model_name=='Model1_thick':
            emission_loss=-(torch.sum(selected_elements))
        elif optim_type_flag=='yield':
            emission_loss=-1e-5*yield_ph#*(torch.sum(sel#cted_elements))
        else:
            emission_loss=-1e-5*yield_ph*(torch.sum(selected_elements))
        loss = emission_loss+barrier_loss+scint_size_error #
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # Use .clamp_ (in-place) to set any negative value to 0
            E2Eoptim.layer_vec.clamp_(min=threshold)
        loss_arr.append(loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_arr[-1]:.4f} ')
        print(f'layers: {E2Eoptim.layer_vec}')
    plt.figure()
    plt.plot(emission_theta[0].detach().numpy() ,emission_theta[1].detach().numpy(),'.' )
    plt.savefig('estimated_emission_pattern.png')
    plt.close()
    if loss_arr[-1]<min_err or begin_flag:
        begin_flag=False
        min_err=loss_arr[-1]
        min_vec=E2Eoptim.layer_vec.detach().numpy()
        opt_stat_scint=starts_with_scintilator
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




