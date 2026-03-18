import sys
sys.path.append('/home/nathan.regev/software/git_repo/')
import ScintCityPython.END2END_optim_v02 as opt_block
from ScintCityPython.TPMRS_single_emitter import calculating_emmision
import G4_Nanophotonic_Scintillator.utils.generate_database as geant4
import numpy as np
import torch
import matplotlib.pyplot as plt
from ScintCityPython.NN_models import CompleteAutoenconder_Model,PriorLayerModel,Model1_thick
import pickle
import os
import pandas as pd
from datetime import datetime
def simulate_emission_angular(layer_vec,scintillator_list,dialectric_list):
    simulation_scint_list=1e-4*scintillator_list
    simulation_dialectric_list=1e-4*dialectric_list

    propegation_scint_list=1e2*scintillator_list
    propegation_dialectric_list=1e2*dialectric_list
    N=1
    nLayersNS=10
    sim_photons=100000
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
    """
    plt.plot()
    plt.imshow(grid)
    plt.colorbar()
    plt.savefig(f'{mod} : grid optimized')
    plt.close()
    """
    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(saving_dict, f)

    #simulate actual photon propegation 
    absorption_scint=3.95e4 # [nm]]
    absorption_other=4.22e6#[nm]
    """
    control_simulation = {
        'is_Gz': 1,
        'dz_Nz': 10,
        'distribution': 'simulation',
        'simulation_path':filename+'.pkl',
        'absorption_scint': absorption_scint,
        'absorption_other': absorption_other,
        'plot_profile': 0,
        'load_data_from_dir': 1,
        'num_of_layers':nLayersNS
    }
    sim_emmsioion_ditribution,sim_emission_theta,sim_emission_phi=calculating_emmision(control_simulation)
    """
    control= {
            'is_Gz': 1,
            'dz_Nz': 10,
            'distribution':'optimisation',
            'plot_profile': 0,
            'load_data_from_dir': 0,
            'binary vector': torch.tensor(binned_layesr),
            'distribution_map': torch.tensor(grid),
            'layer_vector': 1e2*torch.tensor(layer_vec),
            'start_with_scint': 1,
            'binary_indexes':torch.tensor(binned_layers_index),
            'num_of_layers':10,
            'absorption_scint': absorption_scint,
		    'absorption_other': absorption_other,
        }
    
    sim_emmsioion_ditribution,sim_emission_theta,sim_emission_phi= calculating_emmision(
        control, layer_struct='optimisation'
    )
    print(len(ph_x))
    sim_emission_theta[1]=(len(ph_x)**2)*sim_emission_theta[1].detach().numpy()
    sim_emission_phi[1]=(len(ph_x)**2)*sim_emission_phi[1].detach().numpy()
    os.remove(filename+'.pkl')
    return sim_emission_theta
def get_opt_struct_and_dist(
                        mod='autoencoder',
                        num_of_layers=10,
                        width=2,
                        lr=5e-3,
                        num_epochs = 2000,
                        z_bins=1000,r_bins=100,
                        layer_structure=None
                        ):
    input_size=99
    optimisation_flag='regular'
    if torch.cuda.is_available():
         # If a GPU is available, select it (typically 'cuda:0')
        device = torch.device('cuda')
    else:
        # Otherwise, fall back to the CPU
        device = torch.device('cpu')
    if mod=='autoencoder':
        model = CompleteAutoenconder_Model( nlayers=num_of_layers,kparam=10, z_dim=z_bins-1, r_dim=r_bins-1)
        Predictor_Path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/run_autoencoder_mod4/optimization_model_nonolinear_12_11_2025/results_wieghts_network.pth'
        #Predictor_Path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/run_autoencoder_mod4/251116134846/results_wieghts_network.pth'
        predictor_state_dict = torch.load(Predictor_Path, map_location=device)
        model.FCPred.load_state_dict(predictor_state_dict)
        deconder_Path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/run_autoencoder_mod4/optimization_model_nonolinear_12_11_2025/results_wieghts_decoder.pth'
        #deconder_Path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/run_autoencoder_mod4/251116134846/results_wieghts_decoder.pth'
        decoder_state_dict = torch.load(deconder_Path, map_location=device)
        model.decoder.load_state_dict(decoder_state_dict)
    elif mod=='prior_layer_model':
            model = PriorLayerModel(num_of_layers,99, 99,0.000250)
            prior_layer_model_path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/run_layer_exp_mod4/251117164407/results_wieghts.pth'
            prior_layer_model_dict = torch.load(prior_layer_model_path)
            model.load_state_dict(prior_layer_model_dict)
    
    elif mod=='pure_exponential':
            model = Model1_thick(99,99, 99*99)
            prior_layer_model_path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/thickness_model1/optimization_version_11_11_205/results_wieghts.pth'
            prior_layer_model_dict = torch.load(prior_layer_model_path)
            model.load_state_dict(prior_layer_model_dict)
            #optimisation_flag='constrained'
    else:
        pass
    """
    exp_model = Model1_thick(99,99, 99*99)
    prior_layer_model_path=r'/home/nathan.regev/software/git_repo/G4_Nanophotonic_Scintillator/thickness_model1/optimization_version_11_11_205/results_wieghts.pth'
    prior_layer_model_dict = torch.load(prior_layer_model_path)
    exp_model.load_state_dict(prior_layer_model_dict)
    """
    d_scint = 200
    d_dielectric = 200
    min_vec=np.zeros(num_of_layers)
    ini_layer_vec=torch.ones(num_of_layers)
    if layer_structure is None:
        
        ini_layer_vec=width*ini_layer_vec/torch.sum(ini_layer_vec)
    else:
         ini_layer_vec=torch.tensor(layer_structure)
    starts_with_scintilator=0
    #opt_stat_scint,min_vec,min_err=opt_block.optimise(model,num_epochs,num_of_ini_points=num_of_ini_points,optimisation_flag='regular',plot_loss=False,lr=lr,relative_wiehgt=relative_wiehgt,width=width)
    attempts=2
    opt_stat_scint,min_vec,min_err=opt_block.optimise(model,num_epochs,ini_layer_vec,starts_with_scintilator,optimisation_flag=optimisation_flag,lr=lr,width=width,optim_type_flag='FarField')
    

    nLayersNS=min_vec.size
    dialectric_list=min_vec[opt_stat_scint::2]
    scintillator_list=min_vec[1-opt_stat_scint::2]
    simulation_scint_list=1e-4*scintillator_list
    simulation_dialectric_list=1e-4*dialectric_list

    propegation_scint_list=1e2*scintillator_list
    propegation_dialectric_list=1e2*dialectric_list
    wavelength=430e-6
    N=1
    sim_photons=10000
    bins=100
    max_rad=230e-5
    sample_size=np.sum(simulation_scint_list)+np.sum(simulation_dialectric_list)
    filename='simulation_sim_'+str(sim_photons)+'_photons_'+str(nLayersNS)+'layers'
    scenario=geant4.generate_scenario(simulation_scint_list,simulation_dialectric_list,sim_photons,startsWithScint=opt_stat_scint,nLayersNS=nLayersNS)
    ph_x,ph_y,ph_z=geant4.get_denisty_of_scenario(scenario,filename,database=True,path_pre='G4_Nanophotonic_Scintillator')
    ph_r=np.sqrt(np.power(ph_x,2)+np.power(ph_y,2))
    z_bins=np.linspace(0,sample_size,bins)
    r_bins=np.linspace(0,max_rad,bins)
    binned_layesr,binned_layers_index=opt_block.create_boolean_width_numpy(z_bins,simulation_scint_list,simulation_dialectric_list,opt_stat_scint)
    grid,edges_1,edes_2=np.histogram2d(ph_r,ph_z,bins=(r_bins,z_bins))
    saving_dict={'scintlator':[],'dialectric':[],'binned':[],'grid':[]}
    saving_dict['scintlator'].append(propegation_scint_list)
    saving_dict['dialectric'].append(propegation_dialectric_list)
    saving_dict['binned'].append(binned_layesr)
    saving_dict['grid'].append(grid)
    """
    plt.plot()
    plt.imshow(grid)
    plt.colorbar()
    plt.savefig(f'{mod} : grid optimized')
    plt.close()
    """
    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(saving_dict, f)

    #simulate actual photon propegation 
    absorption_scint=3.95e4 # [nm]]
    absorption_other=4.22e6#[nm]
    """
    control_simulation = {
        'is_Gz': 1,
        'dz_Nz': 10,
        'distribution': 'simulation',
        'simulation_path':filename+'.pkl',
        'absorption_scint': absorption_scint,
        'absorption_other': absorption_other,
        'plot_profile': 0,
        'load_data_from_dir': 1,
        'num_of_layers':nLayersNS
    }
    sim_emmsioion_ditribution,sim_emission_theta,sim_emission_phi=calculating_emmision(control_simulation)
    """
    control= {
            'is_Gz': 1,
            'dz_Nz': 10,
            'distribution':'optimisation',
            'plot_profile': 0,
            'load_data_from_dir': 0,
            'binary vector': torch.tensor(binned_layesr),
            'distribution_map': torch.tensor(grid),
            'layer_vector': 1e2*torch.tensor(min_vec),
            'start_with_scint': opt_stat_scint,
            'binary_indexes':torch.tensor(binned_layers_index),
            'num_of_layers':num_of_layers,
            'absorption_scint': absorption_scint,
		    'absorption_other': absorption_other,
        }
    
    sim_emmsioion_ditribution,sim_emission_theta,sim_emission_phi= calculating_emmision(
        control, layer_struct='optimisation'
    )
    print(len(ph_x))
    sim_emission_theta[1]=(len(ph_x))*sim_emission_theta[1].detach().numpy()
    sim_emission_phi[1]=(len(ph_x))*sim_emission_phi[1].detach().numpy()
    os.remove(filename+'.pkl')
    return min_vec,grid,sim_emission_theta,scintillator_list,dialectric_list
def plot_alternating_rectangles(thicknesses,title, colors=['blue', 'green']):
    """
    Generates an image of rectangles with alternating colors.

    Parameters:
    - thicknesses: A list of thicknesses for each rectangle.
    - colors: A list of two colors to alternate between (default is ['blue', 'green']).
    """
    # Set up the figure and axis
    thicknesses=10*thicknesses/np.sum(thicknesses)
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_xlim(0, 1)  # x-axis goes from 0 to 1
    ax.set_ylim(0, sum(thicknesses))  # y-axis will span the total thickness
    
    current_position = 0  # Start at the bottom of the plot
    
    # Loop through the thicknesses and draw rectangles
    for i, thickness in enumerate(thicknesses):
        color = colors[i % 2]  # Alternate between the two colors
        rect = plt.Rectangle((0, current_position), 1, thickness, color=color)
        ax.add_patch(rect)
        current_position += thickness  # Move the starting position for the next rectangle
    
    # Remove the axis for a cleaner look
    ax.axis('off')
    
    # Display the image
    plt.show()
    plt.savefig('layer_structure '+title)

def optimise_statistics_single_width(width,repetetitions=5,statisitcs=True):
    theta_threshold=0.3
    loss_autoencoder_list=[]
    loss_layerprior_list=[]
    loss_pureexp_list=[]
    for i in range(repetetitions):
        layers=np.random.uniform(3, 5, 10)
        layer_sructure=width*layers/np.sum(layers)
        layer_struct_autoencoder,grid_autoencoder,sim_emission_theta_autoencoder,scintillator_list_autoencoder,dialectric_list_autoencoder=get_opt_struct_and_dist(mod='autoencoder',num_epochs=200,num_of_layers=10,width=width,z_bins=100,r_bins=100,lr=1e-2*width/30,layer_structure=layer_sructure)
        loss_autoencoder=np.sum(np.abs(sim_emission_theta_autoencoder[1][np.abs(sim_emission_theta_autoencoder[0])<theta_threshold]))
        loss_autoencoder_list.append(loss_autoencoder)

        layer_struct_layerprior,grid_layerprior,sim_emission_theta_layerprior,scintillator_list_layerprior,dialectric_list_layerprior=get_opt_struct_and_dist(mod='prior_layer_model',num_epochs=200,num_of_layers=10,width=width,lr=1e-2*width/30,layer_structure=layer_sructure)
        loss_layerprior=np.sum(np.abs(sim_emission_theta_layerprior[1][np.abs(sim_emission_theta_layerprior[0])<theta_threshold]))
        loss_layerprior_list.append(loss_layerprior)

        layer_struct_pureexp,grid_pureexp,sim_emission_theta_pureexp,scintillator_list_pureexp,dialectric_list_pureexp=get_opt_struct_and_dist(mod='pure_exponential',num_epochs=100,num_of_layers=10,width=width,lr=1e-2*width/30,layer_structure=layer_sructure)
        loss_pureexp=np.sum(np.abs(sim_emission_theta_pureexp[1][np.abs(sim_emission_theta_pureexp[0])<theta_threshold]))
        loss_pureexp_list.append(loss_pureexp)

    print(f'autoencoder loss is {np.median(loss_autoencoder_list)} percentile [{np.percentile(loss_autoencoder_list,25)},{np.percentile(loss_autoencoder_list,75)}]')
    print(f'layerprior loss is {np.median(loss_layerprior_list)} percentile [{np.percentile(loss_layerprior_list,25)},{np.percentile(loss_layerprior_list,75)}]')
    print(f'pure exp loss is {np.median(loss_pureexp_list)} percentile [{np.percentile(loss_pureexp_list,25)},{np.percentile(loss_pureexp_list,75)}]')
    if statisitcs:
        return {'autoencoder':[np.median(loss_autoencoder_list),np.percentile(loss_autoencoder_list,25),np.percentile(loss_autoencoder_list,75)],
            'layerprior':[np.median(loss_layerprior_list),np.percentile(loss_layerprior_list,25),np.percentile(loss_layerprior_list,75)],
                'purexp':[np.median(loss_pureexp_list),np.percentile(loss_pureexp_list,25),np.percentile(loss_pureexp_list,75)]}
    else:
        return loss_autoencoder_list,loss_layerprior_list,loss_pureexp_list
    
def width_optimization_statitics(width):
    loss_autoencoder_list,loss_layerprior_list,loss_pureexp_list=optimise_statistics_single_width(width,repetetitions=300,statisitcs=False)
    layer_prior_ratio=np.array(loss_layerprior_list)/np.array(loss_pureexp_list)
    autoencoder_ratio=np.array(loss_autoencoder_list)/np.array(loss_pureexp_list)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: layer_prior_ratio
    axes[0].hist(layer_prior_ratio, bins=10, color='C0')
    # Calculate and plot mean
    mean_prior = np.mean(layer_prior_ratio)
    axes[0].axvline(mean_prior, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_prior:.2f}')
    axes[0].set_title(r'Histogram of $R_{\text{prior}} = \frac{\text{Loss}_{\text{layer\_prior}}}{\text{Loss}_{\text{pureexp}}}$')
    axes[0].set_xlabel('Ratio Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Plot 2: autoencoder_ratio
    axes[1].hist(autoencoder_ratio, bins=10, color='C0')
    # Calculate and plot mean
    mean_ae = np.mean(autoencoder_ratio)
    axes[1].axvline(mean_ae, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_ae:.2f}')
    axes[1].set_title(r'Histogram of $R_{\text{autoencoder}} = \frac{\text{Loss}_{\text{autoencoder}}}{\text{Loss}_{\text{pureexp}}}$')
    axes[1].set_xlabel('Ratio Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.suptitle('Histograms of Loss Ratios', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('ratio_histograms.png')
    plt.savefig('ratio_histograms.svg')
    print("ratio_histograms.png")


import numpy as np
import matplotlib.pyplot as plt

def sweep_sample_stat_thickness():
    """
    Sweeps over a list of widths, uses the aggregated results from 
    optimise_statistics_single_width to plot the median and 50% range (25%-75%) 
    for each method per width.
    
    Returns:
        dict: Concatenated list of [median, p25, p75] lists across all widths 
              for each method.
    """
    # 1. Setup
    width_list = np.linspace(15, 50, 10)
    methods = ['autoencoder', 'layerprior', 'purexp']
    
    # Store the results [median, p25, p75] for each width
    # Structure: {'autoencoder': [[med1, p25_1, p75_1], [med2, p25_2, p75_2], ...], ...}
    results_per_width = {method: [] for method in methods}
    
    # Store the statistics in separate lists for easy plotting
    stats_for_plot = {method: {'median': [], 'p25': [], 'p75': []} for method in methods}

    # 2. Data Collection and Statistic Aggregation
    print("Collecting data and aggregating statistics across widths...")
    for width in width_list:
        # Calls the ORIGINAL function, which returns [median, p25, p75]
        res_dict = optimise_statistics_single_width(width, repetetitions=35)
        
        for key in res_dict.keys():
            # [0] is median, [1] is 25th percentile, [2] is 75th percentile
            median_val = res_dict[key][0]
            p25_val = res_dict[key][1]
            p75_val = res_dict[key][2]
            
            # 2a. Store for final concatenation
            results_per_width[key].append(res_dict[key])
            
            # 2b. Store for plotting
            stats_for_plot[key]['median'].append(median_val)
            stats_for_plot[key]['p25'].append(p25_val)
            stats_for_plot[key]['p75'].append(p75_val)
    
    colors = {
        'autoencoder': 'blue',
        'layerprior': 'green',
        'purexp': 'red'
    }
    
    plt.figure(figsize=(10, 6))
    
    for method in methods:
        median = np.array(stats_for_plot[method]['median'])
        p25 = np.array(stats_for_plot[method]['p25'])
        p75 = np.array(stats_for_plot[method]['p75'])
        color = colors[method]
        
        # Plot the median line
        plt.plot(width_list, median, 
                 label=f'{method} Median', 
                 color=color, 
                 linestyle='-', 
                 marker='o',
                 linewidth=2)
        
        # Plot the 50% region (25%-75% percentile) as a semi-opaque box
        # This region wraps the median line
        plt.fill_between(width_list, p25, p75, 
                         color=color, 
                         alpha=0.3, # Semi-opaque (30% transparency)
                         label=f'{method} 50% IQR')

    # 4. Final Plot Customization
    plt.title('Optimization Loss Statistics vs. Sample Thickness (Width)')
    plt.xlabel('Sample Thickness (Width)')
    plt.ylabel('Loss (Absolute Emission Difference)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('optimization_eval.svg')
    plt.show()
    
    print("\nPlot generated successfully.")
    
    # 5. Concatenate the aggregated results (as raw loss values are unavailable)
    # The result is a list of [median, p25, p75] lists, flattened into one big list.
    concatenated_results = {method: [item for sublist in results_per_width[method] for item in sublist] for method in methods}
    
    return concatenated_results


def optimise_single_width():
    current_date = datetime.now()
    directory_reults=r'/home/nathan.regev/software/git_repo/optimisation_results_new_models_24_11'
    if not os.path.exists(directory_reults):
        os.makedirs(directory_reults)
    width=30
    output_path=os.path.join(directory_reults,f"{width}_"+current_date.strftime('%y%m%d%H%M%S'))
    os.mkdir(output_path)
    """
    def get_opt_struct_and_dist(
                        mod='autoencoder',
                        num_of_layers=10,
                        width=2,
                        lr=1e2,
                        relative_wiehgt=1e-3,
                        num_epochs = 2000,
                        num_of_ini_points=20,
                        ):
    """
    layer_struct_autoencoder,grid_autoencoder,sim_emission_theta_autoencoder,scintillator_list_autoencoder,dialectric_list_autoencoder=get_opt_struct_and_dist(mod='autoencoder',num_epochs=400,num_of_layers=10,width=width,z_bins=100,r_bins=100,lr=1e-3*width/30)
    autoencoder_data=[layer_struct_autoencoder,grid_autoencoder,sim_emission_theta_autoencoder,scintillator_list_autoencoder,dialectric_list_autoencoder]
    print('finished autoencoder optimiseation')

    layer_struct_layerprior,grid_layerprior,sim_emission_theta_layerprior,scintillator_list_layerprior,dialectric_list_layerprior=get_opt_struct_and_dist(mod='prior_layer_model',num_epochs=400,num_of_layers=10,width=width,lr=1e-3*width/30)
    layerprior_data=[layer_struct_layerprior,grid_layerprior,sim_emission_theta_layerprior,scintillator_list_layerprior,dialectric_list_layerprior]
    print('finished layerprior optimiseation')


    layer_struct_pureexp,grid_pureexp,sim_emission_theta_pureexp,scintillator_list_pureexp,dialectric_list_pureexp=get_opt_struct_and_dist(mod='pure_exponential',num_epochs=400,num_of_layers=10,width=width,lr=1e-3*width/30)
    pureexp_data=[layer_struct_pureexp,grid_pureexp,sim_emission_theta_pureexp,scintillator_list_pureexp,dialectric_list_pureexp]
    print('finished pureexp optimiseation')

    rand_layer_vec=np.random.uniform(2.5, 2.5, 10)
    rand_scintillator_list=rand_layer_vec[::2]
    rand_dialectric_list=rand_layer_vec[1::2]
    sim_emission_theta_rand=simulate_emission_angular(rand_layer_vec,rand_scintillator_list,rand_dialectric_list)
    plt.figure()
    plt.plot(sim_emission_theta_autoencoder[0],np.abs(sim_emission_theta_autoencoder[1]),label='autoencoder distribution optimisation')
    plt.plot(sim_emission_theta_layerprior[0],np.abs(sim_emission_theta_layerprior[1]),label='layerprior distribution optimisation')
    plt.plot(sim_emission_theta_pureexp[0],np.abs(sim_emission_theta_pureexp[1]),label='pureexp distribution optimisation')
    #plt.plot(sim_emission_theta_rand[0],np.abs(sim_emission_theta_rand[1]),label='random structure distribution optimisation')
    plt.legend()
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(output_path,'long_eval_stochastical.png'))
    plt.savefig(os.path.join(output_path,'long_eval_stochastical.svg'))
    save_data={'autoencoder':autoencoder_data,'layerprior':layerprior_data,'pureexp':pureexp_data}
    with open(os.path.join(output_path,'optim_results,pkl'), 'wb') as file:
            pickle.dump(save_data, file)
#sweep_sample_thickness()
#optimise_single_width()
#optimise_statistics_single_width(10)
#sweep_sample_stat_thickness()
width_optimization_statitics(15)