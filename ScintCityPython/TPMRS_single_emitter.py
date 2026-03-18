#!/usr/bin/env python3
''' This is Python implementation of the Matlab code of ScintCity. 
The current script should reproduce the TPMRS_single_emitter.m results.
However, tests to reproduce Matlab's results were only done for a few set of 
parameters, so the current script is *not* fully validated.
F. Loignon-Houle, 2024-03-21.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from ScintCityPython.Scint_City_fun import Scint_City_fun
import pickle
import warnings
import torch
#warnings.simplefilter("ignore", np.ComplexWarning)

# Set figure properties
plt.rcParams['figure.figsize'] = [9, 6]
plt.rcParams['text.usetex'] = False
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['xtick.labelsize'] = 'medium'
plt.rcParams['ytick.labelsize'] = 'medium'
plt.rcParams['lines.linewidth'] = 4


def drawStackedLayers(vector, palette_colors, draw_emitters, figure_num, num_plots):
	# Ensure the vector is a column vector
	if vector.ndim == 1 and vector.shape[0] == 1:
		vector = vector.reshape(-1, 1)
	elif vector.ndim == 1:
		vector = vector.reshape(1, -1)

	scintillator_color = np.array([165, 178, 186]) / 255
	dielectric_color = np.array([219, 228, 238]) / 255

	# Create a figure
	plt.figure(figure_num)
	plt.subplot(1, num_plots, 1)

	# Initialize the y position of the first rectangle
	y = 0

	# Define the width of the rectangles
	width = 1

	counter = 1
	# For each element in the vector
	for i in range(vector.shape[1]):
		if i % 2 == 0:
			color = scintillator_color
		else:
			color = dielectric_color

		# Draw a rectangle at the current y position with height equal to the vector element
		plt.rectangle = plt.Rectangle((0, y), width, vector[0, i], facecolor=color)
		plt.gca().add_patch(plt.rectangle)

		if draw_emitters:
			if i % 2 == 0:
				# Draw a point in the middle of the current layer
				plt.plot(width / 2, y + vector[0, i] / 2, '.', markersize=20, color=palette_colors[counter-1])
				txt = f'emitter {counter}'
				plt.text(width / 2 - 0.06, y + vector[0, i] / 2 + 60, txt)
				counter += 1

		# Update the y position for the next rectangle
		y = y + vector[0, i]

	plt.xlim([0, width])
	plt.ylim([0, y])
	plt.xticks([])
	plt.tick_params(axis='y', labelsize=10, length=0)
	plt.xlabel("x")
	plt.ylabel("z [nm]")


def findMaxInRange(x, y, x_range):
	''' Find maximum of curve within x-range.
	x (1D np array): X values.
	y (1D np array): Y values.
	x_range (tuple): Left and right values to define x-range. 
	Returns: float.
	'''
	return np.max(y[np.where((x >= x_range[0]) & (x <= x_range[1]))])


def calculating_emmision(control,plot_figs=False,layer_struct='database'):
	# Initialize parameters

    if layer_struct=='database':
        with open(control['simulation_path'], 'rb') as file:
          data = pickle.load(file)
        n_scint=2.27
        n_dial=1.45
        optimized_d=np.zeros(data['dialectric'][0].size+data['scintlator'][0].size)
        starts_with_scintilator=int(data['binned'][0][0])
        starts_with_dialectic=1-starts_with_scintilator
        optimized_d[starts_with_scintilator::2]=data['scintlator'][0]#*1e6 #[nm]
        optimized_d[starts_with_dialectic::2]=data['dialectric'][0]#*1e6 #[nm]
        scint_thickness=np.sum(data['scintlator'][0])
        i_scint=2*np.arange(0,data['scintlator'][0].size,1)+1+starts_with_dialectic
        n=np.zeros(optimized_d.size)
        n[starts_with_dialectic::2]=n_scint
        n[starts_with_scintilator::2]=n_dial
        n=np.concatenate(([n_dial], n, [1]), axis=0)
        pairs = data['scintlator'][0].size
        optimized_d=torch.tensor(optimized_d)
    elif layer_struct=='optimisation':
        n_scint=2.27
        n_dial=1.45
        optimized_d=control['layer_vector']
        num_of_layers=control['num_of_layers']
        starts_with_scintilator=control['start_with_scint']
        starts_with_dialectic=1-starts_with_scintilator
        i_scint=2*np.arange(0,num_of_layers//2,1)+1

        n=np.zeros(num_of_layers)
        n[starts_with_dialectic::2]=n_scint
        n[starts_with_scintilator::2]=n_dial
        n=np.concatenate(([n_dial], n, [1]), axis=0)
        pairs = num_of_layers//2
        scint_thickness=float(torch.sum(optimized_d[starts_with_dialectic::2]).detach().numpy())
    else:
            n_scint=2.27
            n_dial=1.45
            

            
            d_scint = 200 #[nm]
            d_dielectric = 400 #[nm]
            starts_with_scintilator=1
            #optimized_d = np.array([d_scint, d_dielectric, d_scint, d_dielectric, d_scint, d_dielectric, d_scint, d_dielectric, d_scint, d_dielectric])
            optimized_d_NN = np.array([170.36434405 ,103.72112739 ,188.46680608 ,286.26414983 ,184.70550132 ,182.15175905])# optimsed for NN
            optimized_d = np.array([289.3233821   ,98.90700388 ,181.06014221, 310.36714721 ,174.44964098,191.62531881])
            optimized_d=np.array([165.75903586,419.23699478,173.49848034,303.59285467,300.1424706,450.90732611,403.78180003 ,93.91328929,291.21572006,401.15718941])
            optimized_d=np.array([186.76011097 ,375.75039924 ,299.53567039, 326.15721641 ,279.51996319,379.10120884 ,205.32427484 ,293.99336469 ,292.33130049 ,399.60659251])
            pairs = optimized_d.size//2
            num_of_layers=pairs*2
            scint_thickness=np.sum(optimized_d[1-starts_with_scintilator::2])
            #i_scint = np.array([1, 3, 5, 7, 9])
            
            starts_with_dialectic=1-starts_with_scintilator
            i_scint=2*np.arange(0,num_of_layers//2,1)+1+starts_with_dialectic
            n=np.zeros(num_of_layers)
            n[starts_with_dialectic::2]=n_scint
            n[starts_with_scintilator::2]=n_dial
            n=np.concatenate(([n_dial], n, [1]), axis=0)
            #n = np.array([1.45, 2.27, 1.45, 2.27, 1.45, 2.27, 1.45, 2.27, 1.45, 2.27, 1.45, 1])
            optimized_d=torch.tensor(optimized_d)

    theta = np.linspace(-np.pi/2, np.pi/2, 101)
    mu_lambda = np.array([505]) # Impose an array even with single value to keep axes ok
    sigma_lambda = 0
    lambda_val = mu_lambda
	#lambda_val = np.linspace(mu_lambda - 3*sigma_lambda, mu_lambda + 3*sigma_lambda, 31).squeeze()

    
    n_substrate = 1.45
    n_scint = 2.27
    n_other = 1.45
    d0 = optimized_d
    dir_name = "Results_single_emitter"
    save_fig = True

	# Spectral distribution
	# if the std of the spectral distribution is 0, we calculate for only one wavelength.
    if sigma_lambda == 0:
        Y_orig = np.zeros_like(lambda_val)
        Y_orig[lambda_val == mu_lambda] = 1
    else:
        Y_orig = np.exp(-(lambda_val - mu_lambda)**2 / (2 * sigma_lambda**2)) / (sigma_lambda * np.sqrt(2 * np.pi))
    Y = Y_orig.copy()

	# General parameters
	# Saving the original values of theta and lambda. Will bu used when plotting.
    theta_orig = theta
    lambda_orig = lambda_val
    num_theta_plot = 301

	# Defining the colors in the plots.
    lighter = -0.071
    Green = np.array([0.4660, 0.6740, 0.1880]) - lighter
    Red = np.array([0.6350, 0.0780, 0.1840]) - lighter
    Gold = np.array([0.9290, 0.6940, 0.1250]) - lighter
    Green_Sim = np.array([0.4660, 0.6740, 0.1880]) + lighter
    Red_Sim = np.array([0.6350, 0.0780, 0.1840]) + lighter
    Gold_Sim = np.array([0.9290, 0.6940, 0.1250]) + lighter

	# Add the current directory to the provided folder name
    dir_name = dir_name

	# Coupled and inside parameters
    coupled = True
    inside = False

	# Calculate results
    control['orientation'] = 4
    #theta = np.linspace(-np.pi/2 + 10/180*np.pi, np.pi/2 - 10/180*np.pi, num_theta_plot)
    theta = torch.linspace(-torch.pi/2 + 10/180*torch.pi, torch.pi/2 - 10/180*torch.pi, num_theta_plot)
    lambda_val = lambda_orig

    control['sum_on_z'] = False
    outcouples_purcell_factor,profile = Scint_City_fun(lambda_val, theta, optimized_d, n, i_scint, coupled, control,ret_profile=True)
    control['sum_on_z'] = True
    norm_F = 1
    averaged_outcouples_purcell_factor = Scint_City_fun(lambda_val, theta, optimized_d, n, i_scint, coupled, control)

    if Y.shape[0] > 1:
       averaged_outcouples_purcell_factor = torch.dot(averaged_outcouples_purcell_factor, torch.tensor(Y))
    else:
       averaged_outcouples_purcell_factor = averaged_outcouples_purcell_factor *  torch.tensor(Y)

    norm_F_avg = outcouples_purcell_factor.shape[0] * outcouples_purcell_factor.shape[1]
	# === Plotting (1st figure)
    
    y_avg = averaged_outcouples_purcell_factor / norm_F_avg
    #y_avg = averaged_outcouples_purcell_factor 
    param_1=[theta,y_avg]

	# Now same thing but with "inside" instead of "coupled"
    control['sum_on_z'] = False
    outcouples_purcell_factor = Scint_City_fun(lambda_val, theta, optimized_d, n, i_scint, inside, control)
    control['sum_on_z'] = True
    averaged_outcouples_purcell_factor = Scint_City_fun(lambda_val, theta, optimized_d, n, i_scint, inside, control)

    if Y.shape[0] > 1:
        averaged_outcouples_purcell_factor = np.dot(averaged_outcouples_purcell_factor, torch.tensor(Y))
    else:
        averaged_outcouples_purcell_factor = averaged_outcouples_purcell_factor * torch.tensor(Y)

    norm_F_avg = outcouples_purcell_factor.shape[0] * outcouples_purcell_factor.shape[1]
    y_avg = averaged_outcouples_purcell_factor / norm_F_avg
    #y_avg = averaged_outcouples_purcell_factor 
    param_2=[theta,y_avg]
    try:
        depth_profile=np.array([np.linspace(0,scint_thickness,np.size(profile)),profile])
    except:
          depth_profile=0
    return depth_profile,param_1,param_2
# ======================
def main():
    num_of_layers=10
    absorption_scint=3.95e4 # [nm]]
    absorption_other=4.22e6#[nm]
    control_simulation = {
		'is_Gz': 1,
		'dz_Nz': 10,
		'distribution': 'simulation',
		'simulation_path':r"C:\Users\nathan\OneDrive - Technion\Desktop\technion\7\projectB\ScintCityPython\20240417132503423431Database_10samples_nonpriodic_sim_10000000_photons_6layers.pkl",
		'absorption_scint': absorption_scint,
		'absorption_other': absorption_other,
		'plot_profile': 0,
		'load_data_from_dir': 1,
        'num_of_layers':num_of_layers
    }
    control_ML = {
		'is_Gz': 1,
		'dz_Nz': 10,
		'distribution': 'NN',
		'simulation_path':r"C:\Users\nathan\OneDrive - Technion\Desktop\technion\7\projectB\ScintCityPython\20240417132503423431Database_10samples_nonpriodic_sim_10000000_photons_6layers.pkl",
		'absorption_scint': absorption_scint,
		'absorption_other': absorption_other,
		'plot_profile': 0,
		'load_data_from_dir': 0,
         'num_of_layers':num_of_layers          
	}
    control_exp = {
		'is_Gz': 1,
		'dz_Nz': 10,
		'distribution': 'exponential',
		'simulation_path':r"C:\Users\nathan\OneDrive - Technion\Desktop\technion\7\projectB\ScintCityPython\20240417132503423431Database_10samples_nonpriodic_sim_10000000_photons_6layers.pkl",
		'absorption_scint': absorption_scint,
		'absorption_other': absorption_other,
		'plot_profile': 0,
		'load_data_from_dir': 0,
        'num_of_layers':num_of_layers
	}
    ML_emmsioion_ditribution,ML_emission_theta,ML_emission_phi=calculating_emmision(control_ML,layer_struct='default')
    exp_emmsioion_ditribution,exp_emission_theta,exp_emission_phi=calculating_emmision(control_exp,layer_struct='default')
    #sim_emmsioion_ditribution,sim_emission_theta,sim_emission_phi=calculating_emmision(control_simulation)
    ML_emission_theta[1]=ML_emission_theta[1].detach().numpy()
    ML_emission_phi[1]=ML_emission_phi[1].detach().numpy()
    exp_emission_theta[1]=exp_emission_theta[1].detach().numpy()
    exp_emission_phi[1]=exp_emission_phi[1].detach().numpy()
    #sim_emission_theta[1]=sim_emission_theta[1].detach().numpy()
    #sim_emission_phi[1]=sim_emission_phi[1].detach().numpy()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(ML_emission_theta[0],ML_emission_theta[1]/np.sum(ML_emission_theta[1]),label='NN distribution')
    plt.plot(exp_emission_theta[0],exp_emission_theta[1]/np.sum(exp_emission_theta[1]),label='exponential distribution')
    #plt.plot(sim_emission_theta[0],sim_emission_theta[1]/np.sum(sim_emission_theta[1]),label='simulationn distribution')
    plt.xlabel(r'$\theta$ - far-field angle [rad]')
    plt.ylabel('outcoupled emission rate enhancement [a.u.]')
    plt.subplot(1, 2, 2)
    plt.plot(ML_emission_phi[0],ML_emission_phi[1]/np.sum(ML_emission_phi[1]),label='NN distribution')
    plt.plot(exp_emission_phi[0],exp_emission_phi[1]/np.sum(exp_emission_phi[1]),label='exponential distribution')
    #plt.plot(sim_emission_phi[0],sim_emission_phi[1]/np.sum(sim_emission_phi[1]),label='simulation distribution')
    plt.xlabel(r'$\phi$ - emitter orientation [rad]')
    plt.ylabel('$F_P$')
    plt.legend(loc="upper right")
    #plt.show()
    """
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(exp_emission_theta[0],np.abs(exp_emission_theta[1]/np.sum(exp_emission_theta[1])-sim_emission_theta[1]/np.sum(sim_emission_theta[1])))
    plt.xlabel(r'$\theta$ - far-field angle [rad]')
    plt.ylabel('error')
    plt.title(r'$\theta$ error exponential vs simulation')
    plt.subplot(2, 2, 2)
    plt.plot(exp_emission_phi[0],np.abs(exp_emission_phi[1]/np.sum(exp_emission_phi[1])-sim_emission_phi[1]/np.sum(sim_emission_phi[1])))
    plt.xlabel(r'$\phi$ - far-field angle [rad]')
    plt.ylabel('error')
    plt.title(r'$\phi$ error exponential vs simulation')
    plt.subplot(2, 2, 3)
    plt.plot(ML_emission_theta[0],np.abs(ML_emission_theta[1]/np.sum(ML_emission_theta[1])-sim_emission_theta[1]/np.sum(sim_emission_theta[1])))
    plt.xlabel(r'$\theta$ - far-field angle [rad]')
    plt.ylabel('error')
    plt.title(r'$\theta$ error NN vs simulation')
    plt.subplot(2, 2, 4)
    plt.plot(ML_emission_phi[0],np.abs(ML_emission_phi[1]/np.sum(ML_emission_phi[1])-sim_emission_phi[1]/np.sum(sim_emission_phi[1])))
    plt.xlabel(r'$\phi$ - far-field angle [rad]')
    plt.ylabel('error')
    plt.title(r'$\phi$ error NN vs simulation')
    plt.tight_layout()
    #plt.show()
    """
    plt.figure()
    plt.plot(ML_emmsioion_ditribution[0],ML_emmsioion_ditribution[1],label='NN')
    plt.plot(exp_emmsioion_ditribution[0],exp_emmsioion_ditribution[1],label='exp')
    #plt.plot(sim_emmsioion_ditribution[0],sim_emmsioion_ditribution[1],label='simulation')
    plt.xlabel('distance (scintilator only) [nm]')
    plt.ylabel('normalised emmission profile')
    plt.title('emiiters distribution profile comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()
if __name__=='__main__':
	main()