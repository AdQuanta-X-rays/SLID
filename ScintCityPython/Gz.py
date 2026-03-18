#!/usr/bin/env python3
import numpy as np
from ScintCityPython.compute_b import compute_b
import torch.optim as optim
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
class Model1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).double() 
        self.fc2 = nn.Linear(hidden_size, output_size).double() 
        self.activation = nn.ReLU().double()
        self.grid_dim=input_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.view(self.grid_dim, self.grid_dim)
        # conv layers....

        return x
class Model2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).double() 
        self.fc2 = nn.Linear(hidden_size, output_size).double()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5,padding=2).double()
        #self.conv1.bias.data = self.conv1.bias.data.double()
        self.activation = nn.ReLU().double()
        self.grid_dim=input_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.view(-1, 1, self.grid_dim, self.grid_dim)
        x = self.conv1(x)
        # conv layers....
        #x = self.activation(x)
        
        

        return x
# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.double), torch.tensor(self.labels[idx], dtype=torch.double)

#params
def MSE_ERR(arr1,arr2):
    return np.sum(np.power(arr1-arr2,2))/np.sum(np.power(arr1-np.mean(arr1),2))
def load_parameters(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
def create_boolean_width(z_bins,scint,dialect,starts_with_scintilator):
    total_list=  np.empty(scint.size + dialect.size, dtype=scint.dtype)
    total_list[(1-starts_with_scintilator)::2]=scint
    total_list[starts_with_scintilator::2]=dialect
    boolean_list=np.zeros(z_bins.size-1)
    acumulated_bins_depth=0
    layer_counter=0
    partial_sum_of_layers=total_list[0]
    z_diff=z_bins[1]-z_bins[0]
    binned_layers_index=[0]# indexing where layers start and end
    for i in range(boolean_list.size):
        if (z_diff+acumulated_bins_depth)<partial_sum_of_layers: #next bin is within the layer
            boolean_list[i]=(layer_counter+starts_with_scintilator)%2
        else:
            #print(i)
            if layer_counter<(total_list.size-1):
                layer_counter+=1
                partial_sum_of_layers+=total_list[layer_counter]
            boolean_list[i]=-1 #interface between layers
        if(np.abs(boolean_list[i])-np.abs(boolean_list[i-1])!=0): # if layers change than need to append the value
            binned_layers_index.append(i)
        acumulated_bins_depth+=z_diff
    if(binned_layers_index[-1]<i):
        binned_layers_index.append(i)
    return boolean_list,np.array(binned_layers_index)
def load_weights(mod, X_val):
    mod='2'
    weights_path=r"C:\Users\nathan\OneDrive - Technion\Desktop\technion\7\projectB\ScintCityPython\20240414130338861210Database_40000samples_nonpriodic_sim_100000_photons_6layers_model2_parameters.pth"
    state_dict = torch.load(weights_path)
    for key, value in state_dict.items():
        print(key, value.shape)
    if mod=='1':
        model = Model1( state_dict['fc1.weight'].shape[1],state_dict['fc2.weight'].shape[1], state_dict['fc2.weight'].shape[0])
    elif mod=='2':
        #model = Model2( state_dict['fc1.weight'].shape[1],state_dict['fc2.weight'].shape[1], state_dict['fc2.weight'].shape[0])
        model = Model2( 99,99, 99*99)
    model.load_state_dict(state_dict)
    res = model(torch.tensor(X_val).double())
    if mod =='2':
        res=res[0,0,:,:]
    return res
def Gz(f, d_tot, i_scint, control):
    rel_i_scint=i_scint/np.sum(i_scint)
    if control["is_Gz"]:
        if (control["distribution"]=='exponential'):
            G = np.transpose(f, (2, 3, 0, 1))
            d = d_tot.detach().numpy()[1:(-1)]
            profile=[]
            bottom_distance, _ = compute_b(d, i_scint, 1, 1, control["is_Gz"], control["dz_Nz"])
            for i in range(len(i_scint)):	# Iterating over all scintillator layers
                for j in range(control["dz_Nz"]): # Iterating over all emitters in a layer
                    full_layers_scint_distance = np.sum(d_tot.detach().numpy()[i_scint[:(i)]])
                    scint_distance = full_layers_scint_distance + bottom_distance[i, j, 0, 0]
                    other_distance = np.sum( d_tot.detach().numpy()[:(i_scint[i])]) - full_layers_scint_distance
                    nomalisation_coeff=1#d_tot[i_scint[i]]/(np.sum(d))
                    G[:, :, i, j] *=rel_i_scint[i]*nomalisation_coeff* np.exp(-scint_distance / control["absorption_scint"] - other_distance / control["absorption_other"])
                    profile.append( nomalisation_coeff*np.exp(-scint_distance.detach().numpy() / control["absorption_scint"] - other_distance / control["absorption_other"]))
            if control["plot_profile"]:
                plt.figure()
                plt.plot(np.array(profile))
                plt.xlabel('distance')
                plt.ylabel('distribution')
                plt.title('exponential distribution')
                plt.show()

            G = np.transpose(G, (2, 3, 0, 1))
        elif control["distribution"]=='exponential_optim':
            G=f.permute(2, 3, 0, 1)
            profile=[]
            bottom_distance, _ = compute_b(d_tot, i_scint, 1, 1, control["is_Gz"], control["dz_Nz"])
            for i in range(len(i_scint)):	# Iterating over all scintillator layers
                for j in range(control["dz_Nz"]): # Iterating over all emitters in a layer
                    full_layers_scint_distance = torch.sum(d_tot[i_scint[:(i)]])
                    scint_distance = full_layers_scint_distance + bottom_distance[i, j, 0, 0]
                    other_distance = torch.sum( d_tot[:(i_scint[i])]) - full_layers_scint_distance
                    nomalisation_coeff=1#d_tot[i_scint[i]]/(np.sum(d))
                    G[:, :, i, j] *=rel_i_scint[i]*nomalisation_coeff* torch.exp(-scint_distance / control["absorption_scint"] - other_distance / control["absorption_other"])
                    profile.append( nomalisation_coeff*torch.exp(-scint_distance / control["absorption_scint"] - other_distance / control["absorption_other"]))
            G=G.permute(2, 3, 0, 1)
        elif control["distribution"]=='NN':
            mode='1'
            layer_vec=d_tot.detach().numpy()[1:-1]
            starts_with_scintilator=(bool(i_scint[0]-1))#not (bool(i_scint[0]-1))
            #G,profile=ML_wieghts_emmitors_distrebution(dist,binned_layers_index,f, starts_with_scintilator,control)
            z_bins=np.linspace(0,np.sum(layer_vec),100)
            starts_with_dielectric=1-starts_with_scintilator
            binned_vec,binned_layers_index=create_boolean_width(z_bins,layer_vec[starts_with_dielectric::2],layer_vec[starts_with_scintilator::2],starts_with_scintilator)
            dist=load_weights(mode, binned_vec)
            dist=dist.detach().numpy()
            G,profile=ML_wieghts_emmitors_distrebution(dist,binned_layers_index[1:],f, starts_with_scintilator,control,rel_i_scint)
            if control["plot_profile"]:
                plt.figure()
                plt.plot(profile)
                plt.xlabel('distance')
                plt.ylabel('distribution')
                plt.title('NN distribution')
                plt.show()
            G = np.transpose(G, (2, 3, 0, 1))
        
        elif control["distribution"]=='optimisation':
            try:
                layer_vec=d_tot[1:-1]
                starts_with_scintilator=bool(i_scint[0]-1)
                #G,profile=ML_wieghts_emmitors_distrebution(dist,binned_layers_index,f, starts_with_scintilator,control)
                z_bins=torch.linspace(0,torch.sum(layer_vec),100)
                starts_with_dielectric=1-starts_with_scintilator
                #binned_vec,binned_layers_index=create_boolean_width(z_bins,layer_vec[starts_with_dielectric::2],layer_vec[starts_with_scintilator::2],starts_with_scintilator)
                dist=control["distribution_map"]
                binned_layers_index=control["binary_indexes"]
                G,profile=ML_wieghts_emmitors_distrebution_torch(dist,binned_layers_index,f, starts_with_scintilator,control,rel_i_scint)
                if control["plot_profile"]:
                    plt.figure()
                    plt.plot(profile)
                    plt.xlabel('distance')
                    plt.ylabel('distribution')
                    plt.title('NN distribution')
                    plt.show()
                G=G.permute(2, 3, 0, 1)
                # G= np.transpose(G, (2, 3, 0, 1))
            except:
                G,profile=ML_wieghts_emmitors_distrebution_torch(dist,binned_layers_index,f, starts_with_scintilator,control,rel_i_scint)
        elif control["distribution"]=='simulation':
                layer_vec=d_tot[1:-1]
                starts_with_scintilator=not (bool(i_scint[0]-1))
                G,profile=simulation_distrebution(layer_vec,f, starts_with_scintilator,control,rel_i_scint)
                if control["plot_profile"]:
                    plt.figure()
                    plt.plot(profile)
                    plt.xlabel('distance')
                    plt.ylabel('distribution')
                    plt.title('simulation distribution')
                    plt.show()
                G = np.transpose(G, (2, 3, 0, 1))
        else:  # uniform distribution of emitters
            d_scint = d_tot[i_scint]
            G = np.transpose(f, (3, 1, 2, 0))

            for i in range(len(d_scint)):
                G[:, :, :, i] *= d_scint[i]

            G = np.transpose(G, (3, 1, 2, 0))
    
    else:
         f

    return G,profile
def simulation_distrebution(layer_vec,f, starts_with_scintilator,control,rel_i_scint):
    with open(control['simulation_path'], 'rb') as file:
        data = pickle.load(file)
    G = np.transpose(f, (2, 3, 0, 1))
    z_bins=np.linspace(0,np.sum(layer_vec.detach().numpy()),100)
    starts_with_dielectric=1-starts_with_scintilator
    binned_vec,binned_layers_index=create_boolean_width(z_bins,layer_vec.detach().numpy()[starts_with_dielectric::2],layer_vec.detach().numpy()[starts_with_scintilator::2],starts_with_scintilator)
    profile=np.array([])
    dist=data['grid'][0]
    z_axis_emiter_dist=np.sum(dist,axis=0)
    #z_axis_emiter_dist_normalised=z_axis_emiter_dist/np.sum(dist)
    z_axis_emiter_dist_normalised=z_axis_emiter_dist
    binned_layers_index=binned_layers_index
    for scint_ind in range(len(binned_layers_index+1)//2):
        nomalisation_coeff=1#layer_vec[list_ind]/(np.sum(layer_vec))
        list_ind=scint_ind*2+(starts_with_dielectric)
        layer_dist=z_axis_emiter_dist_normalised[binned_layers_index[list_ind]:binned_layers_index[list_ind+1]] # get all the bins in the relevant layer
        bin_val=np.arange(0,binned_layers_index[list_ind+1]-binned_layers_index[list_ind]) # bins between the two points
        interpolated_vals=np.linspace (0,binned_layers_index[list_ind]-binned_layers_index[list_ind-1],control["dz_Nz"])# generate a list of emitters evenly spaced accross scint layer
        """
        if starts_with_dielectric:
            layer_dist=z_axis_emiter_dist_normalised[binned_layers_index[list_ind]:binned_layers_index[list_ind+1]] # get all the bins in the relevant layer
            bin_val=np.arange(0,binned_layers_index[list_ind]-binned_layers_index[list_ind-1]) # bins between the two points
            interpolated_vals=np.linspace (0,binned_layers_index[list_ind]-binned_layers_index[list_ind-1],control["dz_Nz"])# generate a list of emitters evenly spaced accross scint layer
        else:
            layer_dist=z_axis_emiter_dist_normalised[binned_layers_index[list_ind]:binned_layers_index[list_ind+1]] # get all the bins in the relevant layer
            bin_val=np.arange(0,binned_layers_index[list_ind+1]-binned_layers_index[list_ind]) # bins between the two points
            interpolated_vals=np.linspace (0,binned_layers_index[list_ind+1]-binned_layers_index[list_ind],control["dz_Nz"])# generate a list of emitters evenly spaced accross scint layer
        """
        
        interp_emit_vals=np.interp(interpolated_vals,bin_val,layer_dist)
        G[:, :, scint_ind,:]*=rel_i_scint[scint_ind]*interp_emit_vals*nomalisation_coeff # push the relevant disterbution with normalisation facor refeing to the layers thickness
        profile=np.concatenate((profile,interp_emit_vals*nomalisation_coeff))
    return G,profile
    
#def ML_wieghts_emmitors_distrebution(mode,layer_vec,f, starts_with_scintilator,control): #dtot thickness vec
def ML_wieghts_emmitors_distrebution(dist,binned_layers_index,f, starts_with_scintilator,control,rel_i_scint): #dtot thickness vec
    G = np.transpose(f, (2, 3, 0, 1))
    profile=np.array([])
    #z_bins=np.linspace(0,np.sum(layer_vec),100)
    starts_with_dielectric=1-starts_with_scintilator
    #binned_vec,binned_layers_index=create_boolean_width(z_bins,layer_vec[starts_with_dielectric::2],layer_vec[starts_with_scintilator::2],starts_with_scintilator)
    
    #dist=load_weights(mode, binned_vec)
    #dist=dist.detach().numpy()
    z_axis_emiter_dist=np.sum(dist,axis=1)
    z_axis_emiter_dist_normalised=z_axis_emiter_dist/np.sum(dist)
    for scint_ind in range(len(binned_layers_index+1)//2):
        nomalisation_coeff=1#layer_vec[list_ind]/(np.sum(layer_vec))
        list_ind=scint_ind*2+starts_with_dielectric
        if starts_with_dielectric:
            layer_dist=z_axis_emiter_dist_normalised[binned_layers_index[list_ind-1]:binned_layers_index[list_ind]] # get all the bins in the relevant layer
            bin_val=np.arange(0,binned_layers_index[list_ind]-binned_layers_index[list_ind-1]) # bins between the two points
            interpolated_vals=np.linspace (0,binned_layers_index[list_ind]-binned_layers_index[list_ind-1],control["dz_Nz"])# generate a list of emitters evenly spaced accross scint layer
        else:
            layer_dist=z_axis_emiter_dist_normalised[binned_layers_index[list_ind]:binned_layers_index[list_ind+1]] # get all the bins in the relevant layer
            bin_val=np.arange(0,binned_layers_index[list_ind+1]-binned_layers_index[list_ind]) # bins between the two points
            interpolated_vals=np.linspace (0,binned_layers_index[list_ind+1]-binned_layers_index[list_ind],control["dz_Nz"])# generate a list of emitters evenly spaced accross scint layer
        interp_emit_vals=np.interp(interpolated_vals,bin_val,layer_dist)
        G[:, :, scint_ind,:]*=rel_i_scint[scint_ind]*interp_emit_vals*nomalisation_coeff # push the relevant disterbution with normalisation facor refeing to the layers thickness
        profile=np.concatenate((profile,interp_emit_vals*nomalisation_coeff))
    return G,profile
def ML_wieghts_emmitors_distrebution_torch(dist,binned_layers_index,f, starts_with_scintilator,control,rel_i_scint): #dtot thickness vec
    G = f.permute(2, 3, 0, 1)
    profile=torch.tensor([])
    starts_with_dielectric=1-starts_with_scintilator
    z_axis_emiter_dist=torch.sum(torch.abs(dist),axis=0)
    z_axis_emiter_dist_normalised=z_axis_emiter_dist/torch.sum(dist)
    try:
        for scint_ind in range(len(binned_layers_index+1)//2):
            nomalisation_coeff=1#layer_vec[list_ind]/(np.sum(layer_vec))
            list_ind=scint_ind*2+starts_with_dielectric
            layer_dist=z_axis_emiter_dist_normalised[binned_layers_index[list_ind]:binned_layers_index[list_ind+1]] # get all the bins in the relevant layer
            interpolated_vals=torch.linspace (0,binned_layers_index[list_ind+1]-binned_layers_index[list_ind],control["dz_Nz"])# generate a list of emitters evenly spaced accross scint layer
            bin_val=torch.arange(0,binned_layers_index[list_ind+1]-binned_layers_index[list_ind]) # bins between the two points
            interp_emit_vals=torch.interp(interpolated_vals,bin_val,layer_dist)
            G[:, :, scint_ind,:]*=rel_i_scint[scint_ind]*torch.tensor(interp_emit_vals*nomalisation_coeff )# push the relevant disterbution with normalisation facor refeing to the layers thickness
            profile=torch.concatenate((profile,interp_emit_vals*nomalisation_coeff))
    except:
        pass
    return G,profile


def graph_params(ptitle, pxlabel, pylabel):
	plt.grid(True)
	plt.title(ptitle)
	plt.xlabel(pxlabel)
	plt.ylabel(pylabel)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.show()