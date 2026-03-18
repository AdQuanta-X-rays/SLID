#!/usr/bin/env python3
import numpy as np
from ScintCityPython.Gz import Gz
from ScintCityPython.compute_b import compute_b
import torch

def R_eff(eps, d, u, lambda_, type_):
	"""
	Calculates the special return coefficient

	Parameters:
	eps : ndarray
		Vector of the permittivity coefficients.
	d : ndarray
		Vector of the layers' widths.
	u : ndarray or scalar
		Planar wave vector size.
	lambda_ : float
		Wavelength.
	type_ : str
		's' (TE) or 'p' (TM).

	Drawing
				d1      d2      d3       d4     d5
			 ------- ------- ------- ------- -------     ...
			|       |       |       |       |       |
			|       |       |       |       |       |
			|       |       |       |       |       |
	  R*--> |       |       |       |       |       |    ...
			|       |       |       |       |       |
			|       |       |       |       |       |
			|       |       |       |       |       |
	  eps1    eps2     eps3    eps4    eps5    eps6      ...

	Returns:
	r : ndarray
		The return coefficients.
	"""

	if len(eps) == 1:
		r = torch.zeros(u.shape)
	else:
		k1 = 2 * torch.pi / lambda_ * np.sqrt(eps[0])
		k2 = 2 * torch.pi / lambda_ * np.sqrt(eps[1])

		l1 = np.sqrt(k1**2 - u**2)  # kz1
		l2 = np.sqrt(k2**2 - u**2)  # kz2

		if type_ == 's':
			r12 = torch.tensor((l1 - l2) / (l1 + l2), dtype=torch.float64)  # TE
		elif type_ == 'p':
			r12 =torch.tensor((l2 * eps[0] - l1 * eps[1]) / (l2 * eps[0] + l1 * eps[1]), dtype=torch.float64)    # TM
		else:
			return []

		R = R_eff(eps[1:], d[1:], u, lambda_, type_)
		
		exp_term = torch.exp(2j *torch.tensor(l2, dtype=torch.float64)  * d[0])
		r = (r12 + R[0] * exp_term) / (torch.tensor(1, dtype=torch.float64) + r12 * R[0] * exp_term)
		r = torch.concatenate((r, R))

	return r



def T_eff(eps, d, u, R, lambda_, type_):
	"""
	Calculates the special transmission coefficient

	Parameters:
	eps : ndarray
		Vector of the permittivity coefficients.
	d : ndarray
		Vector of the layers' widths.
	u : ndarray or scalar
		Planar wave vector size.
	lambda_ : float
		Wavelength.
	R : ndarray
		The return coefficients calculated previously.
	type_ : str
		's' (TE) or 'p' (TM).

	Drawing
	            d1      d2      d3       d4     d5
	          ------- ------- ------- ------- -------     ...
	         |       |       |       |       |       |
	         |       |       |       |       |       |
	         |       |       |       |       |       |
  	   R*--> |       |       |       |       |       |    ...
	         |       |       |       |       |       |
	         |       |       |       |       |       |
	         |       |       |       |       |       |
	   eps1    eps2     eps3    eps4    eps5    eps6      ...

	Returns:
	t : ndarray
		The transmission coefficients.
	"""

	if len(eps) == 1:
		t = torch.ones(u.shape)
	else:
		k1 = 2 * torch.pi / lambda_ * np.sqrt(eps[0])
		k2 = 2 * torch.pi / lambda_ * np.sqrt(eps[1])

		l1 = np.sqrt(k1**2 - u**2)  # kz1
		l2 = np.sqrt(k2**2 - u**2)  # kz2

		if type_ == 's':
			r12 =torch.tensor((l1 - l2) / (l1 + l2), dtype=torch.float64)    # TE
			t12 =torch.tensor( 2 * l1 / (l1 + l2) , dtype=torch.float64)      # TE
		elif type_ == 'p':
			r12 =torch.tensor( (l2 * eps[0] - l1 * eps[1]) / (l2 * eps[0] + l1 * eps[1]), dtype=torch.float64)  # TM
			t12 =torch.tensor(2 * l1 * np.sqrt(eps[0] * eps[1]) / (l2 * eps[0] + l1 * eps[1]), dtype=torch.float64)   # TM
		else:
			return []
		
		T = T_eff(eps[1:], d[1:], u, R[1:], lambda_, type_)

		exp_term_1 = torch.exp(1j * torch.tensor(l2, dtype=torch.float64) * d[0])
		exp_term_2 = torch.exp(2j * torch.tensor(l2, dtype=torch.float64) * d[0])
		t = (t12 * T[0] * exp_term_1) / (1 + r12 * R[1] * exp_term_2)
		t = torch.concatenate((t, T))

	return t


def Scint_City_fun(lambda_, theta, d, n, i_scint, coupled, control,ret_profile=False):
	"""
	This function calculates the emission rate enhancement of a given structure.

	Itorchuts:
		theta : scalar or array
			The solid angle in which we compute the emission rate.
		lambda_ : scalar or array
			The wavelength in which we compute the emission rate.
		d : array
			A array of the thicknesses of the structure.
		n : array
			A array of the refractive indices of the structure.
		i_scint : array
			A vector of the indices of the scintillator layers.
		coupled : bool
			If True, compute the emission rate of the coupled wave (outside the structure)
			and the wave inside the device.
		control : dict
			A dictionary containing computation parameters:
				- 'is_Gz': str
					There are two modes of computation - 'Gz' or 'dz'.
					'Gz' distributes a fixed number of emitters in each layer of scintillator,
					then weights the contribution of each dipole to the total emission with the Gz function.
					'dz' assumes a distance of dz between two emitters.
					Thus, thicker layers will have more emitters.
				- 'dz_Nz': float
					In 'Gz' mode, this parameter is the fixed number of emitters in each layer.
					In 'dz' mode, this number is the distance between two emitters.
				- 'sum_on_z': bool
					If True, the function sums all the contributions of the emitters and returns the total emission rate.
					Otherwise, the function returns the emission rate of each dipole separately.

	Outputs:
		f : ndarray
			The emission rate inside or outside the device.
			If 'sum_on_z' is True, the dimensions of f are [N_theta X N_lambda],
			otherwise the dimensions of f are [Scintilator layers X emitters in each layer X N_theta X N_lambda].
	"""
	dz_Nz = control['dz_Nz']
	is_Gz = control['is_Gz']
	sum_on_z = control['sum_on_z']

	num_layers = len(n)

	# theta and lambda can be a single value or a vector of value
	# the output is a matrix of the emission rate per angle and wavelength
	N_theta = len(theta)
	N_lambda = len(lambda_)

	#d_tot = torch.concatenate((torch.tensor(0), d, torch.tensor(0)))
	d_tot = torch.cat((torch.tensor([0.0]), d, torch.tensor([0.0])))
	# b is the distance of each emitter to the nearest bottom of the interface.
	# compute_b compute this matrix
	b, max_Nz = compute_b(d, i_scint, N_theta, N_lambda, is_Gz, dz_Nz)
	d_scint = torch.tile(d_tot[i_scint, None, None, None], (1, max_Nz, N_theta, N_lambda))#.squeeze()
	n_scint = np.tile(n[i_scint, np.newaxis, np.newaxis, np.newaxis], (1, max_Nz, N_theta, N_lambda))
	top = d_scint - b  # distance of dipole from nearest top interface

	# In dz mode, each layer can have a different layer, thus a different
	# number of emitters in each layer. We assigned a matrix assuming all layers have
	# the thickness of the thickest layer, then we zero the irrelevant emitters using a mask.
	mask = b != 0

	## Effective Fresnel's coeff calculation

	# Assuming theta is a 1xN numpy array and lambda is a list of two numbers
	theta_mat = np.tile(theta, (N_lambda, 1, 1)).transpose(1, 2, 0)
	lambda_mat = np.tile(np.array(lambda_), (N_theta, 1)).reshape((1, N_theta, len(lambda_)))
	last_k = (2 * torch.pi * n[-1] / lambda_mat).reshape((1, N_theta, N_lambda))
	u = last_k * np.sin(theta_mat)

	r_s_up = R_eff(n[1:]**2, torch.cat((d[1:], torch.tensor([0.0]))), u, lambda_mat, 's')
	
	r_p_up = R_eff(n[1:]**2, torch.cat((d[1:], torch.tensor([0.0]))), u, lambda_mat, 'p')
	r_s_down = R_eff(np.flipud(n[:-1]**2), torch.flipud(torch.cat((torch.tensor([0.0]), d[:-1]))), u, lambda_mat, 's')
	r_p_down = R_eff(np.flipud(n[:-1]**2), torch.flipud(torch.cat((torch.tensor([0.0]), d[:-1]))), u, lambda_mat, 'p')

	t_s_up = T_eff(n[1:]**2,  torch.cat((d[1:], torch.tensor([0.0]))), u, r_s_up, lambda_mat, 's')
	t_p_up = T_eff(n[1:]**2,  torch.cat((d[1:], torch.tensor([0.0]))), u, r_p_up, lambda_mat, 'p')

	# Choosing only the relevant coeffs
	r_s_up = r_s_up[i_scint - 1, :, :]
	r_p_up = r_p_up[i_scint - 1, :, :]
	r_s_down = r_s_down[num_layers - i_scint - 2, :, :]
	r_p_down = r_p_down[num_layers - i_scint - 2, :, :]
	t_s_up = t_s_up[i_scint - 1, :, :]
	t_p_up = t_p_up[i_scint - 1, :, :]

	#r_s_up = np.transpose(np.repeat(r_s_up[:, :, None, :], max_Nz, axis=2), (0, 2, 1, 3))
	r_s_up = torch.unsqueeze(r_s_up, 2)  # Add a new axis at position 2
	r_s_up = r_s_up.repeat(1, 1, max_Nz, 1)  # Repeat along the new axis
	r_s_up = r_s_up.permute(0, 2, 1, 3)  # Transpose the specified dimensions 
	#r_p_up = np.transpose(np.repeat(r_p_up[:, :, None, :], max_Nz, axis=2), (0, 2, 1, 3))
	r_p_up = torch.unsqueeze(r_p_up, 2)  # Add a new axis at position 2
	r_p_up = r_p_up.repeat(1, 1, max_Nz, 1)  # Repeat along the new axis
	r_p_up = r_p_up.permute(0, 2, 1, 3)  # Transpose the specified dimensions
	#r_s_down = np.transpose(np.repeat(r_s_down[:, :, None, :], max_Nz, axis=2), (0, 2, 1, 3))
	r_s_down = torch.unsqueeze(r_s_down, 2)  # Add a new axis at position 2
	r_s_down = r_s_down.repeat(1, 1, max_Nz, 1)  # Repeat along the new axis
	r_s_down = r_s_down.permute(0, 2, 1, 3)  # Transpose the specified dimensions 
	#r_p_down = np.transpose(np.repeat(r_p_down[:, :, None, :], max_Nz, axis=2), (0, 2, 1, 3))
	r_p_down = torch.unsqueeze(r_p_down, 2)  # Add a new axis at position 2
	r_p_down = r_p_down.repeat(1, 1, max_Nz, 1)  # Repeat along the new axis
	r_p_down = r_p_down.permute(0, 2, 1, 3)  # Transpose the specified dimensions 
	#t_s_up = np.transpose(np.repeat(t_s_up[:, :, None, :], max_Nz, axis=2), (0, 2, 1, 3))
	t_s_up = torch.unsqueeze(t_s_up, 2)  # Add a new axis at position 2
	t_s_up = t_s_up.repeat(1, 1, max_Nz, 1)  # Repeat along the new axis
	t_s_up = t_s_up.permute(0, 2, 1, 3)  # Transpose the specified dimensions 
	#t_p_up = np.transpose(np.repeat(t_p_up[:, :, None, :], max_Nz, axis=2), (0, 2, 1, 3))
	t_p_up = torch.unsqueeze(t_p_up, 2)  # Add a new axis at position 2
	t_p_up = t_p_up.repeat(1, 1, max_Nz, 1)  # Repeat along the new axis
	t_p_up = t_p_up.permute(0, 2, 1, 3)  # Transpose the specified dimensions 

	## Calculating the enhancment for all scintillator's layers
	theta_mat = np.transpose(np.tile(np.expand_dims(theta, axis=(1, 2, 3)), (1, len(i_scint), max_Nz, N_lambda)), (1, 2, 0, 3))
	""" 	# Expand dimensions
	theta_expanded = theta.unsqueeze(1).unsqueeze(2).unsqueeze(3)
	# Repeat along the new axes
	theta_tiled = theta_expanded.repeat(1, len(i_scint), max_Nz, N_lambda)
	# Transpose the specified dimensions
	theta_mat = theta_tiled.permute(1, 2, 0, 3) """
	lambda_mat = np.transpose(np.tile(np.expand_dims(lambda_, axis=(1, 2, 3)), (1, len(i_scint), max_Nz, N_theta)), (1, 2, 3, 0))
	""" lambda_expanded = lambda_.unsqueeze(1).unsqueeze(2).unsqueeze(3)
	# Repeat along the new axes
	lambda_tiled = lambda_expanded.repeat(1, len(i_scint), max_Nz, N_lambda)
	# Transpose the specified dimensions
	lambda_mat = lambda_tiled.permute(1, 2, 0, 3) """
	k = np.tile(2 * torch.pi * n[-1], (len(i_scint), max_Nz, len(theta), N_lambda)) / lambda_mat
	k_scint = np.tile(2 * torch.pi * n[i_scint][:, None, None, None], (1, max_Nz, len(theta), N_lambda)) / lambda_mat

	without_transmission = False
	if without_transmission:
		k = k_scint

	if coupled:
		u = k * np.sin(theta_mat)
	else:  # inside
		u = k_scint * np.sin(theta_mat)

	last_l = np.sqrt(k**2 - u**2, dtype=np.complex128) # specifying complex here, matlab does it automatically
	l_scint_vec=np.sqrt(k_scint**2 - u**2)
	l_scint = torch.tensor(np.sqrt(k_scint**2 - u**2))
	#b=torch.tensor(b)
	if coupled:
		T_par_s_up = t_s_up * (1 + r_s_down * torch.exp(2j * l_scint * b)) / (1 - r_s_down * r_s_up * torch.exp(2j * l_scint * d_scint))
		T_par_p_up = t_p_up * (1 + r_p_down * torch.exp(2j * l_scint * b)) / (1 - r_p_down * r_p_up * torch.exp(2j * l_scint * d_scint))
		T_perp_p_up = t_p_up * (1 - r_p_down * torch.exp(2j * l_scint * b)) / (1 - r_p_down * r_p_up * torch.exp(2j * l_scint * d_scint))
		temp =torch.tensor( (1/4) * (k * last_l**2 / k_scint**3))

		Par_s_up = temp * (torch.tensor(k_scint**2 )* (n_scint / n[-1])**2 * torch.abs(T_par_s_up / l_scint * torch.exp(1j * l_scint * top))**2)
		Par_p_up = temp * torch.abs(T_par_p_up * torch.exp(1j * l_scint * top))**2
		Perp_p_up = temp * u**2 * torch.abs(T_perp_p_up / l_scint * torch.exp(1j * l_scint * top))**2

		f = (Par_s_up + Par_p_up + Perp_p_up)
		f = f * mask

	else:  # Purcell Factor
		R_perp_p_up = (r_p_down * (r_p_up * torch.exp(2j * l_scint * top) - 1)) / (1 - r_p_down * r_p_up * torch.exp(2j * l_scint * d_scint))
		R_perp_p_down = (r_p_up * (r_p_down * torch.exp(2j * l_scint * b) - 1)) / (1 - r_p_down * r_p_up * torch.exp(2j * l_scint * d_scint))
		R_par_s_up = (r_s_down * (r_s_up * torch.exp(2j * l_scint * top) + 1)) / (1 - r_s_down * r_s_up * torch.exp(2j * l_scint * d_scint))
		R_par_s_down = (r_s_up * (r_s_down * torch.exp(2j * l_scint * b) + 1)) / (1 - r_s_down * r_s_up * torch.exp(2j * l_scint * d_scint))
		R_par_p_up = (r_p_down * (r_p_up * torch.exp(2j * l_scint * top) + 1)) / (1 - r_p_down * r_p_up * torch.exp(2j * l_scint * d_scint))
		R_par_p_down = (r_p_up * (r_p_down * torch.exp(2j * l_scint * b) + 1)) / (1 - r_p_down * r_p_up * torch.exp(2j * l_scint * d_scint))
		temp = 1 / (2 *torch.tensor(k_scint) * l_scint)

		Par_s_up = temp * torch.tensor(u**2 )* (1 + R_perp_p_up*torch.exp(2j * l_scint * b) + R_perp_p_down*torch.exp(2j * l_scint * top))
		Par_p_up = temp * torch.tensor(k_scint**2) * (1 + R_par_s_up*torch.exp(2j * l_scint * b) + R_par_s_down*torch.exp(2j * l_scint * top))
		Perp_p_up = temp * torch.tensor(l_scint_vec**2)* (1 + R_par_p_up*torch.exp(2j * l_scint * b) + R_par_p_down*torch.exp(2j * l_scint * top))

		f = (Par_s_up + Par_p_up + Perp_p_up) * mask
	G,profile=Gz(f, d_tot, i_scint, control)
	if sum_on_z:
		f = torch.sum(torch.sum(torch.squeeze(G), dim=1), dim=0)
	if ret_profile:
		return f,profile
	else:
		return f