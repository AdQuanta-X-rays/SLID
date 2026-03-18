#!/usr/bin/env python3
#import numpy as np

import torch
def compute_b(d, i_scint, N_theta, N_lambda, is_Gz, dz_Nz):
	if is_Gz:
		max_Nz = dz_Nz
		b = torch.zeros((len(i_scint), max_Nz))  # distance of dipole from closer bottom interface
		for i in range(len(i_scint)):
				delta = d[i_scint[i] - 1] / max_Nz
				b[i, 0:max_Nz] = torch.linspace(delta, d[i_scint[i] - 1] - delta, max_Nz)
	else:
		dz = dz_Nz
		Nz = torch.zeros(len(i_scint))
		max_Nz = torch.ceil(torch.max(d[i_scint - 1]) / dz).astype(int)
		b = torch.zeros((len(i_scint), max_Nz))  # distance of dipole from closer bottom interface
		for i in range(len(i_scint)):
			Nz[i] = len(torch.arange(dz, d[i_scint[i] - 1] - dz, dz))
			b[i, 0:Nz[i]] = torch.arange(dz, d[i_scint[i] - 1] - dz, dz)

	b = torch.tile(b[:, :, None, None], (1, 1, N_theta, N_lambda))
	return b, max_Nz