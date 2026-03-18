import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
# Replace with your actual file paths
type_loss='FarField'
#type_loss='total_photon'
if (type_loss=='FarField'):
    vmin, vcenter, vmax = 0.998, 1.0, 1.002
    dataMSE = "/home/nathan.regev/software/git_repo/loss_mapMSE.pkl"
    dataCHI = "/home/nathan.regev/software/git_repo/loss_mapchi.pkl"
    color='viridis'
else:
    vmin, vcenter, vmax = 0.8, 1.0, 1.2
    dataMSE = "/home/nathan.regev/software/git_repo/total photon_loss_mapMSE.pkl"
    dataCHI = "/home/nathan.regev/software/git_repo/total photon_loss_mapchi.pkl"
    color='inferno'

thickness_scint=np.linspace(150,300,100)
thickness_dialecric=np.linspace(150,300,100)
# Load the matrices
with open(dataMSE, 'rb') as f1:
    MSE = pickle.load(f1)

with open(dataCHI, 'rb') as f2:
    chi = pickle.load(f2)

# Ensure both are numpy arrays
MSE = np.array(MSE)
chi = np.array(chi)
"""
MSE=MSE/np.sum(MSE)
chi=chi/np.sum(chi)
"""
# Compute difference
diff =(np.sum(chi)/np.sum(MSE))*( MSE/(chi+1e-10) )

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im1 = axes[0].imshow(MSE,
                    extent=[thickness_scint[0], thickness_scint[-1], thickness_dialecric[0], thickness_dialecric[-1]],
                    origin='lower',
                    aspect='auto', 
                    cmap=color)
axes[0].set_title('MSE model ')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(chi,
                    extent=[thickness_scint[0], thickness_scint[-1], thickness_dialecric[0], thickness_dialecric[-1]],
                    origin='lower',
                    aspect='auto', 
                    cmap=color)
axes[1].set_title('chi model ')
plt.colorbar(im2, ax=axes[1])



# Define normalization centered at 1
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

im3 = axes[2].imshow(diff,
                     extent=[thickness_scint[0], thickness_scint[-1], thickness_dialecric[0], thickness_dialecric[-1]],
                     origin='lower',
                     aspect='auto', 
                     cmap='bwr',
                     norm=norm)
axes[2].set_title('ratio models')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()
plt.savefig(type_loss+'_model _loss difference.png')
plt.savefig(type_loss+'_model _loss difference.svg')
