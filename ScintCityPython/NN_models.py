import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset,TensorDataset, DataLoader
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
        self.activation = nn.LeakyReLU().double()
        self.grid_dim=input_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.view(-1, self.grid_dim, self.grid_dim)
        # conv layers....

        return x
class Model1_thick(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model1_thick, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).double() 
        self.fc2 = nn.Linear(hidden_size+1, hidden_size).double()
        self.fc3 = nn.Linear(hidden_size, output_size).double() 
        self.activation = nn.LeakyReLU().double()
        self.activation_out = nn.ReLU().double()
        self.grid_dim=input_size

    def forward(self, x,total_thickness):
        inference_flag=False
        if x.dim()==1: #checks if no batches (optimization stage)
             total_thickness=total_thickness.unsqueeze(0)
             x=x.unsqueeze(0)
             inference_flag=True
        total_thickness = total_thickness.unsqueeze(1)

        x = self.fc1(x)
        x = self.activation(x)
        x = torch.cat((x, total_thickness), dim=1)
        x = self.fc2(x)
        x = self.activation(x)
        x= self.fc3(x)
        #x=self.activation_out(x)
        x = x.view(-1, self.grid_dim, self.grid_dim)
        # conv layers....
        if inference_flag: #checks if no batches (optimization stage)
             x=x[0,:,:]
        return x
    def get_name(self):
        return self.__class__.__name__
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
# ScintCityPython/NN_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# If you already have Model1/Model2/... keep them as-is here.
# This file only adds the new model + helper. Your old imports
# (Model1, Model2, ScintProcessStochasticModel, etc.) can remain.
# ---------------------------------------------------------

class AlwaysOnDropout(nn.Dropout):
    """Dropout that stays active even in eval() -> MC-Dropout."""
    def forward(self, input):
        # force training=True to keep randomness during eval
        return F.dropout(input, self.p, training=True)
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlwaysOnDropout(nn.Dropout):
    def forward(self, input):
        return F.dropout(input, self.p, training=True)

class ModelWithCNNPostDropout2D(nn.Module):
    """
    Vector (len=99) -> thickness modulation -> FC -> ReLU -> Dropout -> Linear to grid ->
    CNN head (no dropout) -> returns (grid_dim, grid_dim) ALWAYS.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        cnn_hidden_channels: int = 16,
        cnn_kernel_size: int = 3,
        p_dropout: float = 0.0,
        always_on_dropout: bool = False,
    ):
        super().__init__()
        assert input_size == 99, "Model expects input_size == 99."
        self.grid_dim = input_size

        # trunk
        self.fc1 = nn.Linear(input_size, hidden_size).double()
        self.activation = nn.ReLU().double()

        # dropout (applies only to linear features)
        self.p_dropout = float(p_dropout)
        if self.p_dropout > 0:
            self.dropout = AlwaysOnDropout(self.p_dropout) if always_on_dropout else nn.Dropout(self.p_dropout)
        else:
            self.dropout = None

        # project to grid for CNN
        self.to_grid = nn.Linear(hidden_size, self.grid_dim * self.grid_dim).double()

        # learnable thickness weighting vector
        self.thickness_weight = nn.Parameter(torch.ones(1, input_size, dtype=torch.float64))

        # CNN head (no dropout)
        pad = cnn_kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_hidden_channels, kernel_size=cnn_kernel_size, padding=pad).double(),
            nn.ReLU().double(),
            nn.Conv2d(cnn_hidden_channels, cnn_hidden_channels, kernel_size=cnn_kernel_size, padding=pad).double(),
            nn.ReLU().double(),
            nn.Conv2d(cnn_hidden_channels, 1, kernel_size=1, padding=0).double(),
        )

    def forward(self, x, thickness):
        """
        x: (99,) or (1,99) — batch size must be 1 if batched
        thickness: scalar or tensor broadcastable to (1,1)
        Returns: (grid_dim, grid_dim)
        """
        # normalize input shape and enforce batch_size == 1
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, 99)
        assert x.shape[0] == 1, "This model returns a 2D map and requires batch_size == 1."

        x = x.to(torch.float64)

        # thickness -> (1,1)
        if not torch.is_tensor(thickness):
            thickness = torch.tensor(thickness, dtype=torch.float64, device=x.device)
        thickness = thickness.to(dtype=torch.float64, device=x.device)
        if thickness.dim() == 0:
            thickness = thickness.unsqueeze(0)
        thickness = thickness.view(1, 1)  # (1,1)

        # broadcast thickness to (1,99)
        t = thickness.expand_as(x)

        # elementwise modulation
        mod_in = self.thickness_weight * (x * t)  # (1,99) float64

        # linear trunk
        out = self.fc1(mod_in)
        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.to_grid(out)  # (1, H*W)

        # reshape for CNN
        out = out.view(1, 1, self.grid_dim, self.grid_dim)  # (1,1,H,W)

        # CNN head
        out = self.cnn(out)  # (1,1,H,W)

        # ALWAYS return 2D (H, W)
        return out[0, 0]  # (H, W)
class LocalMix2D(nn.Module):
    """
    Single 1→1 conv with a learnable k×k kernel, constrained to be a convex comb (softmax).
    Acts like a learnable smoothing / local mixer.
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd."
        self.k = kernel_size
        # unconstrained params -> softmax -> convex kernel
        self.logits = nn.Parameter(torch.zeros(1, 1, kernel_size, kernel_size, dtype=torch.float64))
        self.pad = kernel_size // 2

    def forward(self, x):  # x: (1,H,W) or (B,1,H,W)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        # softmax over kernel entries -> nonneg + sum=1
        w = F.softmax(self.logits.view(1, -1), dim=1).view(1, 1, self.k, self.k)
        w = w.to(dtype=torch.float64, device=x.device)
        y = F.conv2d(x, w, bias=None, padding=self.pad)
        return y.squeeze(0).squeeze(0) if y.shape[0] == 1 else y
class ModelWithThicknessVector(nn.Module):
    """
    New model (indicator mod=5).
    - Expects input_size == 100 (binary vector of length 100).
    - Learnable thickness_weight: shape (1, 100), dtype=float64.
    - Forward uses elementwise: thickness_weight ⊙ (x * thickness)
      where thickness is a scalar (broadcast to 100) or (batch,1).
    - Optional dropout between fc1 and fc2:
        * p_dropout>0 with nn.Dropout  -> classic dropout (off in eval())
        * p_dropout>0 with AlwaysOnDropout -> MC-Dropout (on in eval()).
    - Output is reshaped to (grid_dim, grid_dim) where grid_dim=input_size.
      So output_size must equal input_size**2.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 p_dropout: float = 0.0, always_on_dropout: bool = False):
        super().__init__()
        assert input_size == 99, "ModelWithThicknessVector expects input_size == 99."
        assert output_size == input_size * input_size, \
            "output_size must equal input_size**2 so we can view(grid_dim, grid_dim)."

        self.fc1 = nn.Linear(input_size, hidden_size).double()
        self.fc2 = nn.Linear(hidden_size, output_size).double()
        self.activation = nn.ReLU().double()
        self.grid_dim = input_size
        self.activation_output = nn.Softplus().double()

        # Learnable (1,100) vector (NOT dropped/pruned elsewhere)
        self.thickness_weight = nn.Parameter(torch.ones(1, input_size, dtype=torch.float64))

        self.p_dropout = float(p_dropout)
        if self.p_dropout > 0:
            self.dropout = AlwaysOnDropout(self.p_dropout) if always_on_dropout else nn.Dropout(self.p_dropout)
        else:
            self.dropout = None
                # 👉 Add local mixing module here
        self.localmix = LocalMix2D(kernel_size=3)  # kernel_size controls mixing radius
    def forward(self, x, thickness):
        """
        x: tensor (batch, 100) or (100,)
        thickness: scalar/tensor broadcastable to (batch, 1)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, 100)
        x = x.to(torch.float64)

        if not torch.is_tensor(thickness):
            thickness = torch.tensor(thickness, dtype=torch.float64, device=x.device)
        thickness = thickness.to(dtype=torch.float64, device=x.device)
        if thickness.dim() == 0:
            thickness = thickness.unsqueeze(0)  # (1,)
        thickness = thickness.view(-1, 1)  # (batch, 1)

        # broadcast thickness to (batch, 100)
        t = thickness.expand_as(x)

        # elementwise modulation
        mod_in = self.thickness_weight * (x * t)  # (batch, 100) in float64

        out = self.fc1(mod_in)
        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.fc2(out)
        out = self.activation_output(out)
        #out = self.activation_output(out)

        # reshape to 2D
        grid = out.view(self.grid_dim, self.grid_dim)

        # 👉 apply local mixing step
        #grid = self.localmix(grid)
        grid[:,(1-x[0]).bool()]=0
        return grid
    def get_name(self):
        return self.__class__.__name__



class StochasticLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(StochasticLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters for the mean and standard deviation of the weights
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_sigma_param = nn.Parameter(torch.randn(out_features, in_features) * -2.0)  # Log-space sigma
        
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma_param = nn.Parameter(torch.ones(out_features) * -2.0)  # Log-space sigma

    def forward(self, x,cell_size):
        # Ensure sigma is positive using softplus (sigma = log(1 + exp(param)))
        weight_sigma = F.softplus(self.weight_sigma_param)
        bias_sigma = F.softplus(self.bias_sigma_param)

        # Sample weights and biases from a Gaussian distribution
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = (self.bias_mu + bias_sigma * torch.randn_like(bias_sigma))*cell_size

        return F.linear(x, weight, bias)
class StochasticConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(StochasticConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Mean (\mu) and log-variance (\log\sigma^2)
        self.weight_mu = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.weight_log_sigma = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * -2.0)
        self.bias_mu = nn.Parameter(torch.zeros(out_channels))
        self.bias_log_sigma = nn.Parameter(torch.ones(out_channels) * -2.0)
    
    def forward(self, x):
        # Sample weights and biases from the Gaussian distribution
        weight_sigma = torch.exp(0.5 * self.weight_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        
        bias_sigma = torch.exp(0.5 * self.bias_log_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        
        return F.conv2d(x, weight, bias, padding=self.padding)

class ScintProcessStochasticModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ScintProcessStochasticModel, self).__init__()
        self.fc1 = StochasticLinear(input_size, hidden_size).double()
        self.fc2 = StochasticLinear(hidden_size, output_size).double()
        self.conv1 = StochasticConv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2).double()
        self.activation = nn.ReLU().double()
        self.grid_dim = input_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.view(-1, 1, self.grid_dim, self.grid_dim)
        x = self.conv1(x)
        return x

class ScintProcessStochasticModel_no_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ScintProcessStochasticModel_no_CNN, self).__init__()
        self.fc1 = StochasticLinear(input_size, hidden_size).double()
        self.fc2 = StochasticLinear(hidden_size, output_size).double()
        self.activation = nn.LeakyReLU().double()
        self.activation_output = nn.Softplus().double()
        #replace with GeLU?
        self.grid_dim = input_size

    def forward(self, x,cell_size=torch.Tensor(1)):
        out = self.fc1(x,cell_size)
        out = self.activation(out)
        out = self.fc2(out,cell_size)
        #out=self.activation_output(out)
        out = out.view(-1, self.grid_dim, self.grid_dim)
        #out = out.view(self.grid_dim, self.grid_dim)
        #out[:,(x[0]).bool()]=0
        return out
        """
class ScintProcessStochasticModel_no_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.1):
        super(ScintProcessStochasticModel_no_CNN, self).__init__()
        self.fc1 = StochasticLinear(input_size, hidden_size).double()
        self.fc2 = StochasticLinear(hidden_size, output_size).double()
        self.activation = nn.ReLU().double()
        #self.dropout_prob = dropout_prob
        self.grid_dim = input_size

    def forward(self, x):
        x = self.fc1(x)
        # Apply dropout only during training
        #x = F.dropout(x, p=self.dropout_prob, training=self.training) 
        x = self.activation(x)
        x = self.fc2(x)
        x = x.view(self.grid_dim, self.grid_dim)
        return x
"""
class PriorLayerModel(nn.Module):
    """
    Physics-shaped emitter map:
    Input:  thickness vector per sample of shape (B, N) [double]
    Params: per-layer (w0_i, w1_i) with positivity enforced (softplus)
    Output: depth-binned map of shape (B, Z) [or (B, 1, Z, R) if you set r_bins>1]
            For a bin at depth z that lies in layer i, value = w0_i * exp(-w1_i * z_local),
            where z_local is the depth measured from the start of that layer.
    """

    def __init__(self, n_layers: int, z_bins: int, r_bins: int ,r_max:float=250.0):
        super().__init__()
        self.fc1 = nn.Linear(2*n_layers, 2*n_layers).double() 
        self.fc2 = nn.Linear(2*n_layers, 2*n_layers).double() 
        self.activation = nn.LeakyReLU().double()
        self.n_layers = n_layers
        self.z_bins = z_bins
        self.r_bins = r_bins
        self.r_max=r_max
    def forward(self, thicknesses,layer_type):
        layer_type=layer_type.double()
        thicknesses=thicknesses.double()
        BATCH_SIZE = thicknesses.shape[0]
        NUM_LAYERS = self.n_layers
        Z_BINS = self.z_bins
        R_BINS = self.r_bins
        inference_flag=False

        # ----------------- 1. Inference and Parameter Extraction -----------------
        # ----------------- Input Shapes: [B, L], [B, D] ---------------------------

        # Use input tensors directly; avoid creating a new tensor with torch.tensor() if inputs are already tensors
        # However, if 'thicknesses' is coming from the DataLoader, it should already be a tensor on the correct device.
        # Assuming inputs are correctly typed and device-placed:

        if thicknesses.dim()==1: #checks if no batches (optimization stage)
             thicknesses=thicknesses.unsqueeze(0)
             layer_type=layer_type.unsqueeze(0)
             inference_flag=True
        x = torch.cat([thicknesses, layer_type], dim=-1)  # [B, L+D]
        x = torch.abs(self.fc1(x))                       # [B, 2 * L] 
        x = self.activation(x)
        x= self.fc2(x)
        # (Assuming your linear layer outputs 2 * Num_Layers per sample)

        # Separate w0 and w1 parameters for each sample in the batch
        # w0: [B, Num_Layers], w1: [B, Num_Layers]
        w0 = x[:, :NUM_LAYERS]
        w1 = x[:, NUM_LAYERS:]


        # ----------------- 2. Z-Coordinate and Layer Index Calculation -----------------

        # a) Calculate cumulative sum over the LAYER dimension (dim=1)
        # cum: [B, Num_Layers]
        cum = torch.cumsum(thicknesses, dim=1) 

        # b) Total thickness per sample (the last layer thickness for every sample)
        # total: [B]
        total = cum[:, -1].clamp_min(1e-12) 

        # c) Create the normalized Z-vector (0 to 1) and expand it (Broadcasting)
        # z_normalized: [Z_BINS]
        z_normalized = torch.linspace(
            0, 
            1, 
            Z_BINS, 
            device=total.device, 
            dtype=total.dtype
        )

        # z_vec: actual Z-coordinates for every sample in the batch [B, Z_BINS]
        z_vec = total.unsqueeze(1) * z_normalized 

        # d) Determine the Layer Index for Each Z-Bin (Vectorized Conditional Logic)
        # Reshape 'cum' from [B, L] to [B, L, 1]
        cum_expanded = cum.unsqueeze(2) 

        # Reshape 'z_vec' from [B, Z] to [B, 1, Z]
        z_vec_expanded = z_vec.unsqueeze(1) 

        # MASK: [B, Num_Layers, z_bins]. True where z_coord <= cumulative_thickness
        mask = z_vec_expanded <= cum_expanded

        # z_layer_indices: [B, z_bins]. Finds the index (layer_ind) of the FIRST True value (first layer boundary crossed)
        z_layer_indices = torch.argmax(mask.to(dtype=torch.int32), dim=1)


        # ----------------- 3. Final Output Matrix Calculation -----------------

        # The goal is an output matrix of shape [B, z_bins, r_bins]

        # a) Prepare Z_layer for Indexing w0/w1
        # Expand z_layer_indices from [B, Z_BINS] to [B, Z_BINS, R_BINS]
        # The -1 keeps the batch size dimension intact
        Z_layer = z_layer_indices.unsqueeze(2).expand(-1, -1, R_BINS).long()

        # b) Prepare R_mat
        # R_mat: [1, 1, R_BINS]. Use the batch size to expand to [B, Z_BINS, R_BINS]
        r_vec = torch.linspace(0, self.r_max, R_BINS, device=total.device, dtype=total.dtype)
        R_mat = r_vec.unsqueeze(0).unsqueeze(0).expand(BATCH_SIZE, Z_BINS, R_BINS)

        # c) Index w0 and w1 using Z_layer indices
        # Use torch.gather to select the correct w0/w1 parameter for every [Z, R] point in the batch
        # w0 is [B, L]. We need to select w0[b, l] where l = Z_layer[b, z, r].
        # The indices must be [B, Z_BINS, R_BINS] for gathering along dim=1 (the layer index dimension)
        w0_indexed = torch.gather(w0.unsqueeze(2).expand(-1, -1, Z_BINS), 1, Z_layer).squeeze(1) # [B, Z, R]
        w1_indexed = torch.gather(w1.unsqueeze(2).expand(-1, -1, Z_BINS), 1, Z_layer).squeeze(1) # [B, Z, R]

        # d) Calculate the exponential term
        output_mat = w0_indexed * torch.exp( w1_indexed * R_mat)
        output_mat=output_mat.transpose(1, 2)
        # The output is now a 3D batched tensor [B, Z, R]
        if (inference_flag):
            output_mat=output_mat[0,:,:]
        return output_mat
    

    def get_name(self):
        return self.__class__.__name__
# --- Model Definitions ---
"""
class PCAEncoder(nn.Module):
    Stage 1: Encodes the 99x99 grid to K*N parameters.
    def __init__(self, n_layers,k_value, z_bins, r_bins):
        super(PCAEncoder, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pca_linear = nn.Linear(r_bins*z_bins, n_layers * k_value,device=device).double()
        self.n_layers = n_layers
        self.k_value=k_value
        self.z_bins = z_bins
        self.r_bins = r_bins

    def forward(self, x):
        # Flatten (Batch_Size, 99, 99) -> (Batch_Size, 9801)
        x = x.view(x.size(0), -1) 
        compressed_params = self.pca_linear(x)
        # Reshape to (Batch_Size, K, N)
        return compressed_params.view(x.size(0), self.k_value, self.n_layers)

class PCADecoder(nn.Module):
    Stage 1: Decodes K*N parameters back to the 99x99 grid for autoencoder training.
    def __init__(self, n_layers,k_value, z_bins, r_bins):
        super(PCADecoder, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pca_linear = nn.Linear(n_layers * k_value,r_bins*z_bins,device=device ).double()
        self.positive_output = nn.ReLU()
        self.n_layers = n_layers
        self.z_bins = z_bins
        self.r_bins = r_bins
    def forward(self, x):
        # Flatten (Batch_Size, K, N) -> (Batch_Size, K*N)
        x = x.view(x.size(0), -1)
        compressed_params = self.pca_linear(x)
        compressed_params =self.positive_output(compressed_params)
        # Output (Batch_Size, 9801), then reshape to (Batch_Size, 99, 99)
        return compressed_params.view(x.size(0), self.r_bins, self.z_bins)


"""
class PCAEncoder(nn.Module):
    """
    Stage 1: Non-linear Encoder (Replaces PCA) for Dimensionality Reduction.
    """
    def __init__(self, n_layers, k_value, z_bins, r_bins, dropout_rate=0.2):
        super(PCAEncoder, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        INPUT_SIZE = r_bins * z_bins # 99 * 99 = 9801
        OUTPUT_SIZE = n_layers * k_value # The compressed space size
        HIDDEN_SIZE = INPUT_SIZE // 4 # E.g., 9801 / 4 ≈ 2450
        #HIDDEN_SIZE =INPUT_SIZE // 1_000
        self.n_layers = n_layers
        self.k_value = k_value
        
        # 1. First Layer (Feature Expansion/Reduction)
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE, device=device).double()
        self.activation = nn.LeakyReLU()
        
        # 2. Regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # 3. Final Compression to Latent Space (K*N)
        # This layer acts as the final bottleneck
        self.fc_out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE, device=device).double()

    def forward(self, x):
        # Flatten (Batch_Size, R, Z) -> (Batch_Size, R*Z)
        x = x.view(x.size(0), -1) 
        
        # Non-linear Encoding Block
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Final compressed latent vector
        compressed_params = self.fc_out(x)
        
        # Reshape to (Batch_Size, K, N)
        return compressed_params.view(x.size(0), self.k_value, self.n_layers)
    
class PCADecoder(nn.Module):
    """
    Stage 1: Non-linear Decoder (Replaces PCA) for Reconstruction.
    """
    def __init__(self, n_layers, k_value, z_bins, r_bins, dropout_rate=0.2):
        super(PCADecoder, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        INPUT_SIZE = n_layers * k_value
        OUTPUT_SIZE = r_bins * z_bins
        HIDDEN_SIZE = OUTPUT_SIZE // 4 # Example: Start expansion here
        #HIDDEN_SIZE = INPUT_SIZE // 1_000

        self.r_bins = r_bins
        self.z_bins = z_bins
        
        # 1. Initial Expansion (Linear + Non-linearity)
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE, device=device).double()
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # 2. Final Expansion to Grid Size
        self.fc_out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE, device=device).double()
        
        # 3. Positivity Enforcement: Softplus is a smoother alternative to ReLU
        # that ensures output > 0, necessary if your reconstructed grid must be non-negative.
        self.positive_output = nn.Softplus() 

    def forward(self, x):
        # Flatten (Batch_Size, K, N) -> (Batch_Size, K*N)
        x = x.view(x.size(0), -1)
        
        # Non-linear Decoding Block
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Final Linear Output (Pre-activation)
        reconstructed_flat = self.fc_out(x)
        
        # Enforce strict positivity (Softplus)
        reconstructed_flat = self.positive_output(reconstructed_flat)
        
        # Reshape to (Batch_Size, 99, 99)
        return reconstructed_flat.view(x.size(0), self.r_bins, self.z_bins)

class FCNPredictor(nn.Module):
    """
    Stage 2: Predicts K*N parameters from the N-layer vector.
    Features Dropout for regularization and Softplus for positive output.
    """
    def __init__(self, input_n: int, output_k: int, device=None, dropout_rate: float = 0.2):
        super(FCNPredictor, self).__init__()
        
        # input_n is N (number of layers)
        INPUT_SIZE = 3 * input_n 
        
        self.output_k = output_k
        self.output_n = input_n
        OUTPUT_SIZE = output_k * input_n
        
        # --- Adjusted Hidden Sizes for a deeper, more robust funnel ---
        HIDDEN_SIZE_1 = input_n * 8
        HIDDEN_SIZE_2 = input_n * 4
        
        # 1. First Layer
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1, device=device).double()
        self.dropout1 = nn.Dropout(dropout_rate) # Added Dropout

        # 2. Second Layer (Implicitly fc2 from your commented block)
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2, device=device).double()
        self.dropout2 = nn.Dropout(dropout_rate) # Added Dropout
        
        # 3. Output Layer (Maps to the final K*N size)
        self.fc3 = nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE, device=device).double()
        
        self.activation = nn.LeakyReLU()
        
    def forward(self, thicknesses, layer_type):
        # 0. Get the device from the input or the model's weights
        device = thicknesses.device
        
        # --- Feature Combination (Unchanged) ---
        type_one_hot = F.one_hot(layer_type.long(), num_classes=2).double().to(device) 
        
        if thicknesses.dim() == 2:
            thickness_reshaped = thicknesses.unsqueeze(-1).double()
        else:
            thickness_reshaped = thicknesses.double()
            
        combined_features = torch.cat((thickness_reshaped, type_one_hot), dim=-1)
        x = combined_features.view(combined_features.size(0), -1) 
        
        # --- FCN FORWARD PASS (Deeper architecture with Dropout) ---

        # FC 1 -> Activation -> Dropout
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x) 
        
        # FC 2 -> Activation -> Dropout
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        # FC 3 (Output Layer)
        predicted_params_flat = self.fc3(x)
        
        # --- Enforce Positivity using Softplus ---
        # Softplus is a smooth, differentiable approximation of ReLU that ensures output > 0
        #positive_output = F.softplus(predicted_params_flat)
        
        # Reshape to (Batch_Size, K, N)
        return predicted_params_flat.view(predicted_params_flat.size(0), self.output_k, self.output_n)
    
class CompleteAutoenconder_Model(nn.Module):

    def __init__(self, nlayers=10,kparam=10, z_dim=99, r_dim=99):
        super(CompleteAutoenconder_Model, self).__init__()
        self.FCPred=FCNPredictor(input_n=nlayers, output_k=kparam)
        self.decoder=PCADecoder(nlayers,kparam, z_dim, r_dim)
        #self.eval()
        
    def forward(self, thicknesses, layer_type):
        
        #layer_type_batches=layer_type.unsqueeze(0)
        #thicknesses_batches=thicknesses.unsqueeze(0)
        #predicted_denisty=self.decoder(self.FCPred(thicknesses_batches,layer_type_batches))
        #predicted_denisty=predicted_denisty[0,:,:]
        #return predicted_denisty

        predicted_denisty=self.decoder(self.FCPred(thicknesses,layer_type))
        return predicted_denisty
    def get_name(self):
        return self.__class__.__name__


if __name__ == "__main__":
    N = 5        # number of layers
    Z = 256      # depth bins
    R =10      # set >1 if you want a (B,1,Z,R) map for conv2d blocks

    model = LayerExpModel(n_layers=N, z_bins=Z, r_bins=R).double()

    # Batch of 3 structures (thicknesses in arbitrary units)
    thick = torch.tensor([50., 30., 20., 10., 40.]
                          , dtype=torch.double)
    type = torch.tensor([1., 0., 1., 0., 1.]
                          , dtype=torch.double)
    y = model(thick,type)  # shape (3, Z) or (3,1,Z,R)
    print(y.shape)


