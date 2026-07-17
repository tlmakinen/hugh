import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.special import sph_harm

class RadialBasisFunction(nn.Module):
    """ Gaussian Radial Basis Function """
    def __init__(self, num_radial, sigma=1.0):
        super(RadialBasisFunction, self).__init__()
        self.centers = nn.Parameter(torch.linspace(0, 1, num_radial))
        self.sigma = sigma

    def forward(self, r):
        # Apply Gaussian radial basis function
        return torch.exp(-0.5 * ((r[..., None] - self.centers) ** 2) / (self.sigma ** 2))


class SphericalHarmonics(nn.Module):
    """ Spherical Harmonics up to degree `l_max` """
    def __init__(self, l_max):
        super(SphericalHarmonics, self).__init__()
        self.l_max = l_max
        self.coefficients = nn.ParameterList([nn.Parameter(torch.randn(2 * l + 1)) for l in range(l_max + 1)])

    def forward(self, theta, phi):
        harmonics = []
        for l in range(self.l_max + 1):
            for m in range(-l, l + 1):
                # Calculate real part of spherical harmonics using SciPy (for simplicity)
                Y_l_m = sph_harm(m, l, phi, theta).real  # Compute Y_l^m
                harmonics.append(Y_l_m * self.coefficients[l][m + l])  # Apply learned coefficient
        return torch.stack(harmonics, dim=-1).sum(dim=-1)  # Sum over all harmonics


class MultipoleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, l_max=2, num_radial=3, sigma=1.0):
        super(MultipoleConv3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.l_max = l_max

        # Radial and angular components
        self.radial_func = RadialBasisFunction(num_radial=num_radial, sigma=sigma)
        self.spherical_harmonics = SphericalHarmonics(l_max=l_max)
        
        # Linear layer to learn channel mixing weights
        self.channel_mixer = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Compute 3D grid for spherical coordinates (separate from the input tensor)
        device = x.device
        grid_range = torch.linspace(-1, 1, self.kernel_size, device=device)
        z, y, x_grid = torch.meshgrid(grid_range, grid_range, grid_range, indexing="ij")
        
        r = torch.sqrt(x_grid ** 2 + y ** 2 + z ** 2)
        theta = torch.acos(z / (r + 1e-8))
        phi = torch.atan2(y, x_grid)

        # Compute radial basis and spherical harmonics components
        radial_component = self.radial_func(r)  # Shape: [kernel_size, kernel_size, kernel_size, num_radial]
        angular_component = self.spherical_harmonics(theta, phi)  # Shape: [kernel_size, kernel_size, kernel_size]

        # Combine radial and angular components to form the multipole kernel
        multipole_kernel = (radial_component * angular_component[..., None]).sum(dim=-1)
        
        # Reshape multipole kernel to fit conv3d dimensions: [out_channels, in_channels, kernel_size, kernel_size, kernel_size]
        multipole_kernel = multipole_kernel.view(1, 1, self.kernel_size, self.kernel_size, self.kernel_size)
        multipole_kernel = multipole_kernel.repeat(self.out_channels, self.in_channels, 1, 1, 1)  # Expand to output and input channels

        # Apply kernel to the actual input tensor `x`
        output = F.conv3d(x, multipole_kernel, padding=self.padding)
        
        # Apply channel mixing to get output channels
        return self.channel_mixer(output)


# Testing the MultipoleConv3D Layer
model = MultipoleConv3D(in_channels=1, out_channels=2, kernel_size=5, l_max=2, num_radial=3, sigma=0.5)
model = model.double()  # Convert model to double precision
input_data = torch.rand(1, 1, 32, 32, 32).type(torch.DoubleTensor)  # Random 3D input
output = model(input_data)

print("Output shape:", output.shape)
