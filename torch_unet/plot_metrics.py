
from astropy.cosmology import FlatLambdaCDM
from astropy import units
from astropy import constants as consts
import numpy as np

from scipy.signal.windows import blackmanharris
import matplotlib.pyplot as plt

cosmology = FlatLambdaCDM(H0=100, Om0=0.3) # H0=100 km/s/Mpc so in h units
nu_21 = 1420.405751768*units.MHz

#TODO: THIS NEEDS TO CHANGE IF WE CHANGE THE DATASET

freq_edges=np.linspace(500, 600, 128 + 1)
freqs = freq_edges[:-1] + np.diff(freq_edges[:2])[0] / 2

def z21(freq):
    freq = units.Quantity(freq, unit=units.MHz)
    return (nu_21/freq - 1).value

def k_perpendicular(bl_length, freq):
    freq = units.Quantity(freq, unit=units.MHz)
    bl_length = units.Quantity(bl_length, unit=units.m)
    wl = freq.to('m', equivalencies=units.spectral())
    k_pp = 2*np.pi*np.linalg.norm(bl_length)/wl/cosmology.comoving_transverse_distance(z21(freq))
    return k_pp.to('Mpc^-1')

def delays_from_freqs(freqs):
    freqs = units.Quantity(freqs, unit=units.MHz)
    d = (freqs[1]-freqs[0]).to('ns^-1')
    return np.fft.fftshift(np.fft.fftfreq(len(freqs), d=d))

def k_parallel(delays, freq):
    freq = units.Quantity(freq, unit=units.MHz)
    z = z21(freq)
    eta_to_kh = 2*np.pi*nu_21*cosmology.H(z)/consts.c/(1+z)**2
    k_parallel = (eta_to_kh*delays).to('Mpc^-1')
    return k_parallel
    
def wedge(theta_0, bl_ind, band_center=550*units.MHz):
    
    z = z21(band_center)
    wl = band_center.to('m', equivalencies=units.spectral()).value
    
    bl = baselines[bl_ind, ...]*units.m
    
    k_pp = k_perpendicular(np.linalg.norm(bl), band_center)
    k_par = theta_0.to('rad').value*k_pp*((cosmology.H(z)*cosmology.comoving_transverse_distance(z))/(consts.c*(1+z))).to('').value
    
    return k_par*0.7 # Double check h factor

def baseline_delay_spectrum(data, bl_ind, band_center=550*units.MHz):
    
    #with h5py.File(vis_fname, 'r') as fil:
    #    data = fil['vis'][bl_ind, ...] # bl, freq, ras
    
    data = data[bl_ind]
        
    N = data.shape[0] # Number of freqs
    
    z = z21(band_center)
    eta_to_kh = 2*np.pi*nu_21*cosmology.H(z)/consts.c/(1+z)**2
    
    delays = delays_from_freqs(freqs)
    k_par = k_parallel(delays, band_center)
    window = blackmanharris(N)
    
    delay_spec = np.fft.fftshift(np.abs(np.fft.fft(data*window[:, None], axis=0))**2)*units.K**2
    
    return delays, k_par, delay_spec

def plot_delay_spectrum(data, bl_ind, band_center=550*units.MHz, wedge_angle=90*units.deg, rel_range=12):
    
    delays, k_par, delay_spec = baseline_delay_spectrum(data, bl_ind, band_center=band_center)
    
    ras = np.linspace(180, -180, delay_spec.shape[1] + 1)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    extent = (ras[0], ras[-1], k_par[0].value, k_par[-1].value)
    
    to_plot = np.log10(delay_spec.value[:, ::-1])
    vmin = to_plot.max() - rel_range
    
    im = ax.imshow(to_plot, vmin=vmin, aspect=100, extent=extent, origin='lower', cmap='magma')
    
    wedge_k_pp = wedge(wedge_angle, bl_ind, band_center=band_center)
    ax.axhline(wedge_k_pp.value, color='white', ls='--')
    ax.axhline(-wedge_k_pp.value, color='white', ls='--')
    
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel(r'$k_{\parallel}$ [h Mpc$^{-1}$]')
    
    plt.colorbar(im, label='[K$^2$]')
    
    return fig

def plot_delay_spectrum_of_residual(data1, data_true, bl_ind, band_center=550*units.MHz, wedge_angle=90*units.deg, rel_range=12):
    
    delays, k_par, delay_spec = baseline_delay_spectrum((data1 - data_true), bl_ind, band_center=band_center)
    
    delays, k_par, delay_spec2 = baseline_delay_spectrum((data_true), bl_ind, band_center=band_center)

    delay_spec /= delay_spec2
    
    
    ras = np.linspace(180, -180, delay_spec.shape[1] + 1)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    extent = (ras[0], ras[-1], k_par[0].value, k_par[-1].value)
    
    to_plot = delay_spec.value[:, ::-1] #np.log10(delay_spec.value[:, ::-1])
    vmin = to_plot.max() - rel_range
    
    im = ax.imshow(to_plot, aspect=100, vmin=-2, vmax=2, extent=extent, origin='lower', cmap="coolwarm")
    
    wedge_k_pp = wedge(wedge_angle, bl_ind, band_center=band_center)
    ax.axhline(wedge_k_pp.value, color='white', ls='--')
    ax.axhline(-wedge_k_pp.value, color='white', ls='--')
    
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel(r'$k_{\parallel}$ [h Mpc$^{-1}$]')
    
    plt.colorbar(im, label='error')
    
    return (fig,delay_spec)

def plot_residual_of_delay_spectrum(data1, data_true, bl_ind, band_center=550*units.MHz, wedge_angle=90*units.deg, rel_range=12):
    
    delays1, k_par, delay_spec1 = baseline_delay_spectrum(data1, bl_ind, band_center=band_center)
    
    delays2, k_par, delay_spec2 = baseline_delay_spectrum(data_true, bl_ind, band_center=band_center)
    
    delay_spec = (delay_spec1 - delay_spec2) / delay_spec2
    
    ras = np.linspace(180, -180, delay_spec.shape[1] + 1)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    extent = (ras[0], ras[-1], k_par[0].value, k_par[-1].value)
    
    #to_plot = np.log10(delay_spec.value[:, ::-1])
    to_plot = delay_spec.value[:, ::-1]
    vmin = to_plot.max() - rel_range
    
    im = ax.imshow(to_plot, aspect=100, vmin=-2, vmax=2, extent=extent, origin='lower', cmap="coolwarm")
    
    wedge_k_pp = wedge(wedge_angle, bl_ind, band_center=band_center)
    ax.axhline(wedge_k_pp.value, color='white', ls='--')
    ax.axhline(-wedge_k_pp.value, color='white', ls='--')
    
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel(r'$k_{\parallel}$ [h Mpc$^{-1}$]')
    
    plt.colorbar(im, label='error')
    
    return (fig,delay_spec)