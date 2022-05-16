# Import of standard python libraries
import numpy as np
import os
import time
import corner
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import copy
import random

from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.constants as constants
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.image_util as image_util
from lenstronomy.Analysis.td_cosmography import TDCosmography
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
from lenstronomy.LightModel.Profiles.sersic import Sersic

import pickle

def save_info(obj):
    try:
        with open("info", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def save_models(obj,j):
    try:
        with open("model_"+str(j), "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def Gauss_bounded(mu,sigma,lower,upper):
    
    acceptable = True
    
    while acceptable:
        value = random.gauss(mu, sigma)
        if value < upper and value > lower:
            acceptable = False
    
    return value

################## Things we don't change #####################

## Cosmology

H_0 = 70
omega_mm = 0.3
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.)

## Data

#match the WFC on Hubble 
deltaPix = 0.05

psf_type = 'GAUSSIAN'  # 'GAUSSIAN', 'PIXEL', 'NONE'

#typical Hubble exposure time
exp_time = 1600.

#typical Hubble FWHM in pixels (see https://hst-docs.stsci.edu/)
fwhm = 2*deltaPix

#we fix the source amplitude, and then vary the background and quasar brightness in relation to this
source_amp = 1000

kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

# Choice of source type
source_model_list = ['SERSIC_ELLIPSE']
source_model_class = LightModel(source_model_list)

point_source_list = ['LENSED_POSITION']
point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[False])

################# Upper and Lower Bounds #######################

###cosmology

z_source_min = 0.1
z_source_max = 5

z_lens_frac_min = 0.01
z_lens_frac_max = 0.99

#to avoid having a lens with a higher redshift than the source,
#we determine the lens redshift by randomly sampling a fraction of the source redshift

##data

snr_min = 100      #the lowest acceptable value of the snr
snr_max = 10**4   #the highest acceptable value of the snr
snr_scale = 300    #this is the centre of the Gaussian distribution from which the snr is selected at max d_L
snr_sigma_frac = 0.5 #the standard distribution of the Gaussian, as a fraction of the mean

tE_fac = 5  #width of aperture is this number x Einstein radius in pixels

##lens model

#lens_models = ['EPL', 'NIE', 'SPEMD', 'SPEP']
lens_models = ['EPL']

lens_ellipticity_min = -0.1
lens_ellipticity_max = 0.1

#galaxy mass (in kg)
M_min = 11
M_max = 12

#velocity dispersion (in m/s)
v_disp_mean = 210*(10**3)
v_disp_sigma = 54*(10**3)

#smaller values are reasonable, but these become increasingly poorly resolved
theta_E_min = 0.4

#e.g. lensing and dynamics in two simple steps (Agnello 2013)
gamma_min = 1.5
gamma_max = 2.5

LOS_gamma_min = -0.1
LOS_gamma_max = 0.1

##source model

source_ellipticity_min = -0.1
source_ellipticity_max = 0.1

source_pos_min = -0.1
source_pos_max = 0.1

#half-light radius (in pc) (see e.g. Suess et al. 2019 Half-mass Radii of Quiescent and Star-forming Galaxies)
#R_min = 1000
#R_max = 6000
#(see Starburst Intensity Limit of Galaxies at z~5-6, Hathi 2007)
R_min = 500  #the lowest acceptable value of the source half-light radius
R_max = 7000 #the highest acceptable value of the source half-light radius
R_sigma = 0.3 #the standard distribution of the Gaussian, as a fraction of the mean

n_sersic_min = 0.5
n_sersic_max = 10
n_sersic_mu = 4
n_sersic_sigma = 2

##point source model

#the factor by which the source amplitude is multiplied by to give the point source amplitude
amp_factor_min = 50
amp_factor_max = 500

#time delays (from TDCosmo II)
dt_sigma_factor_min = 0.003
dt_sigma_factor_max = 0.25

n_models = int(input("Enter the number of lens models to be simulated: "))
n_models_use = 0

###################### choosing values #################################

for i in range(n_models):
    print(i)
    
    ###cosmology
    
    z_too_small = True
    
    while z_too_small:
        
        u = random.uniform(0, 1)
        
        comoving_s = (u**(1/3))*cosmo.comoving_distance(z_source_max)
        
        z_s = z_at_value(cosmo.comoving_distance, comoving_s)
        
        if z_s > z_source_min:
            z_too_small = False
    
    z_source = z_s
            
    M_lens = constants.M_sun*(10**random.uniform(M_min, M_max))
    v_disp = np.random.normal(v_disp_mean,v_disp_sigma)

    
    bad_values = True
    
    while bad_values:
        
        u = random.uniform(z_lens_frac_min, z_lens_frac_max)
        
        comoving_l = (u**(1/3))*cosmo.comoving_distance(z_s)
        
        z_lens = z_at_value(cosmo.comoving_distance, comoving_l)
        
        lensCosmo = LensCosmo(z_lens=z_lens, z_source=z_s, cosmo=cosmo)
        d_ls = lensCosmo.dds*constants.Mpc
        d_l = lensCosmo.dd*constants.Mpc
        d_s = lensCosmo.ds*constants.Mpc
        
        theta_E = (1/constants.arcsec)*np.sqrt(4*constants.G*M_lens*d_ls/(d_l*d_s*constants.c**2))
        
        #theta_E = (1/constants.arcsec)*4*np.pi*(v_disp**2/constants.c**2)*(d_ls/d_s)

        if theta_E > theta_E_min:
            print("Found theta_E")
            bad_values = False

    numPix = int(round(tE_fac*theta_E)/deltaPix)
     
    ##data
    
    snr_mu = snr_scale*((1+z_source_max)**4)*(1+z_s)**(-4)
    
    snr_sigma = snr_mu*snr_sigma_frac
    
    snr = Gauss_bounded(snr_mu, snr_sigma, snr_min, snr_max)
    
    background_rms = (1/snr)*source_amp

    ##lens model
    
    dominant_model = random.choice(lens_models)

    lens_ellipticity = [random.uniform(lens_ellipticity_min, lens_ellipticity_max), random.uniform(lens_ellipticity_min, lens_ellipticity_max)]

    gamma = random.uniform(gamma_min,gamma_max)

    gamma_od = np.array([random.uniform(LOS_gamma_min, LOS_gamma_max), random.uniform(LOS_gamma_min, LOS_gamma_max)])
    gamma_os = np.array([random.uniform(LOS_gamma_min, LOS_gamma_max), random.uniform(LOS_gamma_min, LOS_gamma_max)])
    gamma_ds = np.array([random.uniform(LOS_gamma_min, LOS_gamma_max), random.uniform(LOS_gamma_min, LOS_gamma_max)])

    gl = gamma_os + gamma_od - gamma_ds
    
    gamma_los = gl
    
    ##source model
                    
    e1 = random.uniform(source_ellipticity_min, source_ellipticity_max)
    e2 = random.uniform(source_ellipticity_min, source_ellipticity_max)
    source_ellipticity = [e1,e2]
    
    source_radius_start = 1500*cosmo.H(4)/cosmo.H(z_s)

    source_radius_sigma = source_radius_start*R_sigma
    
    source_radius = Gauss_bounded(source_radius_start,source_radius_sigma,R_min,R_max)

    #conversion between source radius in pc and source radius in arcsec
    R_sersic = (source_radius/lensCosmo.ds)*(206265/10**6)

    n_sersic = Gauss_bounded(n_sersic_mu,n_sersic_sigma,n_sersic_min,n_sersic_max)
    
    ##point source model
    
    amp_factor = random.uniform(amp_factor_min,amp_factor_max)

##################### Setting up the lens models ##########################################3

    # Generate the coordinate grid and image properties
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms)
    data_class = ImageData(**kwargs_data)

    kwargs_data_E = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms)
    kwargs_data_Q = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms)
    kwargs_data_EQ = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms)

    data_class_E = ImageData(**kwargs_data_E)
    data_class_Q = ImageData(**kwargs_data_Q)
    data_class_EQ = ImageData(**kwargs_data_EQ)
    
    # Generate the PSF variables
    kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 3}
    psf_class = PSF(**kwargs_psf)

    # the lens model

    lens_mock_list = [dominant_model, 'LOS']

    # Parameters of the main lens
    # We define its centre as the line of sight
    ellipticity_lens = lens_ellipticity
    kwargs_epl = {'theta_E': theta_E,
                   'gamma': gamma,
                   'center_x': 0,
                   'center_y': 0,
                   'e1': ellipticity_lens[0],
                   'e2': ellipticity_lens[1]}

    # LOS shears

    kwargs_los = {
            'kappa_od': 0,
            'kappa_os': 0, 'kappa_ds': 0,
            'omega_od': 0,
            'omega_os': 0, 'omega_ds': 0,
            'gamma1_od': gamma_od[0], 'gamma2_od': gamma_od[1],
            'gamma1_os': gamma_os[0], 'gamma2_os': gamma_os[1],
            'gamma1_ds': gamma_ds[0], 'gamma2_ds': gamma_ds[1]}
        

    kwargs_lens = [kwargs_epl, kwargs_los]
    lens_mock_class = LensModel(lens_model_list=lens_mock_list, z_lens=z_lens, z_source=z_source, cosmo=cosmo)

    lensEquationSolver = LensEquationSolver(lens_mock_class)
    
    single_image = True
    
    
    count = 0
    
    while single_image:
        
        count += 1
        
        ra_source = random.uniform(source_pos_min, source_pos_max)
        dec_source = random.uniform(source_pos_min, source_pos_max)
        
        x_image, y_image = lensEquationSolver.findBrightImage(ra_source, dec_source, kwargs_lens, numImages=4,
                                                          min_distance=deltaPix, search_window= numPix * deltaPix)
    
        if len(x_image) > 1:
            single_image = False
         
        if count > 10:
            break

    if single_image:
        print("theta_E = ", theta_E)
        print("no image")
        continue

    # Sersic parameters in the initial simulation
    e_sersic = source_ellipticity
    kwargs_sersic = {'amp': source_amp,
                     'R_sersic': R_sersic,
                     'n_sersic': n_sersic,
                     'e1': e_sersic[0],
                     'e2': e_sersic[1],
                     'center_x': ra_source,
                     'center_y': dec_source}

    kwargs_source = [kwargs_sersic]

    sersic = Sersic(SersicUtil)
    
    total_source_flux = sersic.total_flux(source_amp, R_sersic, n_sersic, e1=e1, e2=e2)
    
    # compute lensing magnification at image positions
    mag = lens_mock_class.magnification(x_image, y_image, kwargs=kwargs_lens)
    mag = np.abs(mag)  # ignore the sign of the magnification

    # perturb observed magnification due to e.g. micro-lensing
    mag_pert = np.random.normal(mag, 0.01, len(mag))

    ps_amp = amp_factor*total_source_flux
    
    point_amp = mag * ps_amp  # multiply by intrinsic quasar brightness (in counts/s)
    kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image,
                               'point_amp': point_amp}]  # quasar point source position in the lens plane and intrinsic brightness
        
    ## E
    
    imageModel_E = ImageModel(data_class_E, psf_class, lens_mock_class, source_model_class,
                               kwargs_numerics=kwargs_numerics)
    
    image_sim_E = imageModel_E.image(kwargs_lens, 
                                       kwargs_source) 

    #adding noise
    poisson = image_util.add_poisson(image_sim_E, exp_time=exp_time)
    bkg = image_util.add_background(image_sim_E, sigma_bkd=background_rms)
    image_sim_E = image_sim_E + bkg + poisson

    kwargs_data_E['image_data'] = image_sim_E
    data_class_E.update_data(image_sim_E)

    kwargs_model_E = {'lens_model_list': lens_mock_list, 
                     'source_light_model_list': source_model_list}

    ## Q

    imageModel_Q = ImageModel(data_class_Q, psf_class, lens_mock_class,
                                point_source_class=point_source_class, 
                                kwargs_numerics=kwargs_numerics)
    
    image_sim_Q = imageModel_Q.image(kwargs_lens, 
                                 kwargs_source=None,
                                 kwargs_lens_light=None,
                                 lens_light_add = False,
                                 kwargs_ps=kwargs_ps)

    #adding noise
    poisson = image_util.add_poisson(image_sim_Q, exp_time=exp_time)
    bkg = image_util.add_background(image_sim_Q, sigma_bkd=background_rms)
    image_sim_Q = image_sim_Q + bkg + poisson

    kwargs_data_Q['image_data'] = image_sim_Q
    data_class_Q.update_data(image_sim_Q)

    kwargs_model_Q = {'lens_model_list': lens_mock_list,
                'point_source_model_list': point_source_list}

    ## EQ

    imageModel_EQ = ImageModel(data_class_EQ, psf_class, lens_mock_class,
                                    source_model_class,
                                    point_source_class=point_source_class, 
                                    kwargs_numerics=kwargs_numerics)
    
    image_sim_EQ = imageModel_EQ.image(kwargs_lens, 
                                 kwargs_source,
                                 kwargs_lens_light=None,
                                 lens_light_add = False,
                                 kwargs_ps=kwargs_ps)

    #adding noise
    poisson = image_util.add_poisson(image_sim_EQ, exp_time=exp_time)
    bkg = image_util.add_background(image_sim_EQ, sigma_bkd=background_rms)
    image_sim_EQ = image_sim_EQ + bkg + poisson

    kwargs_data_EQ['image_data'] = image_sim_EQ
    data_class_EQ.update_data(image_sim_EQ)

    kwargs_model_EQ = {'lens_model_list': lens_mock_list, 
                     'source_light_model_list': source_model_list,
                'point_source_model_list': point_source_list}


############################### Time Delays ############################################

    td_cosmo = TDCosmography(z_lens, z_source, kwargs_model_Q, cosmo_fiducial=cosmo)

    # time delays, the unit [days] is matched when the lensing angles are in arcsec
    t_days = td_cosmo.time_delays(kwargs_lens, kwargs_ps, kappa_ext=0)

    # relative delays (observable). The convention is relative to the first image
    dt_days = t_days[1:] - t_days[0]
    
    # and errors can be assigned to the measured relative delays (full covariance matrix not yet implemented)
    dt_sigma = []
    
    #here, I want a random percentage error, and I want to apply this to each of the time delays
    for j in range(len(dt_days)):
        dt_sigma.append(dt_days[j]*random.uniform(dt_sigma_factor_min, dt_sigma_factor_max))
    
    # and here a realisation of the measurement with the quoted error bars
    dt_measured = np.random.normal(dt_days, dt_sigma)

    #the time delay distance is
    lensCosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
    ddt = lensCosmo.ddt


######################### Saving Data ################################################

    mod = {
      "z_source": z_source,
      "z_lens": z_lens,
      "background_rms": background_rms,
      "dominant_model": dominant_model,
      "lens_ellipticity": lens_ellipticity,
      "theta_E": theta_E,
      "gamma": gamma,
      "gamma_od": gamma_od,
      "gamma_os": gamma_os,
      "gamma_ds": gamma_ds,
      "gamma_los": gamma_los,
      "source_ellipticity": source_ellipticity,
      "R_sersic": R_sersic,
      "n_sersic": n_sersic,
      "amp_factor": amp_factor,
      "kwargs_data_E": kwargs_data_E,
      "kwargs_data_Q": kwargs_data_Q,
      "kwargs_data_EQ": kwargs_data_EQ,
      "data_class_E": data_class_E,
      "data_class_Q": data_class_Q,
      "data_class_EQ": data_class_EQ,
      "kwargs_psf": kwargs_psf,
      "psf_class": psf_class,
      "lens_mock_list": lens_mock_list,
      "ellipticity_lens": ellipticity_lens,
      "kwargs_epl": kwargs_epl,
      "kwargs_los": kwargs_los,
      "kwargs_lens": kwargs_lens,
      "lens_mock_class": lens_mock_class,
      "e_sersic": e_sersic,
      "kwargs_sersic": kwargs_sersic,
      "kwargs_source": kwargs_source,
      "ra_source": ra_source,
      "dec_source": dec_source,
      "x_image": x_image,
      "y_image": y_image,
      "kwargs_ps": kwargs_ps,
      "snr": snr,
      "image_sim_E": image_sim_E,
      "image_sim_Q": image_sim_Q,
      "image_sim_EQ": image_sim_EQ,
      "dt_measured": dt_measured,
      "dt_sigma": dt_sigma,
      "ddt": ddt,
      "numPix": numPix
    }

    save_models(mod,n_models_use)
    n_models_use += 1

info = {
  "H_0": H_0,
  "omega_mm": omega_mm,
  "cosmo": cosmo,
  "deltaPix": deltaPix,
  "psf_type": psf_type,
  "exp_time": exp_time,
  "fwhm": fwhm,
  "source_amp": source_amp,
  "kwargs_numerics": kwargs_numerics,
  "source_model_list": source_model_list,
  "point_source_list": point_source_list,
  "n_models_use": n_models_use
  }

save_info(info)
print("The number of usable models is ", n_models_use)
                          
