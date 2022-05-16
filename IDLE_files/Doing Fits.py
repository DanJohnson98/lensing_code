# Import of standard python libraries
import numpy as np
import os
import time
import corner
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from chainconsumer import ChainConsumer
import copy
import random

from astropy.cosmology import FlatLambdaCDM
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
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Plots import chain_plot
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Sampling.parameters import Param

########################### Function for unpickling ###################################

import pickle
 
def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

        
############################ Reading in info ##########################################

info = load_object("info")

kwargs_numerics = info["kwargs_numerics"]
source_model_list = info["source_model_list"]
point_source_list = info["point_source_list"]
n_models_use = info["n_models_use"]
source_amp = info["source_amp"]

########################### Things which stay the same #################################

#setting up the modelling

num_source_model = len(source_model_list)

kwargs_likelihood_NTD = {'check_bounds': True,
                        'force_no_add_image': False,
                        'source_marg': False,
                        'image_position_uncertainty': 0.004,
                        'check_matched_source_position': True,
                        'source_position_tolerance': 0.001,
                        'source_position_sigma': 0.001,
                        'time_delay_likelihood': False,
                        }

kwargs_likelihood_TD = {'check_bounds': True,
                        'force_no_add_image': False,
                        'source_marg': False,
                        'image_position_uncertainty': 0.004,
                        'check_matched_source_position': True,
                        'source_position_tolerance': 0.001,
                        'source_position_sigma': 0.001,
                        'time_delay_likelihood': True,
                        }

walker_ratio = 10
fitting_kwargs_list = [['MCMC',
                        {'n_burn': 1000, 'n_run': 1000,
                         'walkerRatio': walker_ratio, 'sigma_scale': 1., 'threadCount': 1}]]


############################ Reading and Fitting models ################################

for i in range(n_models_use):

    print('')
    print('')
    print(i+1,'/', n_models_use)    
    print('')
    print('')

    ################### Reading in Model ################### 
    
    obj = load_object("model_"+str(i))
    
    x_image = obj["x_image"]
    y_image = obj["y_image"]
    kwargs_data_E = obj["kwargs_data_E"]
    kwargs_data_Q = obj["kwargs_data_Q"]
    kwargs_data_EQ = obj["kwargs_data_EQ"]
    kwargs_psf = obj["kwargs_psf"]
    dt_measured = obj["dt_measured"]
    dt_sigma = obj["dt_sigma"]
    theta_E = obj["theta_E"]
    gamma = obj["gamma"]
    ellipticity_lens = obj["ellipticity_lens"]
    gamma_od = obj["gamma_od"]
    gamma_los = obj["gamma_los"]
    gamma_ds = obj["gamma_ds"]
    e_sersic = obj["e_sersic"]
    ra_source = obj["ra_source"]
    dec_source = obj["dec_source"]
    R_sersic = obj["R_sersic"]
    n_sersic = obj["n_sersic"]
    ddt = obj["ddt"]
 
    ################### Setting up the modelling ################### 

    #the parameters for the different models

    lens_model_list = ['EPL','LOS_MINIMAL']

    kwargs_model_E = {'lens_model_list': lens_model_list,
                'source_light_model_list': source_model_list}

    kwargs_model_Q = {'lens_model_list': lens_model_list,
                    'point_source_model_list': point_source_list}
                              
    kwargs_model_EQ = {'lens_model_list': lens_model_list,
                    'source_light_model_list': source_model_list,
                    'point_source_model_list': point_source_list}

                               
    kwargs_constraints_E = {}

    kwargs_constraints_Q = {'num_point_source_list': [len(x_image)], 
                            'Ddt_sampling': False,
                                    }
                                    
    kwargs_constraints_EQ = {'joint_source_with_point_source': [[0, 0]],
                            'num_point_source_list': [len(x_image)], 
                            'Ddt_sampling': False,
                                    }


    kwargs_constraints_QT = {'num_point_source_list': [len(x_image)], 
                            'Ddt_sampling': True,
                                    }

    kwargs_constraints_EQT = {'joint_source_with_point_source': [[0, 0]],
                            'num_point_source_list': [len(x_image)], 
                            'Ddt_sampling': True,
                                    }

    multi_band_list_E = [[kwargs_data_E, kwargs_psf, kwargs_numerics]]
    multi_band_list_Q = [[kwargs_data_Q, kwargs_psf, kwargs_numerics]]
    multi_band_list_EQ = [[kwargs_data_EQ, kwargs_psf, kwargs_numerics]]

    kwargs_data_joint_E = {'multi_band_list': multi_band_list_E,
                            'multi_band_type': 'multi-linear'}

    kwargs_data_joint_Q = {'multi_band_list': multi_band_list_Q,
                         'multi_band_type': 'multi-linear'}

    kwargs_data_joint_EQ = {'multi_band_list': multi_band_list_EQ, 
                             'multi_band_type': 'multi-linear',
                        'time_delays_measured': dt_measured,
                        'time_delays_uncertainties': dt_sigma}

    kwargs_data_joint_QT = {'multi_band_list': multi_band_list_Q,
                         'multi_band_type': 'multi-linear',
                        'time_delays_measured': dt_measured,
                        'time_delays_uncertainties': dt_sigma}

    kwargs_data_joint_EQT = {'multi_band_list': multi_band_list_EQ, 
                             'multi_band_type': 'multi-linear',
                        'time_delays_measured': dt_measured,
                        'time_delays_uncertainties': dt_sigma}
          
    # Lens model

    fixed_lens = []
    kwargs_lens_init = []
    kwargs_lens_sigma = []
    kwargs_lower_lens = []
    kwargs_upper_lens = []

    # Parameters for the main-lens model
    # e_eff = e 
    fixed_lens.append({'center_x': 0, 'center_y': 0}) # fix the line of sight
    kwargs_lens_init.append({'theta_E': theta_E, 'gamma': gamma,
                             'e1': ellipticity_lens[0],
                             'e2': ellipticity_lens[1]})
    kwargs_lens_sigma.append({'theta_E': .0001, 'gamma': 0.1,
                              'e1': 0.01, 'e2': 0.01})
    kwargs_lower_lens.append({'theta_E': theta_E*0.5, 'gamma': gamma*0.5,
                              'e1': -0.7, 'e2': -0.7})
    kwargs_upper_lens.append({'theta_E': theta_E*1.5,  'gamma': gamma*1.5,
                              'e1': 0.7, 'e2': 0.7})

    # LOS corrections
    #gamma_los = gamma_os + gamma_od - gamma_ds
    fixed_lens.append({'kappa_od': 0, 'omega_od': 0, 'kappa_los': 0})
    los_keys = ['gamma1_od', 'gamma2_od', 'gamma1_los', 'gamma2_los', 'omega_los']
    kwargs_lens_init.append({'gamma1_od': gamma_od[0],
                             'gamma2_od': gamma_od[1],
                             'gamma1_los': gamma_los[0],
                             'gamma2_los': gamma_los[1],
                             'omega_los':0})
    kwargs_lens_sigma.append({key: 0.01 for key in los_keys})
    kwargs_lower_lens.append({key: -0.5 for key in los_keys})
    kwargs_upper_lens.append({key: 0.5 for key in los_keys})

    lens_params = [kwargs_lens_init,
                   kwargs_lens_sigma,
                   fixed_lens,
                   kwargs_lower_lens,
                   kwargs_upper_lens]

    # Source models

    fixed_source = []
    kwargs_source_init = []
    kwargs_source_sigma = []
    kwargs_lower_source = []
    kwargs_upper_source = []

    # adapt the initialisation of the model
    gamma_SPT = gamma_od - gamma_ds
    e_eff_sersic = e_sersic - gamma_SPT
    x_eff_sersic = (1 - gamma_SPT[0]) * ra_source - gamma_SPT[1] * dec_source
    y_eff_sersic = - gamma_SPT[1] * ra_source + (1 + gamma_SPT[0]) * dec_source

    fixed_source.append({})
    kwargs_source_init.append({'amp': source_amp,
                                'R_sersic': R_sersic,
                                'n_sersic': n_sersic,
                                'e1': e_eff_sersic[0],
                                'e2': e_eff_sersic[1],
                                'center_x': x_eff_sersic,
                                'center_y': y_eff_sersic})
    kwargs_source_sigma.append({'n_sersic': 0.001, 'R_sersic': 0.001,
                                'e1': 0.01, 'e2': 0.01,
                                'center_x': 0.0001, 'center_y': 0.0001})
    kwargs_lower_source.append({'e1': -0.7, 'e2': -0.7,
                                'R_sersic': R_sersic*0.5, 'n_sersic': n_sersic*0.5,
                                'center_x': -10, 'center_y': -10})
    kwargs_upper_source.append({'e1': 0.7, 'e2': 0.7,
                                'R_sersic': R_sersic*1.5, 'n_sersic': n_sersic*1.5,
                                'center_x': 10, 'center_y': 10})

    source_params = [kwargs_source_init, kwargs_source_sigma,
                     fixed_source, kwargs_lower_source, kwargs_upper_source]


    # Point source models

    ps_model_list = point_source_list

    fixed_ps = []
    kwargs_ps_init = []
    kwargs_ps_sigma = []
    kwargs_lower_ps = []
    kwargs_upper_ps = []

    fixed_ps.append({})
    kwargs_ps_init.append({'ra_image': x_image, 'dec_image': y_image})
    kwargs_ps_sigma.append({'ra_image': [0.01] * len(x_image), 'dec_image': [0.01] * len(x_image)})
    kwargs_lower_ps.append({'ra_image': -10 * np.ones_like(x_image), 'dec_image': -10 * np.ones_like(y_image)})
    kwargs_upper_ps.append({'ra_image': 10 * np.ones_like(x_image), 'dec_image': 10 * np.ones_like(y_image)})

    ps_params = [kwargs_ps_init, kwargs_ps_sigma,
                     fixed_ps, kwargs_lower_ps, kwargs_upper_ps]

    #Cosmo models

    cosmo_model_list = point_source_list

    fixed_cosmo = []
    kwargs_cosmo_init = {'D_dt': ddt}
    kwargs_cosmo_sigma = {'D_dt': 50}
    kwargs_lower_cosmo = {'D_dt': ddt*0.2}
    kwargs_upper_cosmo = {'D_dt': ddt*3}

    cosmo_params = [kwargs_cosmo_init, kwargs_cosmo_sigma,
                     fixed_cosmo, kwargs_lower_cosmo, kwargs_upper_cosmo]


    #the parameters for the different models

    kwargs_params_E = {'lens_model': lens_params,
                    'source_model': source_params}
    
    kwargs_params_Q = {'lens_model': lens_params,
                    'point_source_model': ps_params}

    kwargs_params_EQ = {'lens_model': lens_params,
                    'source_model': source_params,
                    'point_source_model': ps_params}

    kwargs_params_QT = {'lens_model': lens_params,
                    'point_source_model': ps_params,
                    'special': cosmo_params}

    kwargs_params_EQT = {'lens_model': lens_params,
                    'source_model': source_params,
                    'point_source_model': ps_params,
                    'special': cosmo_params}

    ################### The Fitting ################### 
        
    fitting_seq_E = FittingSequence(kwargs_data_joint_E, kwargs_model_E, kwargs_constraints_E, 
                                  kwargs_likelihood_NTD, kwargs_params_E)

    fitting_seq_Q = FittingSequence(kwargs_data_joint_Q, kwargs_model_Q, kwargs_constraints_Q, 
                                  kwargs_likelihood_NTD, kwargs_params_Q)

    fitting_seq_EQ = FittingSequence(kwargs_data_joint_EQ, kwargs_model_EQ, kwargs_constraints_EQ, 
                                          kwargs_likelihood_NTD, kwargs_params_EQ)

    fitting_seq_QT = FittingSequence(kwargs_data_joint_QT, kwargs_model_Q, kwargs_constraints_QT, 
                                  kwargs_likelihood_TD, kwargs_params_QT)

    fitting_seq_EQT = FittingSequence(kwargs_data_joint_EQT, kwargs_model_EQ, kwargs_constraints_EQT, 
                                          kwargs_likelihood_TD, kwargs_params_EQT)
 
    
    chain_list_E = fitting_seq_E.fit_sequence(fitting_kwargs_list)
    kwargs_result_E = fitting_seq_E.best_fit()

    chain_list_Q = fitting_seq_Q.fit_sequence(fitting_kwargs_list)
    kwargs_result_Q = fitting_seq_Q.best_fit()
                           
    chain_list_EQ = fitting_seq_EQ.fit_sequence(fitting_kwargs_list)
    kwargs_result_EQ = fitting_seq_EQ.best_fit()

    chain_list_QT = fitting_seq_QT.fit_sequence(fitting_kwargs_list)
    kwargs_result_QT = fitting_seq_QT.best_fit()   

    chain_list_EQT = fitting_seq_EQT.fit_sequence(fitting_kwargs_list)
    kwargs_result_EQT = fitting_seq_EQT.best_fit()


    ######################### Saving Data ################################################
     
    def save_fit(obj):
        try:
            with open("model_"+str(i)+"_fit", "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)


    obj = {
        "lens_model_list": lens_model_list,
        "kwargs_model_E": kwargs_model_E,
        "kwargs_model_Q": kwargs_model_Q,
        "kwargs_model_EQ": kwargs_model_EQ,
        "kwargs_constraints_E": kwargs_constraints_E,
        "kwargs_constraints_Q": kwargs_constraints_Q,
        "kwargs_constraints_EQ": kwargs_constraints_EQ,
        "kwargs_constraints_QT": kwargs_constraints_QT,
        "kwargs_constraints_EQT": kwargs_constraints_EQT,
        "kwargs_likelihood_NTD": kwargs_likelihood_NTD,
        "kwargs_likelihood_TD": kwargs_likelihood_TD,
        "multi_band_list_E": multi_band_list_E,
        "multi_band_list_Q": multi_band_list_Q,
        "multi_band_list_EQ": multi_band_list_EQ,
        "kwargs_data_joint_E": kwargs_data_joint_E,
        "kwargs_data_joint_Q": kwargs_data_joint_Q,
        "kwargs_data_joint_EQ": kwargs_data_joint_EQ,
        "kwargs_data_joint_QT": kwargs_data_joint_QT,
        "kwargs_data_joint_EQT": kwargs_data_joint_EQT,
        "kwargs_params_E": kwargs_params_E,
        "kwargs_params_Q": kwargs_params_Q,
        "kwargs_params_EQ": kwargs_params_EQ,
        "kwargs_params_QT": kwargs_params_QT,
        "kwargs_params_EQT": kwargs_params_EQT,
        "gamma_SPT": gamma_SPT,
        "e_eff_sersic": e_eff_sersic,
        "x_eff_sersic": x_eff_sersic,
        "y_eff_sersic": y_eff_sersic,
        "walker_ratio": walker_ratio,
        "fitting_kwargs_list": fitting_kwargs_list,
        "fitting_seq_E": fitting_seq_E,
        "fitting_seq_Q": fitting_seq_Q,
        "fitting_seq_EQ": fitting_seq_EQ,
        "fitting_seq_QT": fitting_seq_QT,
        "fitting_seq_EQT": fitting_seq_EQT,
        "chain_list_E": chain_list_E,
        "kwargs_result_E": kwargs_result_E, 
        "chain_list_Q": chain_list_Q,
        "kwargs_result_Q": kwargs_result_Q,  
        "chain_list_EQ": chain_list_EQ, 
        "kwargs_result_EQ": kwargs_result_EQ, 
        "chain_list_QT": chain_list_QT,
        "kwargs_result_QT": kwargs_result_QT,
        "chain_list_EQT": chain_list_EQT, 
        "kwargs_result_EQT": kwargs_result_EQT  
    }

    save_fit(obj)
