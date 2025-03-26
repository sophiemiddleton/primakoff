import numpy as np
import matplotlib.pyplot as plt

import os
import copy
from multiprocessing import Pool
from functools import partial
import tqdm
import time
from scipy import integrate

from scipy.interpolate import interp2d, griddata, interp1d


from mc import *
from lhe_output import create_LHE_file

def alp_width(ma, gag):
    """
    ALP decay width; ma = ALP mass, gag = ALP coupling to photons, defined by the interaction-gag/4 ALP F Ftilde
    """
    return ma**3 * gag**2 / (64. * np.pi)


def get_t_integral(ma, mN, A, Z, Egamma):
    s = s_term(mN,Egamma)
    if s <  (ma+mN)**2:
        return np.zeros(2)
    t1, t0 = t_bounds(ma, mN, Egamma)

    # the integrand is peaked near t0
    # choose a new lower bound on the integral that is not super far from the peak to avoid
    # integration issues
    # the factor 100000 is chosen by hand, and found to give a stable estimate without numerical errrors
    t1_approx = max(t1,100000.*t0)
    sigma, error = integrate.quad(t_distribution, t1_approx, t0, args = (ma, mN, A, Z, Egamma))
    #print(sigma,"\t",error)

    return np.array([sigma, error])


def direct_production_cross_section(N_mcpN, photons, ma, mN, A, Z):
    """
    Nmc is number of electron-target collisions simulated
    photons is an array of photon 4 vectors from all of those collisisons

    returns cross-section in cm^2
    """
    gag = 1. # 1/GeV
    aEM = 1/137.
    hbarc = 0.1973269804 *1e-13 #cm * GeV

    r  = 0.

    Egamma_list = photons[:,0]

    # Use some fraction of available CPU cores
    pool = Pool(int(os.cpu_count()/2))
    # partial can be used to specify function arguments that are common for all instances
    # if there are no such arguments you dont need to wrap in partial
    sigma_list = pool.map(partial(get_t_integral, ma, mN, A, Z), Egamma_list)
    pool.close()
    sigma_list = np.array(sigma_list)


    r = np.sum(sigma_list[:,0])/N_mcpN
    err = np.sum(sigma_list[:,1])/N_mcpN


    # this factor was taken out of sigma_gamma N to defined t_distribution
    sigma_coef = gag**2 * aEM * Z**2 / 8.
    result = {}
    result['sigma'] = sigma_coef * r * (hbarc)**2 #* n*T_g * sigma_pN
    result['error'] = sigma_coef * err * (hbarc)**2 #* n*T_g * sigma_pN
    return result

def bootstrap_sample(data, Nboots):
    rng = np.random.default_rng()

    boostrapped_data = data[rng.integers(0, data.shape[0], size=Nboots)]

    return boostrapped_data


def main():
    ldmx_photons = np.loadtxt('data_from_sophie.csv',delimiter=',')[:,1:] / 1000. # convert MeV to GeV
    len(ldmx_photons)

    ldmx_photons[np.random.randint(0,ldmx_photons.shape[0],size=10)]

    bootstrap_sample(ldmx_photons, 10)

    ldmx_photons_8GeV = np.loadtxt('data_from_sophie_8gev_50K.csv',delimiter=',')[:,1:] / 1000. # convert MeV to GeV

    len(ldmx_photons_8GeV)

    Nboots=1
    len(ldmx_photons_8GeV)/Nboots

    ebins = np.logspace(-3,np.log10(4),20)
    ebins = np.logspace(-3,np.log10(8),20)
    plt.hist(ldmx_photons[:,0],bins=ebins)

    plt.hist(ldmx_photons_8GeV[:,0],bins=ebins,alpha=0.4)
    Nboots = 100000
    plt.hist(bootstrap_sample(ldmx_photons_8GeV, Nboots)[:,0],weights=np.ones(Nboots)*len(ldmx_photons_8GeV)/Nboots,bins=ebins,histtype='step')
    plt.xscale('log')
    plt.yscale('log')

    len(ldmx_photons[ldmx_photons[:,0] > 1e-1])/len(ldmx_photons), len(ldmx_photons_8GeV[ldmx_photons_8GeV[:,0] > 2e-1])/len(ldmx_photons_8GeV)

    len(ldmx_photons_8GeV[ldmx_photons_8GeV[:,0] > 5e-2])

    ebins = np.logspace(-3,np.log10(8),20)
    plt.hist(ldmx_photons_8GeV[:,0],bins=ebins)
    plt.xscale('log')
    plt.yscale('log')

    """
    Generate events using the high-energy sample but boot-strap to have larger statistics
    Nboots samples will be generated, regardless of the actual number of viable photons in the original photon sample.
    """
    #ma_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    ma_list = [0.1]
    coupling =1e-4
    A = 183.84
    Z = 74 #Tungsten

    mN = 183.84 * 0.9314941 #For W, GeV
    hbarc = 1.97327e-13 # GeV mm


    N_mcpN = 50000 # actual number of pN collisions simulated
    N_photon_subset = len(ldmx_photons_8GeV) #20000 # number of photons to use to speed up calculation of cross-section

    Nboots = 200000 # number of samples to generate

    direct_events = []
    xsec_list_8_GeV = []
    for ma in ma_list:
        out_dir_name = "../primakoff_events_8_GeV/"+"m_" + str(int(np.floor(ma*1000.)))+"_g_"+str(coupling)
        out_lhe_fname = "unweighted_events.lhe"
        os.mkdir(out_dir_name)

        # cross-section and width computed for a fiducial value the coupling
        width = alp_width(ma, coupling)
        ctau = hbarc/width

        result_dict = {}
        result_dict['ma'] = ma
        result_dict['tau'] = ctau
        result_dict['sigma'] = direct_production_cross_section(N_mcpN*len(ldmx_photons_8GeV[:N_photon_subset])/len(ldmx_photons_8GeV), ldmx_photons_8GeV[:N_photon_subset], ma, mN, A, Z)

        # pick photons that can actually produce an ALP
        photon_mask = ldmx_photons_8GeV[:,0] > (ma**2 + 2.*ma*mN)/(2.*mN)
        viable_photons = ldmx_photons_8GeV[photon_mask]
        bootstrapped_viable_photons = bootstrap_sample(viable_photons, Nboots)

        result_dict['events'] = generate_primakoff_events_in_parallel(bootstrapped_viable_photons,ma, mN, A, Z,ctau, small_t_cut_over_t0=1000, print_output = True)
        direct_events.append(result_dict)
        xsec_list_8_GeV.append([ma, result_dict['sigma']['sigma']])
        run_info_str = "<runinfo>" + "\n" \
                 + "# Primakoff process: gamma N > a N\n" \
                 + "# ALP Mass [GeV] = " + str(ma) + "\n" \
                 + "# ALP width [GeV] (for gag = 1e-3/GeV) = " + str(width) + "\n" \
                 + "# ALP decay length [mm] " + str(ctau) + "\n" \
                 + "# Nucleus Mass [GeV]= " + str(mN) + "\n" \
                 + "# Nucleus A = " + str(A) + "\n" \
                 + "# Nucleus Z = " + str(Z) + "\n" \
                 + "# ALP decay time stored in the vtim (same units as decay length) entry of the record; use this to reconstruct vertex." + "\n" \
                 + "# Number of Events: " + str(len(result_dict['events'])) + "\n" \
                 + "# Integrated weight (for gag = 1e-3/GeV) [pb] : "+str(result_dict['sigma']['sigma']*1e36*1e-6) + "\n" + "</runinfo>" + "\n"
        print(run_info_str)
        create_LHE_file(ma, mN, result_dict['events'], out_dir_name+"/"+out_lhe_fname, header_str = run_info_str)
    xsec_list_8_GeV = np.array(xsec_list_8_GeV)
    print("number of 8GeV events",len(result_dict['events']))
    plt.plot(xsec_list_8_GeV[:,0],xsec_list_8_GeV[:,1],'-o')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("plot.pdf")

if __name__ == "__main__":
    main()
