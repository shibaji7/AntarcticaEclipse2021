#!/usr/bin/env python

"""build_ionospheric_model.py: module is dedicated to build foF2 model from fitacf data."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import matplotlib.pyplot as plt
import os
import datetime as dt
import pandas as pd
import numpy as np
from astropy import modeling
from scipy.optimize import curve_fit

from get_sd_data import *

def fit_lambda(du, power_drop, tfreq, elv, xlabel="srange", ylabel="p_l", plot=True, fname="images/_out.png"):
    x, y = du[xlabel], du[ylabel]
    def _1gaussian(xx, a0, c0, s0 ):
        return a0 * (1/(s0*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((xx-c0)/s0)**2)))
    def _a_gaussian(xx, a0, c0, s0, sp ):
        return _1gaussian(xx, a0, c0, s0 ) * (0.5 + (np.arctan(sp*(xx-c0))/np.pi))
    def opt(f, x0, y0, p0):
        popt, pcov = curve_fit(f, x0, y0, p0=p0)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    def estimate_skip_distance(popt):
        xx = np.linspace(0,popt[1],30000)
        yy = _a_gaussian(xx, popt_ag[0], popt_ag[1], popt_ag[2], popt_ag[3])
        print("\n Power drop {}".format(power_drop))
        sd = np.round(xx[np.argmin(np.abs(yy - power_drop))], 1)
        return sd
    try:
        popt_g, _ = opt(_1gaussian, x, y, p0=[np.max(y), np.mean(x), np.std(x)])
        popt_ag, _ = opt(_a_gaussian, x, y, p0=[np.max(y), np.mean(x), np.std(x), 0.1])
        skip_distance = estimate_skip_distance(popt_ag)
        if plot:
            fig = plt.figure(dpi=120,figsize=(3,3))
            ax = fig.add_subplot(111)
            ax.bar(x, y, width=30, color="y", ec="k", lw=0.3)
            xx = np.arange(3500)
            ax.plot(xx, _1gaussian(xx, popt_g[0], popt_g[1], popt_g[2]), color="b", lw=0.8, )
            ax.plot(xx, _a_gaussian(xx, popt_ag[0], popt_ag[1], popt_ag[2], popt_ag[3]), color="r", lw=0.8, )
            ax.text(0.8,0.8,r"$x_0$={} km".format(skip_distance),ha="center", va="center",transform=ax.transAxes)
            ax.text(0.2,0.8,r"$\delta_0={}^o$".format(elv),ha="center", va="center",transform=ax.transAxes)
            ax.axvline(skip_distance, color="cyan",lw=0.8)
            ax.set_xlabel("Slant Range, km")
            ax.set_ylabel("Power, db")
            ax.set_xlim(0, 3500)
            ax.set_ylim(0, 30)
            ax.set_title(r"$foF_2^o$={} MHz".format(tfreq))
            fig.savefig(fname, bbox_inches="tight")
            plt.close()
    except:
        import traceback
        traceback.print_exc()
        skip_distance = np.nan
        fig = plt.figure(dpi=120,figsize=(3,3))
        ax = fig.add_subplot(111)
        ax.bar(x, y, width=30, color="y", ec="k", lw=0.3)
        ax.text(0.2,0.8,r"$\delta_0={}^o$".format(elv),ha="center", va="center",transform=ax.transAxes)
        ax.axvline(skip_distance, color="cyan",lw=0.8)
        ax.set_xlabel("Slant Range, km")
        ax.set_ylabel("Power, db")
        ax.set_xlim(0, 3500)
        ax.set_ylim(0, 30)
        ax.set_title(r"$foF_2^o$={} MHz".format(tfreq))
        fig.savefig(fname.replace(".png", "_e.png"), bbox_inches="tight")
        plt.close()
    return skip_distance

def build_oblique_foF2_observed_by_radar(rad="cvw", dates=[dt.datetime(2017,8,21), dt.datetime(2017,8,21)], bmnum=13, scan_num=1,
                                        remove_first_range=1000, power_drop=10.):
    """ Estimate MUF from the GS power """
    fdata = FetchData( rad, [dates[0], dates[1]] )
    s_params = ["noise.sky", "tfreq", "frang", "rsep", "scan", "bmnum"]
    v_params = ["slist", "v", "w_l", "p_l", "elv"]
    _, scans = fdata.fetch_data(by="scan", s_params=s_params, v_params=v_params)
    beams = []
    # Resacle tx_frequency and estimate slant range 
    for scan in scans:
        for beam in scan.beams:
            setattr(beam, "slant_range", beam.frang + np.array(beam.slist.tolist()) * beam.rsep)
            setattr(beam, "tfreq", np.round(beam.tfreq/1e3,1))
            beams.append(beam)
    # Extract oblique foF2 or MUF scan by scan
    print("\n Data will be averaged over {} scans".format(scan_num))
    print(" Processing beam {}".format(bmnum))
    print(" Remove first range {}".format(remove_first_range))
    print(" Power drop {}".format(power_drop))
    skip_distance, o_foF2 = [], []
    for i in range(len(scans)-scan_num):
        rscan = scans[i:i+scan_num]
        p_l, srange, tfrq, angle = [], [], [], []
        for scan in rscan:
            for beam in scan.beams:
                if beam.bmnum == bmnum:
                    p_l.extend(beam.p_l.tolist())
                    srange.extend(beam.slant_range.tolist())
                    tfrq.append(beam.tfreq)
                    angle.extend(beam.elv.tolist())
        du = pd.DataFrame()
        du["p_l"], du["srange"] = p_l, srange
        du = du[du.srange>remove_first_range]
        fname = "images/{}.png".format(rscan[0].stime.strftime("%Y-%m-%d-%H-%M"))
        sd = fit_lambda(du, power_drop, tfreq=np.mean(tfrq), elv=np.mean(angle).round(1), xlabel="srange", ylabel="p_l", fname=fname)
        skip_distance.append(sd)
        o_foF2.append(np.mean(tfrq))
        #break
    #print(slant_range, o_foF2)
    return

def occultation_functions(z, a1=np.pi/2, a2=1):
    fn_sech = lambda x, a0: 1 - 0.5*(2/(np.exp(a0*x) + np.exp(-a0*x)))
    fn_tanh = lambda x, a0: np.tanh(a0*x)**2
    fig = plt.figure(figsize=(4,3), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(z, fn_sech(z,a1), lw=0.7, color="r", label="Sech")
    ax.plot(z, fn_tanh(z,a2), lw=0.7, color="b", label="Tanh")
    ax.set_ylim(0,1)
    ax.set_xlim(z[0],z[-1])
    ax.legend(loc=3)
    ax.set_ylabel("Occultation (%)")
    ax.set_xlabel("Time till tolality (Hours)")
    fig.savefig("images/ocultation.png", bbox_inches="tight")
    return

if __name__ == "__main__":
    build_oblique_foF2_observed_by_radar(dates=[dt.datetime(2017,8,21,15), dt.datetime(2017,8,21,15,30)])
    #occultation_functions(np.linspace(-5,5,1+3600*10))