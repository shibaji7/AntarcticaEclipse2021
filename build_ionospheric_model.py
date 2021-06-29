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
import matplotlib.dates as mdates
import os
import datetime as dt
import pandas as pd
import numpy as np
from astropy import modeling
from scipy.optimize import curve_fit

from get_sd_data import *

def smooth(x,window_len=51,window="hanning"):
    if x.ndim != 1: raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len: raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[int(d/2):-int(d/2)]
    return y

def fit_lambda(du, power_drop, tfreq, elv, xlabel="srange", ylabel="p_l", plot=True, fname="images/_out.png"):
    x, y = du[xlabel], du[ylabel]
    
    def _1gaussian(xx, a0, c0, s0):
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
    
    def estimate_skip_distance_1D_params(a0, c0, s0):
        xx = np.linspace(0,c0,30000)
        yy = _1gaussian(xx, a0, c0, s0)
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
        mx, mean, std = np.max(y), np.mean(x), np.std(x)
        skip_distance = estimate_skip_distance_1D_params(mx, mean, std)
        fig = plt.figure(dpi=120,figsize=(3,3))
        ax = fig.add_subplot(111)
        ax.bar(x, y, width=30, color="y", ec="k", lw=0.3)
        xx = np.arange(3500)
        ax.plot(xx, _1gaussian(xx, mx, mean, std), color="b", lw=0.8, )
        ax.text(0.8,0.8,r"$x_0$={} km".format(skip_distance),ha="center", va="center",transform=ax.transAxes)
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

def build_oblique_foF2_observed_by_radar(rad="cvw", dates=[dt.datetime(2017,8,21), dt.datetime(2017,8,21)], bmnum=11, scan_num=1,
                                        remove_first_range=800, remove_last_range=2500, power_drop=10., plot=True):
    """ Estimate MUF from the GS power """
    csv_fname = "data/{rad}_{start}_{end}.csv".format(rad=rad, start=dates[0].strftime("%Y.%m.%d.%H.%M"), 
                                                      end=dates[1].strftime("%Y.%m.%d.%H.%M"))
    if not os.path.exists(csv_fname):
        fdata = FetchData( rad, [dates[0], dates[1]] )
        s_params = ["noise.sky", "tfreq", "frang", "rsep", "scan", "bmnum"]
        v_params = ["slist", "v", "w_l", "p_l", "elv"]
        _, scans = fdata.fetch_data(by="scan", s_params=s_params, v_params=v_params)
        beams = []
        # Resacle tx_frequency and estimate slant range 
        for scan in scans:
            for beam in scan.beams:
                if len(beam.slist) > 0:
                    setattr(beam, "slant_range", beam.frang + np.array(beam.slist.tolist()) * beam.rsep)
                    setattr(beam, "tfreq", np.round(beam.tfreq/1e3,1))
                beams.append(beam)
        # Extract oblique foF2 or MUF scan by scan
        print("\n Data will be averaged over {} scans".format(scan_num))
        print(" Processing beam {}".format(bmnum))
        print(" Remove first range {}".format(remove_first_range))
        print(" Remove last range {}".format(remove_last_range))
        print(" Power drop {}".format(power_drop))
        skip_distance, o_foF2, time_start, time_end = [], [], [], []
        for i in range(len(scans)-scan_num):
            rscan = scans[i:i+scan_num]
            p_l, srange, tfrq, angle = [], [], [], []
            for scan in rscan:
                for beam in scan.beams:
                    if beam.bmnum == bmnum:
                        if len(beam.slist) > 0:
                            p_l.extend(beam.p_l.tolist())
                            srange.extend(beam.slant_range.tolist())
                            tfrq.append(beam.tfreq)
                            if type(beam.elv) is list: angle.extend(beam.elv)
                            else: angle.extend(beam.elv.tolist())
            du = pd.DataFrame()
            du["p_l"], du["srange"] = p_l, srange
            du = du[(du.srange>remove_first_range) & (du.srange<remove_last_range)]
            fname = "images/{}.png".format(rscan[0].stime.strftime("%Y-%m-%d-%H-%M"))
            if len(du) > 0:
                sd = fit_lambda(du, power_drop, tfreq=np.mean(tfrq), elv=np.mean(angle).round(1), 
                                xlabel="srange", ylabel="p_l", fname=fname, plot=plot)
                if sd != np.nan:
                    skip_distance.append(sd)
                    o_foF2.append(np.mean(tfrq))
                    time_start.append(rscan[0].stime)
                    time_end.append(rscan[-1].stime)
        df = pd.DataFrame()
        df["skip_distance"], df["o_foF2"], df["time_start"], df["time_end"] = skip_distance, o_foF2, time_start, time_end
        df.to_csv(csv_fname, index=False, header=True)
    else: df = pd.read_csv(csv_fname, parse_dates=["time_start", "time_end"])
    print(" Header:\n",df.head())
    return df

def build_occultation_functions(rad, dates, time_range, bmnum=11, scan_num=1, remove_first_range=500, remove_last_range=2500, 
                                power_drop=10., plot=True):
    df = build_oblique_foF2_observed_by_radar(rad, dates, bmnum, scan_num, remove_first_range, remove_last_range, power_drop, plot)
    upper = df[df.time_start <= time_range[0]]
    lower = df[df.time_start >= time_range[1]]
    
    def plot_rays(ax, u, color="r", ms=1, alpha=0.6, lw=1.5, wl=51):
        midnight = dates[0].replace(hour=0, minute=0, second=0, microsecond=0)
        sd = np.array(u.skip_distance)
        sd[sd<=200.] = np.nan
        u.skip_distance = sd
        dfx = u.interpolate(method="polynomial", order=1)
        print(" Modified(intp.) Header:\n",dfx.head())
        ax.plot(u.time_start, u.skip_distance, color+"o", lw=lw, markersize=ms, alpha=alpha)
        ax.plot(dfx.time_start, smooth(dfx.skip_distance, window_len=wl), lw=lw, color="k", ls="--")
        secs, vals = [(t.to_pydatetime()-midnight).seconds for t in dfx.time_start], smooth(dfx.skip_distance, window_len=wl).tolist()
        return secs, vals
    
    def estimate_interpolated(eupper, elower):
        middle_time = time_range[0] + dt.timedelta(seconds=(time_range[1]-time_range[0]).total_seconds())
        midnight = dates[0].replace(hour=0, minute=0, second=0, microsecond=0)
        middle_seconds = (middle_time-midnight).seconds
        print(midnight, middle_time, middle_seconds)
        start_seconds, end_seconds = middle_seconds-(30*60), middle_seconds+(30*60)
        from scipy import interpolate
#         x = eupper[0] + elower[0]
#         y = eupper[1] + elower[1]
#         fnc = interpolate.interp1d(x, y, kind="cubic")
#         midnight = dates[0].replace(hour=0, minute=0, second=0, microsecond=0)
#         new_secs = np.arange(np.min(x), np.max(x))
#         new_dates = [midnight + dt.timedelta(seconds=int(s)) for s in new_secs]
#         print(new_dates[0], new_dates[-1])
#         print(np.min(x), np.max(x), np.min(new_secs), np.max(new_secs))
#         Y = fnc(new_secs)
        return #Y, new_dates
    
    fig = plt.figure(figsize=(4,3), dpi=120)
    ax = fig.add_subplot(111)
    fmt = mdates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(fmt)
    est_upper = plot_rays(ax, upper, "r", wl=21)
    est_lower = plot_rays(ax, lower, "b", wl=21)
    estimate_interpolated(est_upper, est_lower)
    #ax.plot(new_dates, Y, lw=0.6, color="gray")
    ax.set_ylabel("Skip Distance (km)")
    ax.set_xlabel("Time (UT)")
    ax.set_xlim(dates[0], dates[1])
    ax.set_ylim(1000,3000)
    fig.autofmt_xdate()
    fig.savefig("images/ocultation.png", bbox_inches="tight")
    return

def occultation_functions(z, a1=np.pi/2, a2=1):
    fn_sech = lambda x, a0: 0.5*(2/(np.exp(a0*x) + np.exp(-a0*x)))
    fn_tanh = lambda x, a0: 1-np.tanh(a0*x)**2
    fig = plt.figure(figsize=(4,3), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(z, fn_sech(z,a1), lw=0.7, color="r", label="Sech")
    ax.plot(z, fn_tanh(z,a2), lw=0.7, color="b", label="Tanh")
    ax.set_ylim(0,1)
    ax.set_xlim(z[0],z[-1])
    ax.legend(loc=3)
    ax.set_ylabel("Occultation (%)")
    ax.set_xlabel("Time till tolality (Hours)")
    fig.savefig("images/ocultation_model.png", bbox_inches="tight")
    return

if __name__ == "__main__":
    #build_oblique_foF2_observed_by_radar(dates=[dt.datetime(2017,8,21,15), dt.datetime(2017,8,21,20)])
    #occultation_functions(np.linspace(-5,5,1+3600*10))
    #build_occultation_functions(rad="cvw", dates=[dt.datetime(2017,8,21,15), dt.datetime(2017,8,21,19)])
    #build_occultation_functions(rad="gbr", bmnum=7, dates=[dt.datetime(2021,6,9,9), dt.datetime(2021,6,9,10,30)],
    #                           remove_first_range=1000, remove_last_range=2500)
    build_occultation_functions(rad="gbr", bmnum=7, dates=[dt.datetime(2021,6,10,8), dt.datetime(2021,6,10,11)],
                               remove_first_range=1000, remove_last_range=2500, power_drop=5.,
                               time_range=[dt.datetime(2021,6,10,9,40), dt.datetime(2021,6,10,10,10)])
    occultation_functions(np.linspace(-5,5,1+3600*10))
    pass
