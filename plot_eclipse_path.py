import cartopy
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch

import numpy as np
import pydarn
import rad_fov

import os

def overlay_radar(rads, ax, _to, _from, color="k", zorder=2, marker="o", ms=2, font={"size":8, "color":"b", "weight":"bold"}, north=True):
    times = -1 if north else 1    
    for rad in rads:
        hdw = pydarn.read_hdw_file(rad)
        lat, lon, _ = hdw.geographic
        tlat, tlon = lat+3*times, lon+3*times
        x, y = _to.transform_point(lon, lat, _from)
        tx, ty = _to.transform_point(tlon, tlat, _from)
        ax.plot(x, y, color=color, zorder=zorder, marker=marker, ms=ms)
        ax.text(tx, ty, rad.upper(), ha="center", va="center", fontdict=font)
    return

def overlay_fov(rads, ax, _to, _from, maxGate=75, rangeLimits=None, beamLimits=None,
            model="IS", fov_dir="front", fovColor=None, fovAlpha=0.2,
            fovObj=None, zorder=2, lineColor="k", lineWidth=0.5, ls="-"):
    """ Overlay radar FoV """
    from numpy import transpose, ones, concatenate, vstack, shape
    for rad in rads:
        hdw = pydarn.read_hdw_file(rad)
        sgate = 0
        egate = hdw.gates if not maxGate else maxGate
        ebeam = hdw.beams
        if beamLimits is not None: sbeam, ebeam = beamLimits[0], beamLimits[1]
        else: sbeam = 0
        rfov = rad_fov.CalcFov(hdw=hdw, ngates=egate)
        x, y = np.zeros_like(rfov.lonFull), np.zeros_like(rfov.latFull)
        for _i in range(rfov.lonFull.shape[0]):
            for _j in range(rfov.lonFull.shape[1]):
                x[_i, _j], y[_i, _j] = _to.transform_point(rfov.lonFull[_i, _j], rfov.latFull[_i, _j], _from)
        contour_x = concatenate((x[sbeam, sgate:egate], x[sbeam:ebeam, egate],
                                 x[ebeam, egate:sgate:-1],
                                 x[ebeam:sbeam:-1, sgate]))
        contour_y = concatenate((y[sbeam, sgate:egate], y[sbeam:ebeam, egate],
                                 y[ebeam, egate:sgate:-1],
                                 y[ebeam:sbeam:-1, sgate]))
        ax.plot(contour_x, contour_y, color=lineColor, zorder=zorder, linewidth=lineWidth, ls=ls)
        if fovColor:
            contour = transpose(vstack((contour_x, contour_y)))
            polygon = Polygon(contour)
            patch = PolygonPatch(polygon, facecolor=fovColor, edgecolor=fovColor, alpha=fovAlpha, zorder=zorder)
            ax.add_patch(patch)
    return

def create_cartopy(prj):
    fig = plt.figure(dpi=140, figsize=(4,4))
    ax = fig.add_subplot(111, projection=prj)
    ax.add_feature(cartopy.feature.OCEAN, zorder=0, alpha=0.1)
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor="black", alpha=0.2, lw=0.3)
    ax.set_global()
    ax.gridlines(linewidth=0.3)
    return fig, ax

def convert_to_map_lat_lon(xs, ys, _from, _to):
    lat, lon = [], []
    for x, y in zip(xs, ys):
        _lon, _lat = _to.transform_point(x, y, _from)
        lat.append(_lat)
        lon.append(_lon)
    return lat, lon

def create_eclipse_path(fname, center, rads, title, north, pngfname, extend_lims=[50, 90], lsep=10.):
    f = pd.read_csv(fname, parse_dates=["Time"])
    geodetic = ccrs.Geodetic()
    orthographic = ccrs.Orthographic(center[0], center[1])
    fig, ax = create_cartopy(orthographic)
    lat, lon = convert_to_map_lat_lon(f.Center_Longitude, f.Center_Latitude, geodetic, orthographic)
    ax.plot(lon, lat, color="k", linewidth=1, ls="--", transform=orthographic)
    lat_n, lon_n = convert_to_map_lat_lon(f.South_Longitude, f.South_Latitude, geodetic, orthographic)
    ax.plot(lon_n, lat_n, color="b", linewidth=1, ls="-", transform=orthographic)
    lat_s, lon_s = convert_to_map_lat_lon(f.North_Longitude, f.North_Latitude, geodetic, orthographic)
    ax.plot(lon_s, lat_s, color="b", linewidth=1, ls="-", transform=orthographic)
    times = f.Time.tolist()    
    overlay_fov(rads, ax, orthographic, geodetic)
    overlay_radar(rads, ax, orthographic, geodetic, north=north)
    ax.set_title(title)
    ax.set_extent((-180, 180, extend_lims[0],extend_lims[1]), crs = ccrs.PlateCarree())
    for _k in range(int(len(lat_s)/lsep)):
        _k = int(lsep*_k)
        la_n, lo_n, la_s, lo_s = lat_n[_k], lon_n[_k], lat_s[_k], lon_s[_k]
        ax.plot([lo_n, lo_s], [la_n, la_s], color="r", linewidth=1, ls="--", transform=orthographic)
        x, y = orthographic.transform_point(lo_n, la_n, geodetic)
        ax.text(lo_n, la_n, times[_k].strftime("%H:%M"), ha="left", va="center",
                fontdict={"size":7, "color":"r", "weight":"bold"}, transform=orthographic)
    ax.plot([lon_n[-1], lon_s[-1]], [lat_n[-1], lat_s[-1]], color="r", linewidth=1, ls="-.", transform=orthographic)
    date = f.Time.tolist()[0]
    ax.add_feature(Nightshade(date, alpha=0.2))
    ax.text(1.01, 0.5, date.strftime("%H:%M:%S UT"), ha="left", va="center", transform=ax.transAxes, rotation=90)
    ax.text(lon_n[-1], lat_n[-1], times[-1].strftime("%H:%M"), ha="left", va="center", 
            fontdict={"size":7, "color":"r", "weight":"bold"}, transform=orthographic)
    ax.text(-0.01, 0.5, "Geographic Coordinate", ha="right", va="center", transform=ax.transAxes, rotation=90)
    fig.savefig(pngfname)
    return

def create_dec4_eclipse():
    fname, center, rads, title, north, pngfname = "data/Dec4Eclipse.csv", (-95, -90), ["fir", "hal", "san", "sys", "sps", "dce"],\
                            "4 December, 2021", False, "images/Dec4Eclipse_South.png"
    create_eclipse_path(fname, center, rads, title, north, pngfname, extend_lims=[-90, -50])
    #center, rads, north, pngfname = (-95, 60), ["bks", "gbr", "kap"], True, "images/Dec4Eclipse_North.png"
    #create_eclipse_path(fname, center, rads, title, north, pngfname)
    return

def create_june10_eclipse():
    fname, center, rads, title, north, pngfname = "data/June10Eclipse.csv", (-95, 90), ["gbr","cly", "inv"],\
                        "10 June, 2021", True, "images/June10Eclipse_North.png"
    create_eclipse_path(fname, center, rads, title, north, pngfname)
    center, rads, north, pngfname = (-95, -90), ["fir", "hal"], False, "images/June10Eclipse_South.png"
    #create_eclipse_path(fname, center, rads, title, north, pngfname, extend_lims=[-50, -90])
    return

def plot_fov(center=[-95, 90], rads=["wal", "bks", "fhe", "fhw", "cve", "cvw", ], north=True, 
             title="", pngfname = "data/DF2.png", extend_lims=[30, 90]):
    geodetic = ccrs.Geodetic()
    orthographic = ccrs.Orthographic(center[0], center[1])
    fig, ax = create_cartopy(orthographic)
    overlay_fov(rads, ax, orthographic, geodetic)
    overlay_radar(rads, ax, orthographic, geodetic, north=north)
    ax.set_title(title)
    ax.set_extent((-180, 180, extend_lims[0],extend_lims[1]), crs = ccrs.PlateCarree())
    ax.text(-0.01, 0.8, "Geographic Coordinate", ha="right", va="center", transform=ax.transAxes, rotation=90)
    fig.savefig(pngfname)
    return

def plot_run_mode1():
    os.system("rm -rf images/*mode1_*.png")
    fname, center, rads, title, north, pngfname = "data/Dec4Eclipse.csv", (-95, -90), ["fir"],\
                            "4 December, 2021", False, "images/Dec4Eclipse_South_mode1_%02d.png"
    extend_lims, lsep = [-90, -50], 10.
    f = pd.read_csv(fname, parse_dates=["Time"])
    geodetic = ccrs.Geodetic()
    orthographic = ccrs.Orthographic(center[0], center[1])
    
    times = f.Time.tolist()
    beams = [16, 13, 9, 5, 1] * 3
    colors = (["cyan"] * 5) + (["red"] * 5) + (["blue"] * 5)
    for bm, cl, i in zip(beams, colors, range(len(colors))):
        fig, ax = create_cartopy(orthographic)
        lat, lon = convert_to_map_lat_lon(f.Center_Longitude, f.Center_Latitude, geodetic, orthographic)
        ax.plot(lon, lat, color="k", linewidth=1, ls="--", transform=orthographic)
        lat_n, lon_n = convert_to_map_lat_lon(f.South_Longitude, f.South_Latitude, geodetic, orthographic)
        ax.plot(lon_n, lat_n, color="b", linewidth=1, ls="-", transform=orthographic)
        lat_s, lon_s = convert_to_map_lat_lon(f.North_Longitude, f.North_Latitude, geodetic, orthographic)
        ax.plot(lon_s, lat_s, color="b", linewidth=1, ls="-", transform=orthographic)
        
        overlay_fov(rads, ax, orthographic, geodetic, beamLimits=[bm, bm-1], fovColor=cl)
        overlay_fov(rads, ax, orthographic, geodetic,)
        overlay_radar(rads, ax, orthographic, geodetic, north=north)
        ax.text(0.01, 1.05, title, ha="left", va="top", transform=ax.transAxes)
        ax.set_extent((-180, 180, extend_lims[0],extend_lims[1]), crs = ccrs.PlateCarree())
        for _k in range(int(len(lat_s)/lsep)):
            _k = int(lsep*_k)
            la_n, lo_n, la_s, lo_s = lat_n[_k], lon_n[_k], lat_s[_k], lon_s[_k]
            ax.plot([lo_n, lo_s], [la_n, la_s], color="r", linewidth=1, ls="--", transform=orthographic)
            x, y = orthographic.transform_point(lo_n, la_n, geodetic)
            ax.text(lo_n, la_n, times[_k].strftime("%H:%M"), ha="left", va="center",
                    fontdict={"size":7, "color":"r", "weight":"bold"}, transform=orthographic)
        ax.plot([lon_n[-1], lon_s[-1]], [lat_n[-1], lat_s[-1]], color="r", linewidth=1, ls="-.", transform=orthographic)
        date = f.Time.tolist()[0]
        ax.add_feature(Nightshade(date, alpha=0.2))
        ax.text(1.01, 0.99, "Scan Time: %02d s"%(i*4), ha="left", va="top", transform=ax.transAxes, rotation=90)
        ax.text(lon_n[-1], lat_n[-1], times[-1].strftime("%H:%M"), ha="left", va="center", 
                fontdict={"size":7, "color":"r", "weight":"bold"}, transform=orthographic)
        ax.text(-0.01, 0.5, "Geographic Coordinate", ha="right", va="center", transform=ax.transAxes, rotation=90)
        ax.text(0.99, 1.05, r"$B_{%02d}, f_{%d}$"%(bm-1, i/5), ha="right", va="top", transform=ax.transAxes)
        fig.savefig(pngfname%i)
        plt.close()
    code = "ffmpeg -r 1 -i %s -c:v libx264 -vf 'scale=1420:-2,fps=3,format=yuv420p' %s"%(pngfname, "images/Dec4Eclipse_South_mode1.mp4")
    os.system(code)
    os.system("rm -rf images/*mode1_*.png")
    return

def plot_run_mode2():
    os.system("rm -rf images/*mode2_*.png")
    fname, center, rads, title, north, pngfname = "data/Dec4Eclipse.csv", (-95, -90), ["fir"],\
                            "4 December, 2021", False, "images/Dec4Eclipse_South_mode2_%02d.png"
    extend_lims, lsep = [-90, -50], 10.
    f = pd.read_csv(fname, parse_dates=["Time"])
    geodetic = ccrs.Geodetic()
    orthographic = ccrs.Orthographic(center[0], center[1])
    
    times = f.Time.tolist()
    beams = [16, 13, 9, 5, 1] * 3
    colors = ["cyan", "red", "blue"]
    i = 0
    for bcl, cl in enumerate(colors):
        for bm in beams:
            fig, ax = create_cartopy(orthographic)
            lat, lon = convert_to_map_lat_lon(f.Center_Longitude, f.Center_Latitude, geodetic, orthographic)
            ax.plot(lon, lat, color="k", linewidth=1, ls="--", transform=orthographic)
            lat_n, lon_n = convert_to_map_lat_lon(f.South_Longitude, f.South_Latitude, geodetic, orthographic)
            ax.plot(lon_n, lat_n, color="b", linewidth=1, ls="-", transform=orthographic)
            lat_s, lon_s = convert_to_map_lat_lon(f.North_Longitude, f.North_Latitude, geodetic, orthographic)
            ax.plot(lon_s, lat_s, color="b", linewidth=1, ls="-", transform=orthographic)

            overlay_fov(rads, ax, orthographic, geodetic, beamLimits=[bm, bm-1], fovColor=cl)
            overlay_fov(rads, ax, orthographic, geodetic,)
            overlay_radar(rads, ax, orthographic, geodetic, north=north)
            ax.text(0.01, 1.05, title, ha="left", va="top", transform=ax.transAxes)
            ax.set_extent((-180, 180, extend_lims[0],extend_lims[1]), crs = ccrs.PlateCarree())
            for _k in range(int(len(lat_s)/lsep)):
                _k = int(lsep*_k)
                la_n, lo_n, la_s, lo_s = lat_n[_k], lon_n[_k], lat_s[_k], lon_s[_k]
                ax.plot([lo_n, lo_s], [la_n, la_s], color="r", linewidth=1, ls="--", transform=orthographic)
                x, y = orthographic.transform_point(lo_n, la_n, geodetic)
                ax.text(lo_n, la_n, times[_k].strftime("%H:%M"), ha="left", va="center",
                        fontdict={"size":7, "color":"r", "weight":"bold"}, transform=orthographic)
            ax.plot([lon_n[-1], lon_s[-1]], [lat_n[-1], lat_s[-1]], color="r", linewidth=1, ls="-.", transform=orthographic)
            date = f.Time.tolist()[0]
            ax.add_feature(Nightshade(date, alpha=0.2))
            ax.text(1.01, 0.99, "Scan Time: %02d s"%(i*4), ha="left", va="top", transform=ax.transAxes, rotation=90)
            ax.text(lon_n[-1], lat_n[-1], times[-1].strftime("%H:%M"), ha="left", va="center", 
                    fontdict={"size":7, "color":"r", "weight":"bold"}, transform=orthographic)
            ax.text(-0.01, 0.5, "Geographic Coordinate", ha="right", va="center", transform=ax.transAxes, rotation=90)
            ax.text(0.99, 1.05, r"$B_{%02d}, f_{%d}$"%(bm-1, bcl), ha="right", va="top", transform=ax.transAxes)
            fig.savefig(pngfname%i)
            plt.close()
            i += 1
    code = "ffmpeg -r 1 -i %s -c:v libx264 -vf 'scale=1420:-2,fps=3,format=yuv420p' %s"%(pngfname, "images/Dec4Eclipse_South_mode2.mp4")
    os.system(code)
    os.system("rm -rf images/*mode2_*.png")
    return

if __name__ == "__main__":
    #plot_run_mode1()
    plot_run_mode2()
    #create_dec4_eclipse()
    #create_june10_eclipse()
    #plot_fov()
    pass