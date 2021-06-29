"""sounding_mode.py: module is dedicated to generate sounding."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

class FanPlot(object):
    """ Plot Fan Dataset """

    def __init__(self, nrange=75, nbeam=15, r0=180, dr=45, dtheta=3.24, theta0=None):
        """
        Initialize the fanplot do a certain size.
        :param nrange: number of range gates
        :param nbeam: number of beams
        :param r0: initial beam distance - any distance unit as long as it"s consistent with dr
        :param dr: length of each radar - any distance unit as long as it"s consistent with r0
        :param dtheta: degrees per beam gate, degrees (default 3.24 degrees)
        """
        # Set member variables
        self.nrange = int(nrange)
        self.nbeam = int(nbeam+1)
        self.r0 = r0
        self.dr = dr
        self.dtheta = dtheta
        # Initial angle (from X, polar coordinates) for beam 0
        if theta0 == None:
            self.theta0 = (90 - dtheta * nbeam / 2)     # By default, point fanplot towards 90 deg
        else:
            self.theta0 = theta0
        return
    
    def add_axis(self, fig, subplot):
        ax = fig.add_subplot(subplot, polar=True)
        # Set up ticks and labels
        self.r_ticks = range(self.r0, self.r0 + (self.nrange+1) * self.dr, self.dr)
        self.theta_ticks = [self.theta0 + self.dtheta * b for b in range(self.nbeam+1)]
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        plt.rgrids(self.r_ticks, [""] * len(self.r_ticks))
        ax.xaxis.grid(True)
        plt.thetagrids(self.theta_ticks, [""]*len(self.theta_ticks))
        return ax
    
    def _scale_plot(self, ax):
        # Scale min-max
        ax.set_thetamin(self.theta_ticks[0])
        ax.set_thetamax(self.theta_ticks[-1])
        ax.set_rmin(0)
        ax.set_rmax(self.r_ticks[-10])
        return
    
    def shade(self, ax, cond, col="r"):
        ax.plot([np.mean(cond),np.mean(cond)], [0, self.r_ticks[-10]])
        #tkx = np.array(self.theta_ticks)
        #ax.fill_betweenx(y=self.r_ticks, x1=np.ones(len(self.r_ticks))*cond[0], x2=np.ones(len(self.r_ticks))*cond[1], 
        #                  color=col, alpha=0.4, transform=ax.transData)
        #print(cond)
        #ax.plot([self.r_ticks[0],self.r_ticks[-1]], cond, linewidth=10)
        #plt.fill(np.concatenate(([cond[0]], [cond[1]])), np.concatenate(([self.r_ticks[0]], [self.r_ticks[-10]])))
        return
    
    def plot_fov(self):
        fig = plt.figure(figsize=(8,4), dpi=120)
        ax = self.add_axis(fig, 111)
        self._scale_plot(ax)
        # Create interleav mode
        for ix in range(len(self.theta_ticks)-1):
            self.shade(ax, [self.theta_ticks[ix],self.theta_ticks[ix+1]])
            #break
        fig.savefig("out.png")
        return

if __name__ == "__main__":
    fan = FanPlot()
    fan.plot_fov()