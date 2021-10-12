import os
import datetime

import numpy as np
import pvlib.solarposition as SP
import pandas as pd

class Nightshade():
    def __init__(self, dates=None, delta=1., altitudes=[0, 110]):
        for date in dates:
            fname = "data/solar_position_%s.csv"%date.strftime("%Y%m%d-%H%M")
            if not os.path.exists(fname):
                if date is None:
                    date = datetime.datetime.utcnow()

                # make sure date is UTC, or naive with respect to time zones
                if date.utcoffset():
                    raise ValueError(
                        f"datetime instance must be UTC, not {date.tzname()}")
                npts = int(180/delta)+1
                ylat = np.linspace(-90, 90, npts)
                xlon = np.linspace(-180, 180, npts)
                lats, lons = np.meshgrid(ylat, xlon)
                O = pd.DataFrame()
                for i in range(npts):
                    for j in range(npts):
                        print(" Location - ", lats[i,j], lons[i,j], date)
                        for a in altitudes:
                            o = SP.get_solarposition(date, lats[i,j], lons[i,j], altitude=a*1000).reset_index()
                            o = o.rename(columns={"index":"time"})
                            o["alt"], o["lat"], o["lons"] = a, lats[i,j], lons[i,j]
                            O = pd.concat([O,o])
                columns = ["apparent_zenith", "zenith", "apparent_elevation", "elevation",
                           "azimuth", "equation_of_time", "alt", "lat", "lons"]
                for c in columns:
                    O[c] = np.absolute(O[c])
                O.to_csv(fname, index=False, header=True, float_format="%g")
        return
    
if __name__ == "__main__":
    Nightshade([datetime.datetime(2021,12,4,7), datetime.datetime(2021,12,4,7,33), 
                datetime.datetime(2021,12,4,8), datetime.datetime(2021,12,4,8,50)])