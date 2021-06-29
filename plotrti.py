from get_sd_data import *
import datetime as dt

from plotutils import RangeTimePlot as RTI
from timezonefinder import TimezoneFinder
import utils

rad = "gbr"
dates = [dt.datetime(2021,6,10), dt.datetime(2021,6,11)]
fdata = FetchData( rad, [dates[0], dates[1]] )
beam = 7
s_params = ["noise.sky", "tfreq", "frang", "rsep", "scan", "bmnum", "time"]
v_params = ["slist", "v", "w_l", "p_l"]
beams, _ = fdata.fetch_data(by="beam")

df = fdata.convert_to_pandas(beams, v_params=v_params)


rti = RTI(rad, dates[0], 100, np.unique(df.time), TimezoneFinder(in_memory=True), "", num_subplots=3)
rti.addParamPlot(df, beam, "Date: %s, %s, Beam: %d"%(dates[0].strftime("%Y-%m-%d"), rad.upper(), beam), p_max=50, p_min=-50, 
                 p_step=20, xlabel="", zparam="v", label="Velocity [m/s]", ss_obj=True)
#rti.addParamPlot(df, beam, "", p_max=33, p_min=3, p_step=6, xlabel="", zparam="p_l", label="Power [db]", ss_obj=False)
#rti.addParamPlot(df, beam, "", p_max=50, p_min=0, p_step=10, xlabel="Time [UT]", zparam="w_l", 
#                 label="Spect. Width [m/s]", ss_obj=False)
rti.save("images/June10GBR.png")