#!/usr/bin/env python3

"""Extract time series from post-processed forecast on MET THREDDS server (thredds.met.no) 

6h resolution (from 2020-01-01T00:00 up to today): https://thredds.met.no/thredds/fou-hi/norkyst800v2.html

Dayily averages (from 2012-06-27T12:00): https://thredds.met.no/thredds/fou-hi/norkyst800m.html (NOT SUPPORTED YET!)

Test: 

Find sea surface elevation (no use of --depth):
'python3 PPImporter.py -lon 10.7166638 -lat 59.933329 -S 2021-09-18T00:00 -E 2021-09-19T23:59'

IDEA: 
Use forecast weather data instead of observation weather data.
See the MET post-processed data on https://thredds.met.no/thredds/metno.html > products/Archive/Operational/
"""

import argparse
import datetime
from traceback import format_exc
import netCDF4
import numpy as np
import pyproj as proj
import sys
import pandas as pd 

import matplotlib.pyplot as plt

class PPImporter:
    def __init__(self, start_time=None, end_time=None):

        if start_time is None:
            lon, lat, params, start_time, end_time = self.__parse_args()

            self.start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M")
            self.end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M")

            if params is None:
                params = ['air_temperature_2m', 'wind_speed_10m',\
                    'cloud_area_fraction', 'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time']
 
            data = self.pp_data(params, lon, lat, self.start_time, self.end_time)

            # plots first param
            print(data)
            for param in params:
                fig = plt.figure()
                plt.plot(data.reset_index()["referenceTime"],data[param])
                plt.show()
                plt.savefig("fig_"+param+".png")

        else: 
            self.start_time = start_time
            self.end_time = end_time

    @staticmethod
    def daterange(start_time, end_time):
        # +1 to include end_date 
        # and +1 in case the time interval is not divisible with 24 hours (to get the last hours into the last day)
        dates = []
        for d in range(int(((end_time - start_time).days + 2))):
            for h in range(24):
                dates.append(start_time + datetime.timedelta(d) + datetime.timedelta(hours=h))

        return dates

    def pp_filenames(self):
        """Constructing list with filenames of the individual THREDDS netCDF files 
        for the relevant time period"""

        filenames = []
        print("Filename timestamp based on start_time: " + self.start_time.strftime("%Y%m%d%H"))
        print("Filename timestamp based on end_time: " + self.end_time.strftime("%Y%m%d%H"))

        # add all days in specified time interval (including the day self.end_time)
        for single_date in self.daterange(self.start_time, self.end_time):
            if single_date.year >= 2020:
                filenames.append(
                    single_date.strftime("https://thredds.met.no/thredds/dodsC/metpparchive/%Y/%m/%d/met_analysis_1_0km_nordic_%Y%m%dT%HZ.nc"))
            else:
                filenames.append(
                    single_date.strftime("https://thredds.met.no/thredds/dodsC/metpparchivev2/%Y/%m/%d/met_analysis_1_0km_nordic_%Y%m%dT%HZ.nc"))

        #NOTE: For some days there do not exist files in the THREDDS catalog.
        # The list of filenames is cleaned such that all filenames are valid
        clean_filenames = []
        for f in range(len(filenames)):
            try:
                nc_test = netCDF4.Dataset(filenames[f])
                clean_filenames.append(filenames[f])
            except OSError as err:
                print("OS error: {0}".format(err))

        print("Reading following files: " + str(clean_filenames))
        
        return clean_filenames


    def pp_data(self, params, lon, lat, start_time=None, end_time=None):
        """Fetches relevant netCDF files from THREDDS 
        and constructs a timeseries in a data frame"""

        # using member variables if applicable
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time

        # Filenames for fetching
        clean_filenames = self.pp_filenames()

        # Load multi-file object
        nc = netCDF4.MFDataset(clean_filenames)

        # handle projection
        proj_args = nc.variables["projection_lcc"].proj4
        p = proj.Proj(str(proj_args))

        xp,yp = p(lon,lat)
        lats = nc.variables["latitude"][:]
        lons = nc.variables["longitude"][:]
        xps,yps = p(lons,lats)

        # find coordinate of gridpoint to analyze
        x=self.__find_nearest_index(xps[0,:],xp)
        y=self.__find_nearest_index(yps[:,0],yp)

        print('Coordinates model (x,y= '+str(x)+','+str(y)+'): '+str(lats[y,x])+', '+str(lons[y,x]))

        # find correct time indices for start and end of timeseries
        all_times = nc.variables["time"]
        
        first = netCDF4.num2date(all_times[0],all_times.units)
        last = netCDF4.num2date(all_times[-1],all_times.units)
        print("First available time: " + first.strftime("%Y-%m-%dT%H:%M"))
        print("Last available time: " + last.strftime("%Y-%m-%dT%H:%M"))
  
        try:
            t1 = netCDF4.date2index(start_time, all_times, select="before")
        except:
            t1 = 0
        try:
            t2 = netCDF4.date2index(end_time, all_times, select="after")
        except:
            t2 = len(all_times[:])

        # EXTRACT REFERENCE TIMES
        # the times fetched from Thredds are in the cftime.GregorianDatetime format,
        # but since pandas does not understand that format we have to cast to datetime by hand
        cftimes = netCDF4.num2date(nc.variables["time"][t1:t2], all_times.units)
        datetimes = self.__cftime2datetime(cftimes)

        timeseries = pd.DataFrame()
        for param in params:
            # EXTRACT DATA
            data = nc.variables[param][t1:t2,y,x]

            # Dataframe for return
            new_timeseries = pd.DataFrame({"referenceTime":datetimes, param:data})

            #NOTE: Since the other data sources explicitly specify the time zone
            # the tz is manually added to the datetime here
            new_timeseries["referenceTime"] = new_timeseries["referenceTime"].dt.tz_localize(tz="UTC") 
            
            # Outer joining dataset
            if timeseries.empty:
                timeseries = new_timeseries
            else:
                timeseries = pd.merge(timeseries.set_index("referenceTime"), new_timeseries.set_index("referenceTime")[param], how="outer", on="referenceTime")
                timeseries = timeseries.reset_index()

        timeseries = timeseries.set_index("referenceTime")
        return timeseries


    @staticmethod
    def __cftime2datetime(cftimes):
        datetimes = []
        for t in range(len(cftimes)):
            new_datetime = datetime.datetime(cftimes[t].year, cftimes[t].month, cftimes[t].day, cftimes[t].hour, cftimes[t].minute)
            datetimes.append(new_datetime)
        return datetimes


    @staticmethod
    def __find_nearest_index(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx


    @staticmethod
    def __parse_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '-lon', dest='lon', required=True,
            help='fetch data for grid point nearest to given longitude coordinate')
        parser.add_argument(
            '-lat', dest='lat', required=True,
            help='fetch data for grid point nearest to given latitude coordinate')
        parser.add_argument(
            '-param', default=None, action='append',
            help='fetch data for parameter')
        parser.add_argument(
            '-S', '--start-time', required=True,
            help='start time in ISO format (YYYY-MM-DDTHH:MM) UTC')
        parser.add_argument(
            '-E', '--end-time', required=True,
            help='end time in ISO format (YYYY-MM-DDTHH:MM) UTC')
        res = parser.parse_args(sys.argv[1:])
        return res.lon, res.lat, res.param, res.start_time, res.end_time

if __name__ == "__main__":

    try:
        PPImporter()
    except SystemExit as e:
        if e.code != 0:
            print('SystemExit(code={}): {}'.format(e.code, format_exc()), file=sys.stderr)
            sys.exit(e.code)
    except: # pylint: disable=bare-except
        print('error: {}'.format(format_exc()), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)