#!/usr/bin/env python3

"""Extract time series from Norkyst 800 m forecasts on MET THREDDS server (thredds.met.no) 

and do something (for now: print and plot) with them

Hourly resolution (from 2017-02-20T00:00 up to today): https://thredds.met.no/thredds/fou-hi/norkyst800v2.html

Dayily averages (from 2012-06-27T12:00): https://thredds.met.no/thredds/fou-hi/norkyst800m.html (NOT SUPPORTED YET!)

Test: 

Find sea surface elevation (no use of --depth):
'python3 NorKystImporter.py -lon 3 -lat 60 -param zeta -S 2021-04-11T00:00 -E 2021-04-14T23:00'

Find first available timestep before given start time and after given end time for temperature at depth 100 m:
'python3 NorKystImporter.py -lon 3 -lat 60 -depth 100 -param temperature -S 2021-04-11T00:45 -E 2021-04-14T11:15'

TODO:
 - More error handling
 - Tune processing and storing of observational data sets (to suite whatever code that will use the data sets)
 - Nice to have: Make it possible to get multi-level and single-level params in the same fetch
 - (See TODOs in FrostImporter.py)
 - ...

"""

import argparse
import time
import datetime
from traceback import format_exc
import netCDF4
import numpy as np
import pyproj as proj
import sys
import pandas as pd 

import matplotlib.pyplot as plt

class NorKystImporter:
    def __init__(self, start_time=None, end_time=None):

        if start_time is None:
            lon, lat, depth, params, start_time, end_time = self.__parse_args()

            self.start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M")
            self.end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M")
 
            data = {}
            for param in params:
                data[param] = self.norkyst_data(param, lon, lat, self.start_time, self.end_time, depth)
                print(data[param])

            # plots first param
            fig = plt.figure()
            plt.plot(data[params[0]]["referenceTime"],data[params[0]][params[0]+depth])
            plt.show()
            plt.savefig("fig.png")

        else: 
            self.start_time = start_time
            self.end_time = end_time

    @staticmethod
    def daterange(start_date, end_date):
        # +1 to include end_date 
        # and +1 in case the time interval is not divisible with 24 hours (to get the last hours into the last day)
        for i in range(int((end_date - start_date).days + 1)):
            yield (start_date + datetime.timedelta(i)).date()

    def norkyst_filenames(self):
        """Constructing list with filenames of the individual THREDDS netCDF files 
        for the relevant time period"""

        filenames = []

        # add all days in specified time interval (including the day self.end_time)
        for single_date in self.daterange(self.start_time, self.end_time):
            filenames.append(
                single_date.strftime("https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.%Y%m%d00.nc"))

        #NOTE: For some days there do not exist files in the THREDDS catalog.
        # The list of filenames is cleaned such that the first filename is valid
        testing = True
        while testing:
            try: 
                nc = netCDF4.Dataset(filenames[0])
                testing = False
            except:
                filenames.pop(0)

        return filenames


    def norkyst_data(self, param, lon, lat, start_time=None, end_time=None, depth=0):
        """Fetches relevant netCDF files from THREDDS 
        and constructs a timeseries in a data frame"""

        # using member variables if applicable
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time

        # Filenames for fetching
        filenames = self.norkyst_filenames()

        # Load first object
        # and use it to specify the coordinates
        nc = netCDF4.Dataset(filenames[0])
        print("- " + time.strftime("%H:%M:%S", time.gmtime()) + " -")

        # handle projection
        for var in ['polar_stereographic','projection_stere','grid_mapping']:
            if var in nc.variables.keys():
                try:
                    proj1 = nc.variables[var].proj4
                except:
                    proj1 = nc.variables[var].proj4string
        p1 = proj.Proj(str(proj1))
        xp1,yp1 = p1(lon,lat)
        for var in ['latitude','lat']:
            if var in nc.variables.keys():
                lat1 = nc.variables[var][:]
        for var in ['longitude','lon']:
            if var in nc.variables.keys():
                lon1 = nc.variables[var][:]
        xproj1,yproj1 = p1(lon1,lat1)

        # find coordinate of gridpoint to analyze (only wet cells)
        h = np.array(nc["h"])
        land_value = h.min()
        land_mask = np.where((h!=land_value),0,1)
        distances = (xproj1-xp1)**2 + (yproj1-yp1)**2 + land_mask*1e12
        y1, x1 = np.unravel_index(distances.argmin(), distances.shape)

        print('Coordinates model (x,y= '+str(x1)+','+str(y1)+'): '+str(lat1[y1,x1])+', '+str(lon1[y1,x1]))

        # find correct depth index
        all_depths = nc.variables["depth"][:]
        depth_index = np.where(all_depths == int(depth))[0][0]

        # find correct time indices for start and end of timeseries
        times = nc.variables["time"]
        try:
            t1 = netCDF4.date2index(start_time, times, calendar=times.calendar, select="before")
            t1 = max(0,t1)
        except:
            t1 = 0
        
        # FIRST FILE
        timeseries = self.data1file(filenames[0],y1,x1,param,depth,depth_index,t1=t1)

        # LOOP OVER EACH FILE
        for i in range(1,len(filenames)-1):
            try:
                new_timeseries = self.data1file(filenames[i],y1,x1,param,depth,depth_index)
                timeseries = pd.concat([timeseries,new_timeseries], ignore_index=True)
            except:
                pass

        # LAST FILE
        try:
            nc = netCDF4.Dataset(filenames[-1])
            times = nc.variables["time"]
            try:
                t2 = netCDF4.date2index(end_time, times, calendar=times.calendar, select="after")
            except:
                t2 = len(times[:])

            new_timeseries = self.data1file(filenames[-1],y1,x1,param,depth,depth_index,t2=t2)
            timeseries = pd.concat([timeseries,new_timeseries], ignore_index=True)
        except:
            pass

        #NOTE: Since the other data sources explicitly specify the time zone
        # the tz is manually added to the datetime here
        timeseries["referenceTime"] = timeseries["referenceTime"].dt.tz_localize(tz="UTC")         

        return timeseries

    def data1file(self,filename,y1,x1,param,depth,depth_index,t1=0,t2=None):
        nc = netCDF4.Dataset(filename)
        print("- " + time.strftime("%H:%M:%S", time.gmtime()) + " -")
        print("Processing ", filename)
        # EXTRACT REFERENCE TIMES
        # the times fetched from Thredds are in the cftime.GregorianDatetime format,
        # but since pandas does not understand that format we have to cast to datetime by hand
        cftimes = netCDF4.num2date(nc.variables["time"][t1:t2], nc.variables["time"].units)
        datetimes = self.__cftime2datetime(cftimes)

        # FIRST DATA
        data = nc.variables[param][t1:t2,depth_index,y1,x1]
        # Dataframe for return
        timeseries = pd.DataFrame({"referenceTime":datetimes, param+str(depth):data})

        return timeseries


    @staticmethod
    def __cftime2datetime(cftimes):
        datetimes = []
        for t in range(len(cftimes)):
            new_datetime = datetime.datetime(cftimes[t].year, cftimes[t].month, cftimes[t].day, cftimes[t].hour, cftimes[t].minute)
            datetimes.append(new_datetime)
        return datetimes


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
            '-depth', dest='depth', required=False,
            choices=['0', '3', '10', '15', '25', '50', '75', '100', '150', '200', '250', '300', '500', '1000', '2000', '3000'],
            help='fetch data for given depth in meters')
        parser.add_argument(
            '-param', required=True, action='append',
            help='fetch data for parameter')
        parser.add_argument(
            '-S', '--start-time', required=True,
            help='start time in ISO format (YYYY-MM-DDTHH:MM) UTC')
        parser.add_argument(
            '-E', '--end-time', required=True,
            help='end time in ISO format (YYYY-MM-DDTHH:MM) UTC')
        res = parser.parse_args(sys.argv[1:])
        return res.lon, res.lat, res.depth, res.param, res.start_time, res.end_time

if __name__ == "__main__":

    try:
        NorKystImporter()
    except SystemExit as e:
        if e.code != 0:
            print('SystemExit(code={}): {}'.format(e.code, format_exc()), file=sys.stderr)
            sys.exit(e.code)
    except: # pylint: disable=bare-except
        print('error: {}'.format(format_exc()), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)