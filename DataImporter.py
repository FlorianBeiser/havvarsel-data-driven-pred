#!/usr/bin/env python3

"""
Fetching observations observational and forecast data 
 at/around a specified swimming site in the havvarsel-frost data base
Constructing dataset and saving as csv
 to be used for data driven predictions

The data sources include
- havvarsel-frost (see HavvarselFrostImporter)
- frost (see FrostImporter)
- norkyst800 (see NorKyst)
- post-processed weather forecasts ()

Test for the construction of a data set:
'python DataImporter.py -id 1 -S 2020-09-01T00:00 -E 2020-09-02T23:59'

"""

import argparse
import sys
import datetime
from traceback import format_exc
import pandas as pd

import HavvarselFrostImporter
import FrostImporter
import NorKystImporter
import PPImporter

class DataImporter:
    def __init__(self, station_id=None, start_time=None, end_time=None):
        """ Initialisation of DataImporter Class
        If nothing is specified as argument, command line arguments are expected.
        Otherwise an empty instance of the class is created
        """

        # For command line calls the class reads the parameters from argsPars
        if start_time is None:
            station_id, start_time, end_time = self.__parse_args()

            self.start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M")
            self.end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M")

            # Construct dataset
            self.constructDataset(station_id)

        
        # Non-command line calls expect start and end_time to initialise a valid instance
        else:
            self.start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M")
            self.end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M")


    def constructDataset(self, station_id):
        """ construct a csv file containing the water_temperature series of the selected station of havvarsel frost
        and adds params time series from the n closest frost stations
        """
        self.__log("-------------------------------------------")
        self.__log("Starting the construction of an data set...")
        self.__log("-------------------------------------------")

        #########################################################
        times = pd.date_range(self.start_time, self.end_time, freq="H")
        times = times.tz_localize("UTC")
        data = pd.DataFrame(times, columns=["time"])
        
        #########################################################
        # meta data and time series from havvarsel frost
        havvarselFrostImporter = HavvarselFrostImporter.HavvarselFrostImporter(self.start_time, self.end_time)
        self.__log("The Havvarsel Frost observation site:")
        location, timeseries = havvarselFrostImporter.data(station_id)

        timeseries = timeseries.reset_index()
        data = pd.merge(data.set_index("time"), timeseries.set_index("time"), how="left", on="time")

        self.__log("-------------------------------------------")

        #########################################################
        # time series from frost
        if False:
            frost_params = ["air_temperature", "wind_speed", "cloud_area_fraction",\
                "mean(solar_irradiance PT1H)", "sum(duration_of_sunshine PT1H)", \
                "mean(relative_humidity PT1H)", "mean(surface_downwelling_shortwave_flux_in_air PT1H)"]
            frost_ns = [4, 3, 3, 1, 2, 2, 1]

            frostImporter = FrostImporter.FrostImporter(start_time=self.start_time, end_time=self.end_time)
            for ip in range(len(frost_params)):
                param = frost_params[ip]
                n = int(frost_ns[ip])
                self.__log("-------------------------------------------")
                self.__log("Frost element: "+param+".")
                self.__log("-------------------------------------------")
                # identifying closest station_id's on frost
                self.__log("The closest "+str(n)+" Frost stations:")
                frost_station_ids = frostImporter.location_ids(location, n, param)
                self.__log("-------------------------------------------")
                # Fetching data for those ids and add them to data
                for i in range(len(frost_station_ids)):
                    # NOTE: Per call a maximum of 100.000 observations can be fetched at once
                    # Some time series exceed this limit.
                    # TODO: Fetch data year by year to stay within the limit 
                    self.__log("Fetching data for "+ str(frost_station_ids[i]))
                    timeseries = frostImporter.data(frost_station_ids[i],param)
                    if timeseries is not None:
                        self.__log("Postprocessing the fetched data...")
                        data = self.left_join(timeseries,frost_station_ids[i],param,data)
                self.__log("-------------------------------------------")


        #########################################################
        # time series from THREDDS norkyst
        self.__log("Fetching data from THREDDS")
        norkystImporter = NorKystImporter.NorKystImporter(self.start_time, self.end_time)
        timeseries = norkystImporter.norkyst_data("temperature", 
                        float(location["lon"][0]), float(location["lat"][0]), depth=0)

        timeseries = timeseries.rename(columns={"referenceTime":"time"})
        timeseries = timeseries.rename(columns={"temperature0":"norkyst_water_temp"})
        
        data = data.reset_index()
        data = pd.merge(data.set_index("time"), timeseries.set_index("time"), how="left", on="time")
        
        self.__log("-------------------------------------------")


        #########################################################
        # time series from THREDDS post-processed forecast
        pp_params = ['air_temperature_2m', 'wind_speed_10m',\
            'cloud_area_fraction', 'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time']

        self.__log("Fetching data from THREDDS")
        ppImporter = PPImporter.PPImporter(self.start_time, self.end_time)
        timeseries = ppImporter.pp_data(pp_params, float(location["lon"][0]), float(location["lon"][0]), self.start_time, self.end_time)

        #NOTE: The timezone is manually set for THREDDS observations 
        # (this reduces calculation overhead since otherwise it would be handled as missing data
        # however it would be imputed with the right values)
        self.__log("Postprocessing the fetched data...")

        timeseries = timeseries.reset_index()
        timeseries = timeseries.rename(columns={"referenceTime":"time"})
        
        data = data.reset_index()
        data = pd.merge(data.set_index("time"), timeseries.set_index("time")[pp_params], how="left", on="time")

        self.__log("-------------------------------------------")


        #########################################################
        # save dataset
        self.__log("Dataset is constructed and will be saved now...")
        data.to_csv("dataset.csv")
        self.__log("Ready!")

    
    def left_join(self, timeseries, station_id, param, data):
        """Preparing the new timeseries for a join by imputing missing data, and 
        FROM data LEFT JOIN ts(=prepared timeseries) ON time=time"""

        # NOTE: The Frost data commonly holds observations for more times 
        # than the referenced Havvarsel Frost timeseries.
        # Extracting observations only for times that exist in Havvarsel Frost
        if "time" not in data.columns:
            data = data.reset_index()
        ts = timeseries.loc[timeseries['referenceTime'].isin(data["time"])]

        # NOTE: The Frost time series may misses observations 
        # at times which are present in the Havvarsel timeseries
        if len(data)>len(ts):
            self.__log("The time series misses observation(s)...")
            ts = self.imput_missing_data(data, timeseries, ts)

        # NOTE: The Frost data can contain data for different "levels" for a parameter
        cols_param = [s for s in ts.columns if param.lower() in s]

        # LEFT JOIN to add new observations 
        # Join performed on "time", this makes "time" the index
        ts = ts.rename(columns={"referenceTime":"time"})
        data = data.reset_index()
        data = pd.merge(data.set_index("time"), ts.set_index("time")[cols_param], how="left", on="time")
        data = data.drop(columns=["index"])
        
        # Renaming new columns
        for i in range(len(cols_param)):
            data.rename(columns={cols_param[i]:station_id+param+str(i)}, inplace=True)
        self.__log("Data is added to the data set")

        return data


    def imput_missing_data(self, data, timeseries,ts):
        """Missing observations in ts are imputed 
        with the value of the nearest temporal neighbor in timeseries
        such that for all times in data an original or faked observation exists"""
            
        missing = data[~data["time"].isin(timeseries["referenceTime"])]["time"]
        # Find closest observation for times in missing
        # And construct dataframe to fill with
        fill = pd.DataFrame()
        for t in missing:
            # Ensuring pd.datetime in timeseries dataframe 
            if "referenceTime" in timeseries.columns:
                timeseries["referenceTime"] = pd.to_datetime(timeseries["referenceTime"])
                timeseries = timeseries.set_index("referenceTime")

            row = timeseries.iloc[[timeseries.index.get_loc(t, method="nearest")]]
            row = row.reset_index()
            row["referenceTime"] = t

            fill = fill.append(row)

        fill = fill.set_index("referenceTime")
        fill = fill.reset_index()

        # Attaching "fake observations" to relevant timeseries
        ts = ts.append(fill)
        ts = ts.reset_index()

        self.__log("Missing observations have been filled with the value from the closest neighbor.")

        return ts


    @staticmethod
    def __parse_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '-id', dest='station_id', required=True,
            help='fetch data for station with given id')
        parser.add_argument(
            '-S', '--start-time', required=True,
            help='start time in ISO format (YYYY-MM-DDTHH:MM) UTC')
        parser.add_argument(
            '-E', '--end-time', required=True,
            help='end time in ISO format (YYYY-MM-DDTHH:MM) UTC')
        res = parser.parse_args(sys.argv[1:])
        return res.station_id, res.start_time, res.end_time


    def __log(self, msg):
        print(msg)
        with open("log.txt", 'a') as f:
            f.write(msg + '\n')

if __name__ == "__main__":

    try:
        DataImporter()
    except SystemExit as e:
        if e.code != 0:
            print('SystemExit(code={}): {}'.format(e.code, format_exc()), file=sys.stderr)
            sys.exit(e.code)
    except: # pylint: disable=bare-except
        print('error: {}'.format(format_exc()), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)

