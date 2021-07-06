#!/usr/bin/env python3

"""
Fetching observations observational and forecast data 
 at/around a specified swimming site in the havvarsel-frost data base
Constructing dataset and saving as csv
 to be used for data driven predictions

The data sources include
- havvarsel-frost (see HavvarselFrostImporter)
- frost (see FrostImporter)
- norkyst800 (see THREDDS)

Test for the construction of a data set:
'python DataImporter.py -id 1 -param air_temperature -n 10 -param "air_temperature" -n 2 -S 2019-06-01T00:00 -E 2019-06-302T23:59'

TODO:
 - Tune processing and storing of observational data sets (to suite whatever code that will use the data sets)
 - Prototype simple linear regression
 - Prototype simple ANN (with Tensorflow and Keras?)
 - Prototype simple XGBoost?
 - ...
"""

import argparse
import sys
import datetime
from traceback import format_exc
import pandas as pd

import HavvarselFrostImporter
import FrostImporter
import THREDDSImporter

class DataImporter:
    def __init__(self, frost_api_base=None, station_id=None, start_time=None, end_time=None):
        """ Initialisation of DataImporter Class
        If nothing is specified as argument, command line arguments are expected.
        Otherwise an empty instance of the class is created
        """

        # For command line calls the class reads the parameters from argsPars
        if start_time is None:
            frost_api_base, station_id, params, ns, start_time, end_time = self.__parse_args()

            self.start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M")
            self.end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M")


            # FIXME: This feature (the selection of frost_api_base) may get obsolete 
            # keyword "all" triggers workflow to construct a csv file
            # containing the water_temperature series of the selected station of havvarsel frost
            # and adds params time series from the n closest frost stations 
            if frost_api_base == "all":
                # Check sanity of input
                assert len(params) == len(ns), "Specified number of parameters does not match the radii"
                # Construct dataset
                self.constructDataset(station_id, params, ns)

            else:
                for ip in range(len(params)):
                    param = params[ip]
                    # switch between Frost instances/servers
                    if "havvarsel" in frost_api_base:
                        assert len(params)==1, "Havvarsel-Frost only allows the single parameter \"temperature\""
                        _, data = self.havvarsel_frost(station_id, param, frost_api_base, self.start_time, self.end_time)
                        data.to_csv("data.csv")
                    else:
                        data = self.frost(station_id, param, frost_api_base, self.start_time, self.end_time)
                        if data is not None:
                            data.to_csv("data_"+param+".csv")

        
        # Non-command line calls expect start and end_time to initialise a valid instance
        else:
            self.start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M")
            self.end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M")


    def constructDataset(self, station_id, params, ns):
        """ construct a csv file containing the water_temperature series of the selected station of havvarsel frost
        and adds params time series from the n closest frost stations
        """
        self.__log("-------------------------------------------")
        self.__log("Starting the construction of an data set...")
        self.__log("-------------------------------------------")
        
        #########################################################
        # meta data and time series from havvarsel frost
        havvarselFrostImporter = HavvarselFrostImporter.HavvarselFrostImporter(self.start_time, self.end_time)
        self.__log("The Havvarsel Frost observation site:")
        location, data = havvarselFrostImporter.data(station_id)
        self.__log("-------------------------------------------")

        #########################################################
        # time series from frost
        frostImporter = FrostImporter.FrostImporter(start_time=self.start_time, end_time=self.end_time)
        for ip in range(len(params)):
            param = params[ip]
            n = int(ns[ip])
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
        threddsImporter = THREDDSImporter.THREDDSImporter(self.start_time, self.end_time)
        timeseries = threddsImporter.norkyst_data("temperature", 
                        float(location["lon"][0]), float(location["lat"][0]), depth=0)
        
        #NOTE: The timezone is manually set for THREDDS observations 
        # (this reduces calculation overhead since otherwise it would be handled as missing data
        # however it would be imputed with the right values)
        self.__log("Postprocessing the fetched data...")
        data = self.left_join(timeseries, "THREDDSnorkyst", "temperature", data)
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
            '--fab', required=False, dest='frost_api_base', default='all',
            help='the Frost API base')
        parser.add_argument(
            '-id', dest='station_id', required=True,
            help='fetch data for station with given id')
        parser.add_argument(
            '-param', required=True, action='append',
            help='fetch data for parameter')
        parser.add_argument(
            '-n', action='append',
            help='number of closest frost stations to consider')
        parser.add_argument(
            '-S', '--start-time', required=True,
            help='start time in ISO format (YYYY-MM-DDTHH:MM) UTC')
        parser.add_argument(
            '-E', '--end-time', required=True,
            help='end time in ISO format (YYYY-MM-DDTHH:MM) UTC')
        res = parser.parse_args(sys.argv[1:])
        return res.frost_api_base, res.station_id, res.param, res.n, res.start_time, res.end_time


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

