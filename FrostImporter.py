#!/usr/bin/env python3

"""Fetch observations from Havvarsel Frost (havvarsel-frost.met.no) and Frost (frost.met.no) and construct csv dataset 

Test havvarsel-frost.met.no (badevann): 
'python3 frost-plots.py -id 5 -param temperature -S 2019-01-01T00:00 -E 2019-12-31T23:59'

Test frost.met.no (observations) - THIS TAKES A COUPLE OF MINUTES TO RUN: 
'python3 frost-plots.py --fab https://frost.met.no -id SN18700 -param air_temperature -S 2019-01-01T00:00 -E 2019-12-31T23:59'
(other available params we have discussed to include: wind_speed and relative_humidity and cloud_area_fraction and sum(duration_of_sunshinePT1H) or mean(surface_downwelling_shortwave_flux_in_air PT1H) )

Test for the construction of a data set:
'python FrostImporter.py --fab all -id 1 -param air_temperature -n 10 -param "sum(duration_of_sunshine PT1H) -n 5 -S 2019-01-01T00:00 -E 2019-12-31T23:59'

TODO:
 - Tune processing and storing of observational data sets (to suite whatever code that will use the data sets)
 - Prototype simple linear regression
 - Prototype simple ANN (with Tensorflow and Keras?)
 - ...
"""

import argparse
import sys
import json
import datetime
import requests
import io
from traceback import format_exc
import pandas as pd
import numpy as np
from haversine import haversine 

class FrostImporter:
    def __init__(self, frost_api_base=None, station_id=None, start_time=None, end_time=None):
        """ Initialisation of FrostImporter Class
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


    def havvarsel_frost(self, station_id, param="temperature", frost_api_base="http://havvarsel-frost.met.no", \
        start_time=None, end_time=None):
        """Fetch data from Havvarsel Frost server.
        
        References:
        API documentation for obs/badevann http://havvarsel-frost.met.no/docs/apiref#/obs%2Fbadevann/obsBadevannGet 
        Datastructure described on http://havvarsel-frost.met.no/docs/dataset_badevann
        """

        # using member variables if applicable
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time

        # Fetching the data from the server
        endpoint = frost_api_base + "/api/v1/obs/badevann/get"

        payload = {'time': start_time.isoformat() + "Z/" + end_time.isoformat() + "Z", 
                    'incobs':'true', 'buoyids': station_id, 'parameters': param}

        try:
            r = requests.get(endpoint, params=payload)
            self.__log("Trying " + r.url)
            r.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise Exception(err)

        # extract meta information from the Frost response
        # NOTE: Assumes that the response contains only one timeseries
        header = r.json()["data"]["tseries"][0]["header"]
        # Cast to data frame
        header_list = [header["id"]["buoyID"],header["id"]["parameter"]]
        header_list.extend([header["extra"]["name"], header["extra"]["pos"]["lon"], header["extra"]["pos"]["lat"]])
        df_location = pd.DataFrame([header_list], columns=["buoyID","parameter","name","lon","lat"])
        self.__log(df_location.to_string())

        # extract the actual observations from the Frost response
        # NOTE: Assumes that the response contains only one timeseries
        observations = r.json()["data"]["tseries"][0]["observations"]
        
        # massage data for pandas
        rows = []
        for data in observations:
            row = {}
            row['time'] = data['time']
            row[param] = data['body']['value']
            rows.append(row)
        
        # make DataFrame (and convert from strings to datetime and numeric value)
        df = pd.DataFrame(rows)
        df['time'] =  pd.to_datetime(df['time'])
        df[param] = pd.to_numeric(df[param])
        df.columns = ['time', station_id]
        df.set_index('time')
        df.rename(columns={station_id:"water_temp"}, inplace=True)

        # NOTE: some observations are 1min delayed. 
        # To ensure agreement with hourly observations from Frost
        # We floor the times to hours
        df["time"] = df["time"].dt.floor('H')

        return(df_location, df)


    def frost_location_ids(self, havvarsel_location, n, param, client_id='3cf0c17c-9209-4504-910c-176366ad78ba'):
        """Identifying the n closest station_ids in the Frost database around havvarsel_locations"""

        # Fetching source data from frost for the given param 
        url = "https://frost.met.no/sources/v0.jsonld"

        payload = {"validtime":str(self.start_time.date())+"/"+str(self.end_time.date()),
                        "elements":param}

        try:
            r = requests.get(url, params=payload, auth=(client_id,''))
            self.__log("Trying " + r.url)
            r.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise Exception(err)

        data = r.json()['data']

        # storing location information data frame
        df = pd.DataFrame()
        for element in data:
            if "geometry" in element:
                row = pd.DataFrame(element["geometry"])
                row["station_id"] = element["id"]
                df = df.append(row)

        df = df.reset_index()

        # Fetching double check from observations/availableTimeseries
        url_availability = "https://frost.met.no/observations/availableTimeSeries/v0.jsonld"

        payload_availability = {'elements': param,
                    'referencetime': self.start_time.isoformat() + "/" + self.end_time.isoformat() + ""}

        try:
            r_availability = requests.get(url_availability, params=payload_availability, auth=(client_id,''))
            self.__log("Trying " + r_availability.url)
            r.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise Exception(err)

        data_availability = r_availability.json()['data']

        id_list = []
        for element in data_availability:
            dict_tmp = {}
            # NOTE: The sourceIds in observations/availableTimeseries have format AA00000:0 
            # and only the first part is comparable to the sourceIds from Sources/
            dict_tmp.update({"id":element["sourceId"].split(":")[0]})
            id_list.append(dict_tmp)

        df_availability = pd.DataFrame(id_list)
        
        # Extracting only those stations where really time series are available
        df = df.loc[df['station_id'].isin(df_availability["id"])]

        # Building data frame with coordinates and distances with respect to havvarsel_location
        latlon_ref = (float(havvarsel_location["lat"][0]),float(havvarsel_location["lon"][0]))

        df_dist = pd.DataFrame()
        for i in range(int(len(df)/2)):
            id  = df.iloc[2*i]["station_id"]
            latlon = (df.iloc[2*i+1]["coordinates"],df.iloc[2*i]["coordinates"])
            dist  = haversine(latlon_ref,latlon)
            df_dist = df_dist.append({"station_id":id, "lon":latlon[0], "lat":latlon[1], "dist":dist}, ignore_index=True)

        # Identify closest n stations 
        df_ids = df_dist.nsmallest(n,"dist")
        df_ids = df_ids.reset_index(drop=True)

        self.__log(df_ids.to_string())
        
        return(df_ids["station_id"])


    def frost(self, station_id, param, frost_api_base="https://frost.met.no", start_time=None, end_time=None,\
        client_id='3cf0c17c-9209-4504-910c-176366ad78ba'):
        """Fetch data from standard Frost server.

        References:
        API documentation for observations on https://frost.met.no/api.html#!/observations/observations 
        Available elements (params) are listed on https://frost.met.no/elementtable 
        Examples on Frost data manipulation with Python on https://frost.met.no/python_example.html

        See also:
        Complete documentation at https://frost.met.no/howto.html 
        Complete API reference at https://frost.met.no/api.html 
        """

        # using member variables if applicable
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time

        timeseries = pd.DataFrame()

        # NOTE: There is a limit of 100.000 observation which can be fetched at once 
        # Hence, time series over several years are may too long
        # As work-around: We fetch the time series year by year 
        # TODO: Only batch the time series if necessary
        years = end_time.year - start_time.year

        for batch in range(years+1):
            if batch == 0:
                inter_start = start_time
            else: 
                inter_start = datetime.datetime.strptime(str(start_time.year+batch)+"-01-01T00:00", "%Y-%m-%dT%H:%M")
            
            if batch == years:
                inter_end = end_time
            else:
                inter_end = datetime.datetime.strptime(str(start_time.year+batch)+"-12-31T23:59", "%Y-%m-%dT%H:%M")

            
            # Fetching data from server
            endpoint = frost_api_base + "/observations/v0.csv"

            payload = {'referencetime': inter_start.isoformat() + "Z/" + inter_end.isoformat() + "Z", 
                        'sources': station_id, 'elements': param}

            try:
                r = requests.get(endpoint, params=payload, auth=(client_id,''))
                self.__log("Trying " + r.url)
                r.raise_for_status()
                
                # Storing in dataframe
                df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
                df['referenceTime'] =  pd.to_datetime(df['referenceTime'])
                df = df.reset_index()

            except requests.exceptions.HTTPError as err:
                self.__log(str(err))
                return(None)

            timeseries = timeseries.append(df, ignore_index=True)
        
        return(timeseries)
            
        

    def postprocess_frost(self, timeseries, station_id, param, data):
        """Tweaking the frost output timeseries such that it matches the times in df
        and attaching it to data as new column"""

        # NOTE: The Frost data commonly holds observations for more times 
        # than the referenced Havvarsel Frost timeseries.
        # Extracting observations only for relevant times 
        if "time" not in data.columns:
            data = data.reset_index()
        ts = timeseries.loc[timeseries['referenceTime'].isin(data["time"])]

        # NOTE: The Frost time series may misses observations at times which are present in the Havvarsel timeseries
        if len(data)>len(ts):
            self.__log("The time series misses observation(s)...")
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
            
        # NOTE: The Frost data can contain data for different "levels" for a parameter
        cols_param = [s for s in ts.columns if param.lower() in s]

        # Left join to add new observations 
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


    def constructDataset(self, station_id, params, ns):
        """ construct a csv file containing the water_temperature series of the selected station of havvarsel frost
        and adds params time series from the n closest frost stations
        """
        self.__log("-------------------------------------------")
        self.__log("Starting the construction of an data set...")
        # meta data and time series from havvarsel frost
        self.__log("The Havvarsel Frost observation site:")
        location, data = self.havvarsel_frost(station_id)
        for ip in range(len(params)):
            param = params[ip]
            n = int(ns[ip])
            self.__log("-------------------------------------------")
            self.__log("Parameter: "+param+".")
            # identifying closest station_id's on frost
            self.__log("The closest "+str(n)+" Frost stations:")
            frost_station_ids = self.frost_location_ids(location, n, param)
            # Fetching data for those ids and add them to data
            for i in range(len(frost_station_ids)):
                # NOTE: Per call a maximum of 100.000 observations can be fetched at once
                # Some time series exceed this limit.
                # TODO: Fetch data year by year to stay within the limit 
                self.__log("Fetching data for "+ str(frost_station_ids[i]))
                timeseries = self.frost(frost_station_ids[i],param)
                if timeseries is not None:
                    self.__log("Postprocessing the fetched data...")
                    data = self.postprocess_frost(timeseries,frost_station_ids[i],param,data)
        # save dataset
        self.__log("Dataset is constructed and will be saved now...")
        data.to_csv("dataset.csv")
        self.__log("Ready!")



    @staticmethod
    def __parse_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '--fab', required=False, dest='frost_api_base', default='http://havvarsel-frost.met.no',
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
        FrostImporter()
    except SystemExit as e:
        if e.code != 0:
            print('SystemExit(code={}): {}'.format(e.code, format_exc()), file=sys.stderr)
            sys.exit(e.code)
    except: # pylint: disable=bare-except
        print('error: {}'.format(format_exc()), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)

