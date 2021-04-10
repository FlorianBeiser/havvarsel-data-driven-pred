#!/usr/bin/env python3

"""Fetch observations from Havvarsel Frost (havvarsel-frost.met.no) and Frost (frost.met.no) 

and do something (for now: print and plot) with them

Test havvarsel-frost.met.no (badevann): 
'python3 frost-plots.py -id 5 -param temperature -S 2019-01-01T00:00 -E 2019-12-31T23:59'

Test frost.met.no (observations) - THIS TAKES A COUPLE OF MINUTES TO RUN: 
'python3 frost-plots.py --fab https://frost.met.no -id SN18700 -param air_temperature -S 2019-01-01T00:00 -E 2019-12-31T23:59'
(other available params we have discussed to include: wind_speed and relative_humidity and cloud_area_fraction and sum(duration_of_sunshinePT1H) or mean(surface_downwelling_shortwave_flux_in_air PT1H) )

Install requirements with 'pip3 install -r requirements.txt'

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
from traceback import format_exc
import pandas as pd
import numpy as np

class FrostImporter:
    def __init__(self, frost_api_base=None, station_id=None, start_time=None, end_time=None):
        """ Initialisation of FrostImporter Class
        If nothing is specified as argument, command line arguments are expected.
        Otherwise an empty instance of the class is created
        """
        # For command line calls the class reads the parameters from argsPars
        if start_time is None:
            frost_api_base, station_id, param, start_time, end_time = self.__parse_args()

            self.start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M")
            self.end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M")

            # switch between Frost instances/servers
            if "havvarsel" in frost_api_base:
                self.havvarsel_frost(station_id, param, frost_api_base, self.start_time, self.end_time)
            else:
                self.frost(station_id, param, frost_api_base, self.start_time, self.end_time)
        
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
            print("Trying " + r.url)
            r.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise Exception(err)

        # extract meta information from the Frost response
        # NOTE: Assumes that the response contains only one timeseries
        header = r.json()["data"]["tseries"][0]["header"]
        print(header)
        # The dataframe to return only hold the lonlat information for the observation site
        df_header = pd.Series(header["extra"]["pos"])

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

        return(df_header, df)


    def frost_location_ids(self, havvarsel_location, n, client_id='d9b49879-6a30-46bb-8030-de9f74aef5b1'):
        """Identifying the n closest station_ids in the Frost database around havvarsel_locations"""

        # Fetching location data from frost
        url = "https://frost.met.no/sources/v0.jsonld"
        payload = {}

        try:
            r = requests.get(url, params=payload, auth=(client_id,''))
            print("Trying " + r.url)
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

        # Building data frame with coordinates and distances with respect to havvarsel_location
        lon_ref = float(havvarsel_location["lon"])
        lat_ref = float(havvarsel_location["lat"])

        df_dist = pd.DataFrame()
        for i in range(int(len(df)/2)):
            id  = df.iloc[2*i]["station_id"]
            lon = df.iloc[2*i]["coordinates"]
            lat = df.iloc[2*i+1]["coordinates"]
            dist_5  = np.sqrt((lon-lon_ref)**2 + (lat-lat_ref)**2) # this is not a metric distance!
            df_dist = df_dist.append({"station_id":id, "lon":lon, "lat":lat, "dist":dist_5}, ignore_index=True)

        # Identify closest n stations 
        df_ids = df_dist.nsmallest(n,"dist")["station_id"]
        df_ids = df_ids.reset_index(drop=True)
        
        return(df_ids)


    def frost(self, station_id, param, frost_api_base="https://frost.met.no", start_time=None, end_time=None,\
        client_id='d9b49879-6a30-46bb-8030-de9f74aef5b1'):
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

        # Fetching data from server
        endpoint = frost_api_base + "/observations/v0.jsonld"

        payload = {'referencetime': start_time.isoformat() + "Z/" + end_time.isoformat() + "Z", 
                    'sources': station_id, 'elements': param}

        try:
            r = requests.get(endpoint, params=payload, auth=(client_id,''))
            print("Trying " + r.url)
            r.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise Exception(err)
        
        data = r.json()['data']
        
        # full table
        df = pd.DataFrame()
        for element in data:
            row = pd.DataFrame(element['observations'])
            row['referenceTime'] = element['referenceTime']
            row['sourceId'] = element['sourceId']
            df = df.append(row)

        df['referenceTime'] =  pd.to_datetime(df['referenceTime'])

        df = df.reset_index()

        return(df)


    def postprocess_frost(self, timeseries, param, data):
        """Tweaking the frost output timeseries such that it matches the times in df"""

        # NOTE: The Frost data commonly holds observations for more times 
        # than the referenced Havvarsel Frost timeseries.
        # Extracting observations only for relevant times 
        if "time" not in data.columns:
            data = data.reset_index()
        timeseries = timeseries.loc[timeseries['referenceTime'].isin(data["time"])]
        timeseries = timeseries.set_index("referenceTime")

        # NOTE: The Frost data can contain data for different "levels" for a parameter.
        # Extracting the different levels and adding them as new column to data
        if "index" in timeseries.columns:
            for i in range( timeseries["index"].max()+1 ):
                timeseriesIdx = timeseries.loc[timeseries["index"]==i][["value"]]
                timeseriesIdx = timeseriesIdx.reset_index()
                if "time" in data.columns:
                    data = data.set_index("time")
                data = data.reset_index()
                data = data.join(timeseriesIdx["value"])
                data = data.set_index("time")
                data.rename(columns={'value':param+str(i)}, inplace=True)

        return data



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
            '-param', required=True,
            help='fetch data for parameter')
        parser.add_argument(
            '-S', '--start-time', required=True,
            help='start time in ISO format (YYYY-MM-DDTHH:MM) UTC')
        parser.add_argument(
            '-E', '--end-time', required=True,
            help='end time in ISO format (YYYY-MM-DDTHH:MM) UTC')
        res = parser.parse_args(sys.argv[1:])
        return res.frost_api_base, res.station_id, res.param, res.start_time, res.end_time

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

