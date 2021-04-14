#!/usr/bin/env python3

"""Fetch observations from Havvarsel Frost (havvarsel-frost.met.no) and Frost (frost.met.no) 

and do something (for now: print and plot) with them

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
                        data.to_csv("dataset.csv")
                    else:
                        data = self.frost(station_id, param, frost_api_base, self.start_time, self.end_time)
                        if data is not None:
                            data.to_csv("dataset_"+param+".csv")

        
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
        # Cast to data frame
        header_list = [header["id"]["buoyID"],header["id"]["parameter"]]
        header_list.extend([header["extra"]["name"], header["extra"]["pos"]["lon"], header["extra"]["pos"]["lat"]])
        df_location = pd.DataFrame([header_list], columns=["buoyID","parameter","name","lon","lat"])
        print(df_location)

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

        # Fetching double check from observations/availableTimeseries
        url_availability = "https://frost.met.no/observations/availableTimeSeries/v0.jsonld"

        payload_availability = {'elements': param,
                    'referencetime': self.start_time.isoformat() + "/" + self.end_time.isoformat() + ""}

        try:
            r_availability = requests.get(url_availability, params=payload_availability, auth=(client_id,''))
            print("Trying " + r_availability.url)
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
        df_ids = df_dist.nsmallest(n,"dist")#["station_id"]
        df_ids = df_ids.reset_index(drop=True)

        print(df_ids)
        
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

        # Fetching data from server
        endpoint = frost_api_base + "/observations/v0.csv"

        payload = {'referencetime': start_time.isoformat() + "Z/" + end_time.isoformat() + "Z", 
                    'sources': station_id, 'elements': param}

        try:
            r = requests.get(endpoint, params=payload, auth=(client_id,''))
            print("Trying " + r.url)
            r.raise_for_status()
            
            # Storing in dataframe
            df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
            df['referenceTime'] =  pd.to_datetime(df['referenceTime'])
            df = df.reset_index()

            return(df)

        except requests.exceptions.HTTPError as err:
            print(err)
            return(None)
            
        

    def postprocess_frost(self, timeseries, station_id, param, data):
        """Tweaking the frost output timeseries such that it matches the times in df
        and attaching it to data as new column"""

        # NOTE: The Frost data commonly holds observations for more times 
        # than the referenced Havvarsel Frost timeseries.
        # Extracting observations only for relevant times 
        # TODO: match cloud_area_fraction timeseries by hand
        if "time" not in data.columns:
            data = data.reset_index()
        timeseries = timeseries.loc[timeseries['referenceTime'].isin(data["time"])]
        timeseries = timeseries.set_index("referenceTime")

        # Check if time series have same length
        if len(data)==len(timeseries):
            # NOTE: The Frost data can contain data for different "levels" for a parameter
            cols = timeseries.columns
            cols_param = [s for s in cols if param.lower() in s]

            # Adding observations (requires reset of index)
            timeseries = timeseries.reset_index()
            if "time" in data.columns:
                data = data.set_index("time")
            data = data.reset_index()
            data = data.join(timeseries[cols_param])
            data = data.set_index("time")
            
            # Renaming new columns
            for i in range(len(cols_param)):
                data.rename(columns={cols_param[i]:station_id+param+str(i)}, inplace=True)
        
        else:
            print("Frost time series misses values and is neglected")

        return data


    def constructDataset(self, station_id, params, ns):
        """ construct a csv file containing the water_temperature series of the selected station of havvarsel frost
        and adds params time series from the n closest frost stations
        """
        print("-------------------------------------------")
        print("Starting the construction of an data set...")
        # meta data and time series from havvarsel frost
        print("The Havvarsel Frost observation site:")
        location, data = self.havvarsel_frost(station_id)
        for ip in range(len(params)):
            param = params[ip]
            n = int(ns[ip])
            print("-------------------------------------------")
            print("Parameter: ",param,".")
            # identifying closest station_id's on frost
            print("The closest ",n," Frost stations:")
            frost_station_ids = self.frost_location_ids(location, n, param)
            # Fetching data for those ids and add them to data
            for i in range(len(frost_station_ids)):
                # NOTE: Per call a maximum of 100.000 observations can be fetched at once
                # Some time series exceed this limit.
                # TODO: Fetch data year by year to stay within the limit 
                print("Fetching data for ", frost_station_ids[i])
                timeseries = self.frost(frost_station_ids[i],param)
                if timeseries is not None:
                    print("Postprocessing the fetched data...")
                    data = self.postprocess_frost(timeseries,frost_station_ids[i],param,data)
                    print("Done. (Data is added to the data set)")
        # save dataset
        print("Dataset is constructed and will be saved now...")
        data.to_csv("dataset.csv")
        print("Ready!")



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

