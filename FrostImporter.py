#!/usr/bin/env python3

"""
Fetch observations from Frost (frost.met.no) 

Test frost.met.no (observations) - THIS TAKES A COUPLE OF MINUTES TO RUN: 
'python3 FrostImporter.py -id SN18700 -param air_temperature -S 2019-01-01T00:00 -E 2019-12-31T23:59'

Other possible elements as param's:
- wind_speed 
- relative_humidity 
- cloud_area_fraction 
- sum(duration_of_sunshinePT1H) 
- mean(surface_downwelling_shortwave_flux_in_air PT1H)

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
    def __init__(self, station_id=None, start_time=None, end_time=None):
        """ Initialisation of DataImporter Class
        If nothing is specified as argument, command line arguments are expected.
        Otherwise an empty instance of the class is created
        """

        # For command line calls the class reads the parameters from argsPars
        if start_time is None:
            station_id, params, start_time, end_time = self.__parse_args()

            self.start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M")
            self.end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M")

            for ip in range(len(params)):
                param = params[ip]

                data = self.data(station_id, param, self.start_time, self.end_time)
                if data is not None:
                        data.to_csv("data_"+param+".csv")

        
        # Non-command line calls expect start and end_time to initialise a valid instance
        else:
            self.start_time = start_time
            self.end_time = end_time


    def data(self, station_id, param, start_time=None, end_time=None,\
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
            endpoint = "https://frost.met.no" + "/observations/v0.csv"

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


    def location_ids(self, havvarsel_location, n, param, client_id='3cf0c17c-9209-4504-910c-176366ad78ba'):
        """Used in the full DataImporter....
        Identifying the n closest station_ids in the Frost database around havvarsel_locations
        where havvarsel_location is given as a dataframe with latlon coordinates"""


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



    @staticmethod
    def __parse_args():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '-id', dest='station_id', required=True,
            help='fetch data for station with given id')
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
        return res.station_id, res.param, res.start_time, res.end_time

    
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

