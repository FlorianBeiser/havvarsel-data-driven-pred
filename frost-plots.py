#!/usr/bin/env python3

"""Fetch observations from Havvarsel Frost (havvarsel-frost.met.no) and Frost (frost.met.no) 

and do something (for now: print and plot) with them

Test havvarsel-frost.met.no (badevann): 
'python3 frost-plots.py -id 5 -param temperature -S 2019-01-01T00:00 -E 2019-12-31T23:59'

Test frost.met.no (observations) - THIS TAKES A COUPLE OF MINUTES TO RUN: 
'python3 frost-plots.py --fab https://frost.met.no -id SN18700 -param air_temperature -S 2019-01-01T00:00 -E 2019-12-31T23:59'
(other available params we have discussed to include: wind_speed and relative_humidity)

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
import matplotlib.pyplot as plt

class FrostImporter:
    def __init__(self):

        frost_api_base, station_id, param, start_time, end_time = self.__parse_args()

        start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M")
        end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M")

        # switch between Frost instances/servers
        if "havvarsel" in frost_api_base:
            self.__havvarsel_frost(frost_api_base, station_id, param, start_time, end_time)
        else:
            self.__frost(frost_api_base, station_id, param, start_time, end_time)

    def __havvarsel_frost(self, frost_api_base, station_id, param, start_time, end_time):
        """Fetch data from Havvarsel Frost server.
        
        References:
        API documentation for obs/badevann http://havvarsel-frost.met.no/docs/apiref#/obs%2Fbadevann/obsBadevannGet 
        Datastructure described on http://havvarsel-frost.met.no/docs/dataset_badevann
        """

        endpoint = frost_api_base + "/api/v1/obs/badevann/get"

        payload = {'time': start_time.isoformat() + "Z/" + end_time.isoformat() + "Z", 
                    'incobs':'true', 'buoyids': station_id, 'parameters': param}

        try:
            r = requests.get(endpoint, params=payload)
            print("Trying " + r.url)
            r.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise Exception(err)

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

        # save data frame
        df.to_csv('timeseries_badevann_' + param + '.csv', index=False)

        # print and plot
        print(df)      
        df[station_id].plot(title='Temperature (deg C)') # need some tweaking to use time-column as x-axis values
        plt.savefig('timeseries_badevann_' + param + '.png')
        plt.show()

    def __frost(self, frost_api_base, station_id, param, start_time, end_time):
        """Fetch data from standard Frost server.

        References:
        API documentation for observations on https://frost.met.no/api.html#!/observations/observations 
        Available elements (params) are listed on https://frost.met.no/elementtable 
        Examples on Frost data manipulation with Python on https://frost.met.no/python_example.html

        See also:
        Complete documentation at https://frost.met.no/howto.html 
        Complete API reference at https://frost.met.no/api.html 
        """

        client_id = 'd9b49879-6a30-46bb-8030-de9f74aef5b1' # for martinls @ met.no

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

        df = df.reset_index()

        print(df.head())

        # short table
        # These additional columns will be kept
        columns = ['sourceId','referenceTime','elementId','value','unit','timeOffset']
        df2 = df[columns].copy()
        # Convert the time value to something Python understands
        df2['referenceTime'] = pd.to_datetime(df2['referenceTime'])
        print(df2.head())

        # even shorter table
        columns = ['referenceTime','value']
        df3 = df2[columns].copy()
        df3.columns = ['time', station_id]

        # save data frame
        df3.to_csv('timeseries_observations_' + param + ".csv", index=False)

        # NOTE: Need to do some filtering for air_temperature, as we currently get two values per time
        # in two different levels (2 meter and 10 meter)
        print(df['level'][0])
        print(df['level'][1])

        # print and plot
        print(df3)     
        df3[station_id].plot(title='Temperature (deg C)') # need some tweaking to use time-column as x-axis values
        plt.savefig('timeseries_observations_' + param + '.png')
        plt.show()

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

