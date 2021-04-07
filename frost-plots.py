#!/usr/bin/env python3

"""Fetch observations from Havvarsel Frost (havvarsel-frost.met.no) and Frost (frost.met.no) 
and do something (for now: print and plot) with them

Datastructure described on http://havvarsel-frost.met.no/docs/dataset_badevann

Test with 'python3 frost-plots.py -id 5 -param temperature -S 2019-01-01T00:00 -E 2019-12-31T23:59'

Install requirements with 'pip3 install -r requirements.txt'

TODO:
 - Fetch some relevant atmospheric parameters from the "regular" Frost server and expand functionality accordingly
 - Prototype simple linear regression
 - Prototype simple ANN (with Tensorflow and Keras?)
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

        url = frost_api_base + "/api/v1/obs/badevann/get"

        payload = {'time': start_time.isoformat() + "Z/" + end_time.isoformat() + "Z", 
                    'incobs':'true', 'buoyids': station_id, 'parameters': param}

        try:
            r = requests.get(url, params=payload)
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
            row['temperature'] = data['body']['value']
            rows.append(row)
        
        # make DataFrame (and convert from strings to datetime and numeric value)
        df = pd.DataFrame(rows)
        df['time'] =  pd.to_datetime(df['time'])
        df['temperature'] = pd.to_numeric(df['temperature'])
        df.columns = ['time', station_id]
        df.set_index('time')

        # print and plot
        print(df)      
        df[station_id].plot(title='Temperature (deg C)') # need some tweaking to use time-column as x-axis values
        plt.show()

        return

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

