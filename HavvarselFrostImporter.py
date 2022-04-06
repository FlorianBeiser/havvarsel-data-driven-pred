#!/usr/bin/env python3

"""
Fetch observations from Havvarsel Frost (havvarsel-frost.met.no) 

Test havvarsel-frost.met.no (badevann): 
'python3 HavvarselFrostImporter.py -id 5 -S 2019-01-01T00:00 -E 2019-12-31T23:59'

"""

import argparse
import sys
import json
import datetime
import requests
from traceback import format_exc
import pandas as pd


class HavvarselFrostImporter:

    def __init__(self, start_time=None, end_time=None):
        """ Initialisation of DataImporter Class
        If nothing is specified as argument, command line arguments are expected.
        Otherwise an empty instance of the class is created
        """

        # For command line calls the class reads the parameters from argsPars
        if start_time is None:
            station_id, start_time, end_time = self.__parse_args()

            start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M")
            end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M")

            _, data = self.data(station_id, start_time=start_time, end_time=end_time)
            data.to_csv("data.csv")
        
        # Non-command line calls expect start and end_time to initialise a valid instance
        else:
            self.start_time = start_time
            self.end_time = end_time


    def data(self, station_id, param="temperature", frost_api_base="https://havvarsel-frost.met.no", \
        start_time=None, end_time=None):
        """Fetch data from Havvarsel Frost server.
        
        References:
        API documentation for obs/badevann https://havvarsel-frost.met.no/docs/apiref#/obs%2Fbadevann/obsBadevannGet 
        Datastructure described on https://havvarsel-frost.met.no/docs/dataset_badevann
        """

        # using member variables if applicable
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time

        # Fetching the data from the server
        endpoint = frost_api_base + "/api/v1/obs/badevann/get"

        payload = {'time': start_time.isoformat() + "Z/" + end_time.isoformat() + "Z", 
                    'incobs':'true', 'buoyids': station_id, 'parameter': param}
        payload_str = "&".join("%s=%s" % (k,v) for k,v in payload.items())

        try:
            r = requests.get(endpoint, params=payload_str)
            self.__log("Trying " + r.url)
            r.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise Exception(err)

        # extract meta information from the Frost response
        # NOTE: Assumes that the response contains only one timeseries
        header = r.json()["data"]["tseries"][0]["header"]
        # Cast to data frame
        header_list = [header["id"]["buoyid"],header["id"]["parameter"]]
        header_list.extend([header["extra"]["name"], header["extra"]["pos"]["lon"], header["extra"]["pos"]["lat"]])
        df_location = pd.DataFrame([header_list], columns=["buoyid","parameter","name","lon","lat"])
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
        df = df.set_index("time")

        return(df_location, df)
       
   
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
        HavvarselFrostImporter()
    except SystemExit as e:
        if e.code != 0:
            print('SystemExit(code={}): {}'.format(e.code, format_exc()), file=sys.stderr)
            sys.exit(e.code)
    except: # pylint: disable=bare-except
        print('error: {}'.format(format_exc()), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)

