""""
Fetching the full data sets for all available stations
"""

import pandas as pd
import DataImporter

# All station information
df = pd.read_csv("buoys-details.csv").sort_values("N_obs", ascending=False, ignore_index=True)

# Fetching data per station
for l in range(len(df)):
    try:
        start_time = str(df.iloc[l]["first_obs"]) +"T00:00"
        end_time   = str(df.iloc[l]["last_obs"])  +"T23:59"
        dataImporter = DataImporter.DataImporter(start_time=start_time, end_time=end_time)
        dataImporter.constructDataset(station_id=str(df.iloc[l]["buoyid"]))
    except:
        pass

