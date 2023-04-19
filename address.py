import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import json
from geopy.geocoders import Nominatim


df = pd.read_parquet('/home/dell/Documents/mongo_preprocess/addresses.parquet')

lat_long = '25.6993,85.1376'

import pandas as pd
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapiExercises")

def address_convertor(geolocator, X):
    location = geolocator.reverse(X)
    address = location.raw['address']
    return address

def address_finder(geolocator, lat_long):
    address_df = pd.read_parquet('addresses.parquet')
    addresses = address_df[address_df.Lat_Long == lat_long]['address']
    if len(addresses)>0:
        address = json.loads(addresses[0])
    else:
        address = address_convertor(geolocator, lat_long)

    return address



address_df['address'] = address_df['address'].apply(lambda x: json.loads(x.decode()))
address_df['address_keys'] = address_df['address'].apply(lambda x: list(x.keys()))
address_keys_list = list([a for b in address_df.address_keys.tolist() for a in b])
keys_to_extract = ['county','suburb','city','town','village','city_district','neighbourhood']
for i in keys_to_extract:
    address_df['address_'+i+'_specific'] = address_df.address.apply(lambda x: 1 if i in x else 0)

        


m = df[df.Lat_Long == lat_long]['address']


if len(m)>0:
    print(json.loads(m[0]))
else:
    add_returned = address_convertor(geolocator, lat_long)
    print(add_returned)

    df['address'] = df['address'].apply(lambda x: json.loads(x.decode()))
    df['address_keys'] = df['address'].apply(lambda x: list(x.keys()))
    address_keys_list = list([a for b in df.address_keys.tolist() for a in b])
    keys_to_extract = ['county','suburb','city','town','village','city_district','neighbourhood']
    for i in keys_to_extract:
        df['address_'+i+'_specific'] = df.address.apply(lambda x: 1 if i in x else 0)
indices = np.where(data == 25.6093,85.1376)  # Find the indices where the value is 6
row_index, col_index = indices[0][0], indices[1][0]  # Get the first row and column indices
value = data[row_index, col_index]  # Extract the value

# print(value)  # Output: 6
