from tqdm import tqdm
import numpy as np
from xarray import open_dataset
import pandas as pd
import models as m

# Preparing the input files
data_loc = '/Volumes/Untitled/ERA5_Data/'

files = ['2020.grib', '2019.grib', '2018.grib', '2017.grib', '2016.grib',
         '2015.grib', '2014.grib', '2013.grib', '2012.grib', '2011.grib']

# Generating the input birds
birds_coast = np.transpose(
        np.flip(
            pd.read_csv("startBirdPosition.csv (1).csv",
                          header=0,
                          delimiter=",",
                          encoding="utf8").values
        )
    )

birds_coast = birds_coast[:, birds_coast[0, :] >= 30]
birds_coast = birds_coast[:, ::5]

birds_alt = np.reshape(np.full(len(birds_coast[0]), 1000, float), (1, len(birds_coast[0])))

# Length of chunk to run
days = 12

# Set the hours of departure
start_hours = [0, 1, 2, 3, 17, 18, 19, 20, 21, 22, 23]
start_times = [x + 24*i for x in start_hours for i in range(days-2)]
start_times.sort()

birds_time = np.reshape(
    np.repeat(start_times, len(birds_coast[0])
    ), (1, len(birds_coast[0])*len(start_times)))

# Create input file
input = np.concatenate((np.tile(birds_coast, len(start_times)),
                        np.tile(birds_alt, len(start_times)),
                        birds_time))

# Create goal matrix
birds_goal = np.array([
    np.full(len(birds_coast[0])*len(start_times), 15, float),
    np.full(len(birds_coast[0])*len(start_times), -85, float)
])

bird_bearing = np.full(len(birds_coast[0])*len(start_times), np.pi, float)
sim_length = days*24

# Loop for running all grib files through the models
for file in tqdm(files):
    da = open_dataset(
         data_loc+file,
         engine='cfgrib'
    )

    bounds = np.array(
         [max(da.latitude.values), max(da.longitude.values),
          min(da.latitude.values), min(da.longitude.values),
          min(da.isobaricInhPa.values), max(da.isobaricInhPa.values)]
     )
    # Loop for running the grib file on the model in chunks
    for j in tqdm(range(2, 11, 1)):
        # Run model
        df = m.model7(input, da, sim_length, map_coord=bounds,
                      self_speed = 12, orientation='start',
                      offset=j*10*24, bearing=bird_bearing)
        print(f'Ran model on day {(j-2)*10} to day {(j-1)*10}')
        # Leave only the data entries of birds while they are in flight
        df = df[df.in_flight == 1]
        # Save to pickle
        df.to_pickle(data_loc+file+f'_day_{j*10}-{(j+1)*10}_simulation.p')




