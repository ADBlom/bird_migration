import pandas as pd
import geopandas as gpd
from glob import glob
from tqdm import tqdm

# set path to output files of the model
path = '/Volumes/Untitled/ERA5_Data/Dir_pi/'
years = ['2020', '2019', '2018', '2017', '2016']

# loop for processing per year
for year in tqdm(years):
    files = glob(path+year+'/*.p')
    # loop for processing each file
    for file in tqdm(files):
        df = pd.read_pickle(file)
        # select final destinations of each bird
        idx = df.groupby(['bird_number'])['t'].transform(max) == df['t']
        df_end = df[idx]

        # align final destinations to place on world map
        world = gpd.read_file(
            gpd.datasets.get_path('naturalearth_lowres')
        )

        world.drop(['pop_est', 'iso_a3', 'gdp_md_est'],
               axis=1,
               inplace = True
        )

        gdf = gpd.GeoDataFrame(
            df_end,
            crs="EPSG:4326",
            geometry=gpd.points_from_xy(
                df_end.lon,
                df_end.lat
        ))

        joined = gpd.tools.sjoin(
            gdf,
            world,
            op="within",
            how='left'
        )
        # select some meaningful information
        joined['year'] = year
        joined['file'] = file
        joined.set_index('bird_number', inplace=True)
        joined['ave_distance'] = df.groupby('bird_number').mean()['distance']
        joined['ave_bearing'] = df.groupby('bird_number').mean()['bearing']
        joined['std_bearing'] = df.groupby('bird_number').std()['bearing']
        joined['max_distance'] = df.groupby('bird_number').max()['distance']
        joined['ave_p'] = df.groupby('bird_number').mean()['p']
        joined['bird_number'] = joined.index.values

        # extract only the birds that did not arrived in Europe
        if 'means' in globals():
            means = pd.concat([means, joined[joined['continent'] != 'Europe']], ignore_index=True)
        else:
            means = joined[joined['continent'] != 'Europe']

# save to pickle
means.to_pickle(path+'birds_total.p')