import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import xarray
from tqdm import tqdm

def model7(birds, wind_data, sim_length, map_coord,
           goal=[], comp = 0.5, self_speed = 0,
           orientation = 'start', sd = 0,
           max_airtime = 40, p_deviation = 4, offset=0,
           bearing = []):
    '''
    Model for simulating bird migration
    :param birds: A 4xn numpy array containing the latitude, longitude, altitude and start time of the birds in degrees,
    degrees, hPa and index in wind_data respectively.
    :param wind_data: An xarray DataArray of an ERA5 grib file containing u and v component of wind
    :param sim_length: int that determines for how many timesteps the model runs
    :param map_coord: bounds of the GRIB file in the format np.array(
         [max(latitude), max(longitude),
          min(latitude), min(longitude),
          min(altitude), max(altitude)]
    :param goal: 2xn numpy array containing coordinates (lat, lon) of the goals that the birds aim towards during goal
    orientation
    :param comp: compensation factor for compensating for wind drift
    :param self_speed: speed of the bird in m/s
    :param orientation: either 'goal' or 'start', depending on which strategy you want birds to use
    :param sd: standard deviation of stochastic element in wind compensation
    :param max_airtime: maximum timesteps of flight
    :param p_deviation: how many pressure levels the bird can switch to during flight
    :param offset: time offset for starting at a later time than t=0
    :param bearing: array of size n, filled with radians as a starting direction for birds during start orientation
    :return:
    '''
    # ERA5 data has a resolution of 0.25 degrees

    # make sure the orientation type is correct
    orientation_types = ['start', 'goal']
    if orientation not in orientation_types:
        raise ValueError(f"Invalid orientation. Expected one of: {orientation_types}")

    #initialize the end_loc variable (which keeps track of bird location) for use in the main loop
    end_loc = birds[0:3].copy()

    # initialize included times
    times = wind_data.time.values

    start_time = birds[3]

    # find grid dimension for regular grid interpolators
    z = int((map_coord[5] - map_coord[4])/50)
    x = int((map_coord[1] - map_coord[3])/0.25)
    y = int((map_coord[0] - map_coord[2]) / 0.25)

    grid_dim = (np.linspace(0, z, z+1),
                np.linspace(0, y, y+1),
                np.linspace(0, x, x+1))

    # for recording trajectories
    num_birds = np.shape(birds)[1]
    rec = np.zeros((num_birds*(sim_length+1), 8))
    rec[0:num_birds, 0] = 0
    rec[0:num_birds, 1:4] = end_loc.T
    rec[0:num_birds, 4] = 0
    rec[0:num_birds, 5] = 0
    rec[0:num_birds, 6] = np.linspace(0, num_birds - 1, num_birds)
    rec[0:num_birds, 7] = False


    #for the model where you only look at the initial bearing
    if orientation == 'start':
        #get the initial bearing of the birds (in radians)
        bird_bear = bearing
        for i in tqdm(range(sim_length)):

            # variable for checking whether the bird is flying at timestep i
            in_flight = ((start_time) <= i) & (i < start_time+max_airtime)

            #get the place on the wind grid
            loc_grid = align_birds2(end_loc, map_coord[0], map_coord[3], map_coord[4])

            # get interpolation function for wind speed variables
            u = RegularGridInterpolator(grid_dim,
                                        wind_data['u'].sel(time=xarray.DataArray(times[i+offset])).values,
                                        bounds_error=False, fill_value=None)
            v = RegularGridInterpolator(grid_dim,
                                        wind_data['v'].sel(time=xarray.DataArray(times[i+offset])).values,
                                        bounds_error=False, fill_value=None)

            # calculate optimal height
            agg = []
            for h in grid_dim[0]:
                test_loc = loc_grid
                test_loc[2] = h
                test_loc = test_loc.T[:, [2,0,1]]
                wind_speed = np.sqrt(v(test_loc) ** 2 + u(test_loc) ** 2)
                wind_angle = np.pi / 2 - np.arctan2(v(test_loc), u(test_loc))
                tailwind = wind_speed * np.cos(wind_angle-bird_bear)
                agg.append(tailwind)
            optimal_height = np.argmax(np.array(agg), axis=0)
            adjacency = (((loc_grid[2] + p_deviation) >= optimal_height) & (
                    optimal_height >= (loc_grid[2] - p_deviation))) | (
                                start_time == i
                        )
            loc_grid[2, adjacency] = optimal_height[adjacency]


            # get eastward and northward speeds
            v_u =u(loc_grid.transpose()[:, [2,0,1]]) + self_speed * np.sin(bird_bear)
            v_v = v(loc_grid.transpose()[:, [2,0,1]]) + self_speed * np.cos(bird_bear)

            # convert to distance and bearing

            dist = np.sqrt((3.6 * v_v) ** 2 + (3.6 * v_u) ** 2)
            brng = np.pi/2 - np.arctan2(v_v, v_u)

            # calculate new location
            newloc = new_loc(dist, brng, end_loc)
            new_height = loc_grid[2]*50 + map_coord[4]
            end_loc[0:2, in_flight] = newloc[0:2, in_flight]
            end_loc[2, in_flight] = new_height[in_flight]

            # update bird direction
            bird_bear = brng

            # record trajectory
            rec[num_birds*(i+1):num_birds*(i+2), 0] = i+1
            rec[num_birds*(i+1):num_birds*(i+2), 1:4] = end_loc.T
            rec[num_birds * (i + 1):num_birds * (i + 2), 4] = bird_bear
            rec[num_birds * (i + 1):num_birds * (i + 2), 5] = dist
            rec[num_birds * (i + 1):num_birds * (i + 2), 6] = np.linspace(0, num_birds-1, num_birds)
            rec[num_birds * (i + 1):num_birds * (i + 2), 7] = ((start_time-1) <= i) & (i < start_time+max_airtime)

    # for the goal orientation
    if orientation == 'goal':
        for i in tqdm(range(sim_length)):

            # variable for checking whether the bird is flying at timestep i
            in_flight = (start_time <= i) & (i < start_time + max_airtime)

            #calculate shortest angle to the goal
            bird_bear = bearing_from_loc(end_loc, goal)


            # calculate location on wind grid and wind speeds
            loc_grid = align_birds2(end_loc, map_coord[0], map_coord[3], map_coord[4])

            u = RegularGridInterpolator(grid_dim,
                                        wind_data['u'].sel(time=xarray.DataArray(times[i+offset])).values,
                                        bounds_error=False, fill_value=None)
            v = RegularGridInterpolator(grid_dim,
                                        wind_data['v'].sel(time=xarray.DataArray(times[i+offset])).values,
                                        bounds_error=False, fill_value=None)

            # calculate optimal height
            agg = []
            for h in grid_dim[0]:
                test_loc = loc_grid.copy()
                test_loc[2] = h
                test_loc = test_loc.T[:, [2, 0, 1]]
                wind_speed = np.sqrt(v(test_loc) ** 2 + u(test_loc) ** 2)
                wind_angle = np.pi / 2 - np.arctan2(v(test_loc), u(test_loc))
                tailwind = wind_speed * np.cos(wind_angle - bird_bear)
                agg.append(tailwind)
            optimal_height = np.argmax(np.array(agg), axis=0)
            adjacency = (((loc_grid[2] + p_deviation) >= optimal_height) & (
                           optimal_height >= (loc_grid[2] - p_deviation))) |(
                start_time==i
            )
            loc_grid[2, adjacency] = optimal_height[adjacency]

            v_u = u(loc_grid.transpose()[:, [2,0,1]])
            v_v = v(loc_grid.transpose()[:, [2,0,1]])

            # calculate angle to compensate the wind direction
            wind_speed = np.sqrt(v_v ** 2 + v_u ** 2)
            wind_angle = np.pi / 2 - np.arctan2(v_v, v_u)
            comp_angle = np.arcsin((wind_speed/self_speed)*np.sin(bird_bear - wind_angle))
            # step to account for the a compensation angle if their is no angle to compensate the wind (selfspeed < windspeed)
            comp_angle = np.nan_to_num(comp_angle)

            #calculate bearing and distance
            bird_choice = np.random.normal(1, sd, len(birds[0]))
            brng = (bird_bear + comp*comp_angle*bird_choice)%(2*np.pi)
            dist = 3.6 * np.sqrt((v_v + self_speed*np.cos(brng)) ** 2 + (v_u + self_speed*np.sin(brng)) ** 2)

            # update location
            newloc = new_loc(dist, brng, end_loc)
            new_height = loc_grid[2] * 50 + map_coord[4]
            end_loc[0:2, in_flight] = newloc[0:2, in_flight]
            end_loc[2, in_flight] = new_height[in_flight]

            # record trajectory
            rec[num_birds * (i + 1):num_birds * (i + 2), 0] = i + 1
            rec[num_birds * (i + 1):num_birds * (i + 2), 1:4] = end_loc.T
            rec[num_birds * (i + 1):num_birds * (i + 2), 4] = bird_bear
            rec[num_birds * (i + 1):num_birds * (i + 2), 5] = dist
            rec[num_birds * (i + 1):num_birds * (i + 2), 6] = np.linspace(0, num_birds - 1, num_birds)
            rec[num_birds * (i + 1):num_birds * (i + 2), 7] = ((start_time-1) <= i) & (i < start_time+max_airtime)

    # save results to dataframe
    df = pd.DataFrame(rec, columns = ['t', 'lat', 'lon', 'p', 'bearing', 'distance', 'bird_number', 'in_flight'])
    return df

def align_birds2(bird_loc, north, west, low):
    '''
    This function aligns the bird locations in degrees to the corresponding grid map of wind speeds
    '''
    lat = np.abs(north - bird_loc[0])/0.25
    lon = np.abs(west - bird_loc[1])/0.25
    height = (bird_loc[2] - low)/50
    aligned_loc = np.array([lat, lon, height])
    return aligned_loc

def new_loc(dist, brng, old_loc):
    '''
    This function takes a distance, direction and starting position in degrees and calculates the new location.
    '''
    r = 6371
    phi = old_loc[0] * np.pi/180
    labda = old_loc[1] * np.pi/180
    end_loc_lat = np.arcsin(np.sin(phi) * np.cos(dist/r) +
                        np.cos(phi) * np.sin(dist/r) * np.cos(brng))
    end_loc_lon = labda + np.arctan2(np.sin(brng) * np.sin(dist/r) * np.cos(phi),
                                         np.cos(dist/r)-np.sin(phi)*np.sin(end_loc_lat))
    return np.array([end_loc_lat/(np.pi/180), end_loc_lon/(np.pi/180)])

def bearing_from_loc(loc, goal):
    '''
    This function calculates the shortes angle from one location in degrees to another
    '''
    phi1 = loc[0] * np.pi/180
    labda1 = loc[1] * np.pi/180
    phi2 = goal[0] * np.pi / 180
    labda2 = goal[1] * np.pi / 180
    y = np.sin(labda2-labda1) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(labda2 - labda1)
    return np.arctan2(y, x)