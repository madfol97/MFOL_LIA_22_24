# %% Import packages
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import xarray
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import date, timedelta
import time

# %% Find the largest continents definition (nans position vary from a frame to another)
start_time = time.time()
main_dir = "/mnt/bjerknes/"
regionsPossibilities = {'EA': (371,290),
                        'EC': (80,440),
                        'GL': (98,232),
                        'HB': (163,297),
                        'WA': (363,172)}
regionsMasks = {'EA': [],'EC': [], 'GL': [],'HB': [],'WA': []}

print('Starting the Search for Continents...')
print('=====================================')

continents = np.zeros((498,562))
# Create masks per region so that only take into account region if available
for region in regionsPossibilities:
    mask = xarray.open_dataset('/aos/home/mfol/Data/CIS/CIS_' + region + '_19900129_pl_a.nc')['CT']
    regionsMasks[region] = np.where(np.isnan(mask), False, True)

# Loop over the specified year range
for yyyy in range(1982, 2021):
    print(f"Year {yyyy} Starts")
    
    for mm in range(1, 13):
        if yyyy < 1990: 
            main_dir = "/mnt/qanik/CIS/CIS_1982_1990/"
        else:
            main_dir = "/mnt/bjerknes/"
        pattern = f"{main_dir}CIS_*_{yyyy}{mm:02d}*.nc"
        file_paths = glob.glob(pattern)

        if not file_paths:
            print(f"No files found for {yyyy}-{mm:02d}. Skipping.")
            continue

        for file_path in file_paths:
            ds = xarray.open_dataset(file_path)
            # Add other variables if you want them
            CT = ds['CT']

            # Find if there is data and not all nan
            for region in regionsPossibilities:
                if not np.isnan(CT[regionsPossibilities[region][0], regionsPossibilities[region][1]]):
                    # Put 1 out of region so is not interpreted as nan
                    regionToUpdate = np.where(regionsMasks[region], CT, 2000)
                    continents = np.where(np.isnan(regionToUpdate), 1, continents)

    print(f'Year {yyyy}: Completed')
    print('=====================================')

mask = xarray.open_dataset('/mnt/bjerknes/CIS_10km_19900129.nc')['CT']
regionToUpdate = np.where(np.isnan(mask), True, False)           
continents2 = np.where(regionToUpdate, 1, continents)
# np.save('./biggestCommonContinent_Since1982', continents2)

# Plot the biggest common continents
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-90)})
scat = ax.pcolormesh(CT.lon[:,:],CT.lat[:,:], continents2, transform=ccrs.PlateCarree())
c = plt.colorbar(scat, ax=ax, orientation='horizontal',fraction=0.046)
ax.add_feature(cfeature.COASTLINE,color='gray',linewidth=0.3)
ax.gridlines(draw_labels=True)
ax.set_extent([-140, -60, 65, 90], crs=ccrs.PlateCarree()) # CAA

# Calculate and display the total execution time
end_time = time.time()
elapsed_time = end_time - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Data Processing Time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")


# %% Plot the region sampling position will use these point for the data availability matrix
ds = xarray.open_dataset('/mnt/bjerknes/CIS_10km_19910806.nc')
CT = ds['CT']
SA = ds['SA']/ds['CA']
regionsPossibilities = {'EA': (371,290),
                        'EC': (80,440),
                        'GL': (98,232),
                        'HB': (163,297),
                        'WA': (363,172)}
listnum = [-9.0, 1.0, 2.0, 10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0, 91.0, 92.0, 98.0]
replace = [0, 0.01, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1, 0]
replaceThickness = [0, 0.01, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1, 0]

# Replace coded figures in the data with aice concentration
for i in range(len(listnum)):
    CT = CT.where(CT != listnum[i], replace[i])

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-90)})
scat = ax.pcolormesh(ds.lon[:,:],ds.lat[:,:],SA[:,:],vmax=10.0, vmin=-10, transform=ccrs.PlateCarree())
c = plt.colorbar(scat, ax=ax, orientation='horizontal',fraction=0.046)
ax.add_feature(cfeature.COASTLINE,color='gray',linewidth=0.3)
ax.gridlines(draw_labels=True)

# # Plot sampling points of regions
for region in regionsPossibilities:
    plt.scatter(ds.lon[regionsPossibilities[region]], ds.lat[regionsPossibilities[region]], c='red', transform=ccrs.PlateCarree())
    plt.text(ds.lon[regionsPossibilities[region]], ds.lat[regionsPossibilities[region]], region, c='red', transform=ccrs.PlateCarree())


# %%  Create the data availability matrix per region to see what data is available.
FIRSTYEAR = 1982
LASTYEAR = 2020
NUMBEROFYEARS = LASTYEAR - FIRSTYEAR + 1
NUMREGIONS = 5
listnum = [-9.0, 1.0, 2.0, 10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0, 91.0, 92.0, 98.0]
replace = [0, 0.01, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1, 0]

print('Starting the Data Processing...')
print('=====================================')

# Initialize an empty list to store data
CT_monthly_data = []
datesPerRegions = {'all': [],'EA': [], 'EC': [], 'GL': [], 'HB': [], 'WA': []}
regionsPossibilities = {'EA': (371,290),
                        'EC': (80,440),
                        'GL': (98,232),
                        'HB': (163,297),
                        'WA': (363,172)}

# Loop over the specified year range
for yyyy in range(FIRSTYEAR, LASTYEAR + 1):
    print(f"Year {yyyy}: Data Processing Starts")
    
    for mm in range(1, 13):
        if yyyy < 1990: 
            main_dir = "/mnt/qanik/CIS/CIS_1982_1990/"
        else:
            main_dir = "/mnt/bjerknes/"
        pattern = f"{main_dir}CIS_*_{yyyy}{mm:02d}*.nc"
        file_paths = glob.glob(pattern)

        if not file_paths:
            print(f"No files found for {yyyy}-{mm:02d}. Skipping.")
            continue

        monthly_datasets = []
        for file_path in file_paths:
            ds = xarray.open_dataset(file_path)
            CT = ds['CT']

            # Replace coded figures in the data with aice concentration
            for i in range(len(listnum)):
                CT = CT.where(CT != listnum[i], replace[i])

            # Set the time coordinate for each dataset
            date_str = file_path.split('_')[-1].split('.')[0]
            datee = pd.to_datetime(date_str, format='%Y%m%d')
            isAll = 0
            # See if at the sampling position, there is data.
            for region in regionsPossibilities:
                if not np.isnan(CT[regionsPossibilities[region][0], regionsPossibilities[region][1]]):
                    datesPerRegions[region].append(datee)
                    isAll += 1
                    if isAll == NUMREGIONS: 
                        datesPerRegions['all'].append(datee)

# Plot matrices of availability
for region in datesPerRegions:
    dateMatrix = np.zeros((NUMBEROFYEARS, 53))
    for datei in datesPerRegions[region]:
        i = datei.year - FIRSTYEAR
        j = (datei.timetuple().tm_yday // 7) % 52 
        dateMatrix[i,j] = datei.day
    dateMatrix[dateMatrix == 0] = np.nan

    fig, ax = plt.subplots()
    im = ax.imshow(dateMatrix)
    cbar = ax.figure.colorbar(im, ax=ax,fraction=0.046, orientation="horizontal", label='Day of month')

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.linspace(0,NUMBEROFYEARS,NUMBEROFYEARS)[::5], labels=np.linspace(FIRSTYEAR,2020,NUMBEROFYEARS)[::5])
    ax.set_xticks(np.linspace(0,52,53)[::4], labels=np.linspace(0,52,53)[::4])
    ax.set_ylabel('Year')
    ax.set_xlabel('Week')
    ax.set_title("Data availability - " + region)
    fig.tight_layout()
    plt.show()

    
# %% Compute average March and Sept SIC
start_time = time.time()
main_dir = "/mnt/bjerknes/"
FIRSTYEAR = 1982
LASTYEAR = 2020
NUMBEROFYEARS = LASTYEAR - FIRSTYEAR + 1
NUMREGIONS = 5
listnum = [-9.0, 1.0, 2.0, 10.0, 20.0, 30.0, 40.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0, 91.0, 92.0, 98.0]
replace = [0, 0.01, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 1, 0]

print('Starting the Data Processing...')
print('=====================================')

# Initialize an empty list to store monthly data
CT_monthly_data = []
datesPerRegions = {'all': [],'EA': [], 'EC': [], 'GL': [], 'HB': [], 'WA': []}
regionsPossibilities = {'EA': (361,320),
                        'EC': (80,440),
                        'GL': (98,232),
                        'HB': (163,297),
                        'WA': (363,172)}

# Loop over the specified year range
for yyyy in range(FIRSTYEAR, LASTYEAR+1):
    print(f"Year {yyyy}: Data Processing Starts")
    if yyyy < 1990: 
        main_dir = "/mnt/qanik/CIS/CIS_1982_1990/"
    else:
        main_dir = "/mnt/bjerknes/"
    
    for mm in [3,9]:
        if mm == 3 and yyyy < 2006 and yyyy > 1982: 
            file_paths = [f"{main_dir}CIS_10km_{yyyy}0226.nc", f"{main_dir}CIS_10km_{yyyy}0402.nc"]
        else:
            pattern = f"{main_dir}CIS_*_{yyyy}{mm:02d}*.nc"
            file_paths = glob.glob(pattern)

        if not file_paths:
            print(f"No files found for {yyyy}-{mm:02d}. Skipping.")
            continue

        monthly_datasets = []
        for file_path in file_paths:
            ds = xarray.open_dataset(file_path)
            #  Add other variables if we want them
            CT = ds['CT']

            # Replace coded figures in the data with aice concentration
            for i in range(len(listnum)):
                CT = CT.where(CT != listnum[i], replace[i])

            # Set the time coordinate for each dataset
            date_str = file_path.split('_')[-1].split('.')[0]
            datee = pd.to_datetime(date_str, format='%Y%m%d')

            CT = CT.assign_coords(time=datee).expand_dims('time')
            monthly_datasets.append(CT)

        if monthly_datasets:
            # Combine all datasets for the month
            combined_CT = xarray.concat(monthly_datasets, dim='time')
            
            # Compute the monthly mean
            monthly_mean_CT = combined_CT.mean(dim='time', skipna=True)
            
            # Set the time coordinate to the first day of the month for the monthly mean
            month_end_date = pd.Timestamp(year=yyyy, month=mm, day=15)
            monthly_mean_CT = monthly_mean_CT.assign_coords(time=month_end_date)

            CT_monthly_data.append(monthly_mean_CT)
        else:
            print(f"No valid data for {yyyy}-{mm:02d}. Skipping.")

    print(f'Year {yyyy}: Data Processing Completed')
    print('=====================================')

print('Merging Data Across all Years...')       
# Concatenate all monthly average data along the 'time' dimension
CT_all_combined = xarray.concat(CT_monthly_data, dim='time')
print('Data Merging Completed')

# Save the concatenated data to a NetCDF file
CT_all_combined.to_netcdf('/storage/mfol/obs/CT_march_sept_avrgd_1982_2020.nc')
print('Data Saved to NetCDF Format')

# Calculate and display the total execution time
end_time = time.time()
elapsed_time = end_time - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Data Processing Time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

#%% Compute sea ice extent per region (QEI and CAA). Does not cover LIAN
# Must change the CoordinatesQEI or CAA in line 305.
grid = xarray.open_dataset( "/mnt/bjerknes/CIS_10km_20100903.nc")['area']
combined = xarray.open_dataset('/storage/mfol/obs/CT_march_sept_avrgd_1982_2020.nc', decode_times=False)
coordinatesQEI = [(-117, 76.6),(-116.1,76.66),(-115,76.4),(-112,75.4),(-107,75.4),(-106,75.3),(-105.68,75.8),(-103.65,75.9),(-101.75,75.7),(-98.7,75.8),(-97.73,76.5),
                  (-96.4,76.67),(-94.38,76.43),(-90.55,76.6),(-89.5,76.55),
                  (-87.11,76.58),(-82.90,77.09),(-83.5,78.4),(-87.32, 78.17),(-88.84, 78.2),(-92.8,80.5),(-96.3, 80.3),(-99.05, 80.10),(-100.0, 79.9),
                  (-103.78, 79.35), (-105.5, 79.2),(-110.4, 78.75),(-113.10, 78.3),(-114.3, 78.08),(-115.06, 77.95),(-116.47, 77.56),
                  (-117, 76.6)]
coordinatesCAA = [(-128.19, 69.0),(-110.58,66.0),(-95.56,66.5),(-86.2,67.02),(-82.7,71.0),
                  (-81.9, 73.7),(-81.89, 74.52),(-91.67,74.79),(-91,75.55),
                   (-91.69,76.4),(-94.38,76.43),(-96.4,76.67),(-97.73,76.5),(-98.7,75.8),(-101.75,75.7),(-103.65,75.9),
                   (-105.68,75.8),(-106,75.3),(-107,75.4),(-112,75.4),(-115,76.4),(-116.1,76.66),(-120.2,76.6),(-122.0,76.2),
                   (-124.19, 74.32),(-123.2, 73.28),(-125.4, 72.18),(-128.19, 70.16)]
# maskBiggestContinent = np.load('/storage/mfol/obs/biggestCommonContinent_Since1982.npy')       

# Get polygon from coordinates
# Change the polygon to coordinatesQEI or coordinatesCAA to compute the extent per region.
lon, lat = zip(*coordinatesQEI)
path_data = []
for i in range(len(lon)):
    if i == 0:
        path_data.append((mpath.Path.MOVETO, (lon[i], lat[i])))
    else:
        path_data.append((mpath.Path.LINETO, (lon[i], lat[i])))

path_data.append((mpath.Path.CLOSEPOLY, (lon[0], lat[0])))
codes, verts = zip(*path_data)
path = mpath.Path(verts, codes)
vertices = path.vertices
contour_path = mpath.Path(vertices, codes[:len(vertices)])

# Create mask outside of polygon and common continent
lon_flat = np.ravel(combined.lon)
lat_flat = np.ravel(combined.lat)
mask = path.contains_points(np.column_stack((lon_flat, lat_flat))).reshape(combined.lon.shape)

for time in range(combined.time.shape[0]):
    # Uncomment next line and place in the .masked_where() call if want to apply biggest common continent
    # common = np.where(maskBiggestContinent, 0, combined['CT'][time,:,:])
    combined['CT'][time,:,:] = np.ma.masked_where(~mask, combined['CT'][time,:,:])

# Todo uncomment and comment next 2 lines if want to apply the biggest common continent
# masked_data = np.where((maskBiggestContinent), 0, 100)
# masked_data = np.ma.masked_where(~mask, masked_data)
# sverdrupBasinarea = masked_data/100 * grid
sverdrupBasinarea = combined['CT'][time,:,:]/100 * grid
areaSverdrup = np.nansum(sverdrupBasinarea,axis=(0,1)) # In km^2
print(areaSverdrup)

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-90)})
scat = ax.pcolormesh(grid.lon[:,:],grid.lat[:,:], combined['CT'][time,:,:], transform=ccrs.PlateCarree())
c = plt.colorbar(scat, ax=ax, orientation='horizontal',fraction=0.046)
ax.add_feature(cfeature.COASTLINE,color='gray',linewidth=0.3)
ax.gridlines(draw_labels=True)
ax.set_extent([-140, -60, 65, 90], crs=ccrs.PlateCarree()) # CAA

# Plot the polygon
path_patch = mpatches.PathPatch(contour_path, facecolor='#EEB479', alpha=0.4, edgecolor='#EEB479', lw=2, transform=ccrs.PlateCarree(),zorder=1)
ax.add_patch(path_patch)
ax.add_feature(cfeature.LAND, color='lightgray',zorder=2)
ax.add_feature(cfeature.COASTLINE,color='gray',linewidth=0.3,zorder=2)

# Compute SIE CIS
threshold = 0.15
start = date(1982,9,15) 
sie = []
sia = []
dates = []
for time in range(combined.time.shape[0]):
    aiceCut = combined['CT'][time,:,:] >= threshold
    siextent = np.ma.masked_where(~aiceCut, grid).sum(axis=(0,1))
    siarea = (combined['CT'][time,:,:] * grid).sum(axis=(0,1))
    sie.append(siextent)
    sia.append(float(siarea.values))
    dates.append(pd.to_datetime(start + timedelta(int(combined['time'][time].values))))

# Store SIE and SIA from CIS
output_file = './Results23May/CIS_marchSept_1982_1990_sie_QEI' 
dfqei = pd.DataFrame(columns=['Date','SIE', 'SIA'])
for i in range(len(dates)):
    row = pd.Series({
            'Date': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
            'SIE': sie[i],
            'SIA': sia[i]})
    dfqei = pd.concat([dfqei, pd.DataFrame([row])], ignore_index=True)
dfqei.to_excel(output_file+'.xlsx')

