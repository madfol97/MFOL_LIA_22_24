# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import xarray
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy.io.matlab as matlab
import time

#%% Get grids and create masks
# Compute SIC per region. Must change the mask in line 42 and file name that will be saved.
start_time = time.time()
main_dir = "/storage/mfol/obs/nsidccdr/"
cellarea = xarray.open_dataset( "/storage/mfol/obs/nsidc/NSIDC0771_CellArea_PS_N25km_v1.0.nc")['cell_area']
grid = xarray.open_dataset( "/storage/mfol/obs/nsidc/NSIDC0771_LatLon_PS_N25km_v1.0.nc")
# longitude go from -180 to 180.
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
          
# LIA North from DeRepentigny, P., L.B. Tremblay, R. Newton, and S. Pfirman, (2016), 
# Patterns of sea ice retreat in the transition to a seasonally ice-free Arctic. 
# Journal of Climate, DOI: 10.1175/JCLI-D-15-0733.1. For the SITU system.
# I used LIAIndices.mat -> XY2LatLon(blia,alia)
coordLIANewton = matlab.loadmat('/aos/home/mfol/Results/IHESP/LIALatLonEdges.mat')
coordinatesLIANorth = []
for idx, elem in enumerate(coordLIANewton['lon']): 
    coordinatesLIANorth.append((elem[0], coordLIANewton['lat'][idx][0]))
coordinatesLIANorth.append(coordinatesLIANorth[0])

# Get polygon from coordinates
# Change the polygon to coordinatesQEI or coordinatesCAA
lon, lat = zip(*coordinatesCAA)
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
lon_flat = np.ravel(grid['longitude'])
lat_flat = np.ravel(grid['latitude'])

mask = path.contains_points(np.column_stack((lon_flat, lat_flat))).reshape(grid['longitude'].shape)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-90)})
scat = ax.pcolormesh(grid['longitude'][:,:],grid['latitude'][:,:], mask, transform=ccrs.PlateCarree())
# c = plt.colorbar(scat, ax=ax, orientation='horizontal',fraction=0.046)
ax.add_feature(cfeature.COASTLINE,color='gray',linewidth=0.3)
ax.gridlines(draw_labels=True)
ax.set_extent([-140, -60, 65, 90], crs=ccrs.PlateCarree()) # CAA


# %% Pan arctic mask
# To comment, or uncomment if want the mask to be NH - pan Arctic or regional from the upper cell.
lat_min, lat_max = 30.98, 90
mask = np.logical_and((grid.latitude >= lat_min), (grid.latitude <= lat_max))


# %% Compute SIA/SIE Obs. Make sure you have the right mask from previous cells.
siamarch = []
siasept = []
siemarch = []
siesept = []
dates = []
threshold = 0.15 # threshold to compute SIE. 
combinedCDR = []
def getIceConVar(data_vars):
    for key in data_vars:
        if '_ICECON' in key:
            matching_variable = data_vars[key]
            break
    return matching_variable

# <SENSOR>_ICECON where the sensor may be N07, F08, F11, F13, or F17; for example, “F17_ICECON”.
cellarea_msk = np.ma.masked_where(~mask,cellarea)
suffixesmarch = ['n07','n07','n07','n07','n07','n07','n07','n07','n07','f08','f08','f08','f08','f11','f11','f11','f11','f13','f13','f13','f13','f13','f13','f13','f13','f13','f13','f13','f13','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17']
suffixessept = ['n07','n07','n07','n07','n07','n07','n07','n07','f08','f08','f08','f08','f08','f11','f11','f11','f11','f13','f13','f13','f13','f13','f13','f13','f13','f13','f13','f13','f13','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17']
k = 0
for year in range(1979, 2024):
    print(year)
    aicesept = xarray.open_dataset(main_dir+'sept/seaice_conc_monthly_nh_'+str(year)+'09_'+ suffixessept[k] + '_v04r00.nc', decode_times=True)
    aicemarch = xarray.open_dataset(main_dir+'march/seaice_conc_monthly_nh_'+str(year)+'03_' + suffixesmarch[k] + '_v04r00.nc', decode_times=True)
    
    aiceseptVar = aicesept['cdr_seaice_conc_monthly'][0,:,:]
    aiceseptVar2 = np.where(grid['latitude'] > 85, 1.0, aiceseptVar)
    aiceseptVar3 = np.where(aiceseptVar2 > 1.0, np.nan, aiceseptVar2)
    combinedCDR.append(aiceseptVar3)
    aiceseptVar4 = np.where(aiceseptVar3 < threshold, np.nan, aiceseptVar3)
    aicesept_mskd = np.ma.masked_where(~mask, aiceseptVar3)
    aicesept_mskd4 = np.ma.masked_where(~mask, aiceseptVar4)

    aicemarchVar = aicemarch['cdr_seaice_conc_monthly'][0,:,:]
    aicemarchVar2 = np.where(grid['latitude'] > 85, 1.0, aicemarchVar)
    aicemarchVar3 = np.where(aicemarchVar2 > 1.0, np.nan, aicemarchVar2)
    aicemarchVar4 = np.where(aicemarchVar3 < threshold, np.nan, aicemarchVar3)
    aicemarch_mskd = np.ma.masked_where(~mask,aicemarchVar3)
    aicemarch_mskd4 = np.ma.masked_where(~mask,aicemarchVar4)

    siareasept= (np.nan_to_num(aicesept_mskd * cellarea_msk)).sum(axis=(0,1))
    siareamarch = (np.nan_to_num(aicemarch_mskd * cellarea_msk)).sum(axis=(0,1))


    cellarea_msk = np.ma.masked_where(~mask,cellarea)
    siereasept = np.nansum(np.ma.masked_where(~mask,np.where(np.nan_to_num(aicesept_mskd4) != 0, cellarea_msk, 0)),axis=(0,1))
    siereamarch =np.nansum(np.ma.masked_where(~mask,np.where(np.nan_to_num(aicemarch_mskd4) != 0, cellarea_msk, 0)),axis=(0,1))

    siasept.append(float(siareasept))
    siamarch.append(float(siareamarch))
    siesept.append(float(siereasept))
    siemarch.append(float(siereamarch))
    dates.append(pd.to_datetime(f"{year}-{9:02d}"))
    k+=1

output_file = './Results23May/NSDICCDR_1979_2023_sia_CAA' 
dfqei = pd.DataFrame(columns=['Date','SIA_03', 'SIA_09','SIE_03', 'SIE_09'])
for i in range(len(dates)):
    row = pd.Series({
            'Date': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
            'SIA_03': siamarch[i],
            'SIA_09': siasept[i],
            'SIE_03': siemarch[i],
            'SIE_09': siesept[i]})
    dfqei = pd.concat([dfqei, pd.DataFrame([row])], ignore_index=True)

