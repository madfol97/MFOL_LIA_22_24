# %% import packages
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import scipy.io.matlab as matlab

# %%  regions definition
cesm_grid = xarray.open_dataset(r'/aos/home/mfol/Data/CESM/domain.ocn.tx0.1v2_090218.nc', decode_times=False)


coordinatesQEI3 = [(360-117, 76.6),(360-116.1,76.66),(360-115,76.4),(360-112,75.4),(360-107,75.4),(360-106,75.3),(360-105.68,75.8),(360-103.65,75.9),(360-101.75,75.7),(360-98.7,75.8),(360-97.73,76.5),
                  (360-96.4,76.67),(360-94.38,76.43),(360-90.55,76.6),(360-89.5,76.55),
                  (360-87.11,76.58),(360-82.90,77.09),(360-83.5,78.4),(360-87.32, 78.17),(360-88.84, 78.2),(360-92.8,80.5),(360-96.3, 80.3),(360-99.05, 80.10),(360-100.0, 79.9),
                  (360-103.78, 79.35), (360-105.5, 79.2),(360-110.4, 78.75),(360-113.10, 78.3),(360-114.3, 78.08),(360-115.06, 77.95),(360-116.47, 77.56),
                  (360-117, 76.6)]

# LIA North from DeRepentigny, P., L.B. Tremblay, R. Newton, and S. Pfirman, (2016), 
# Patterns of sea ice retreat in the transition to a seasonally ice-free Arctic. 
# Journal of Climate, DOI: 10.1175/JCLI-D-15-0733.1. For the SITU system.
# I used LIAIndices.mat -> XY2LatLon(blia,alia)
coordLIANewton = matlab.loadmat('/aos/home/mfol/Results/IHESP/LIALatLonEdges.mat')
coordinatesLIANorth = []
for idx, elem in enumerate(coordLIANewton): 
    coordinatesLIANorth.append((360+elem[0], coordLIANewton[idx][0]))
coordinatesLIANorth.append(coordinatesLIANorth[0])


coordinatesCAA = [(360-128.19, 69.0),(360-110.58,66.0),(360-95.56,66.5),(360-86.2,67.02),(360-82.7,71.0),
                  (360-81.9, 73.7),(360-81.89, 74.52),(360-91.67,74.79),(360-91,75.55),
                   (360-91.69,76.4),(360-94.38,76.43),(360-96.4,76.67),(360-97.73,76.5),(360-98.7,75.8),(360-101.75,75.7),(360-103.65,75.9),
                   (360-105.68,75.8),(360-106,75.3),(360-107,75.4),(360-112,75.4),(360-115,76.4),(360-116.1,76.66),(360-120.2,76.6),(360-122.0,76.2),
                   (360-124.19, 74.32),(360-123.2, 73.28),(360-125.4, 72.18),(360-128.19, 70.16)]

# %% 
cesm = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.185001-185912.nc', decode_times=True)
tarea = cesm['tarea'][:,:]
aice = cesm['aice'][0,:,:]
lat_min, lat_max = 30.98, 90.0

# Get polygon from coordinates
# Change the polygon to coordinatesCAA, coordinatesQEI or coordinatesLIANorth
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

# Create mask outside of polygon 
lon_flat = np.ravel(cesm.TLON)
lat_flat = np.ravel(cesm.TLAT)
mask = path.contains_points(np.column_stack((lon_flat, lat_flat))).reshape(cesm.TLON.shape)

masked_data2 = np.where(~np.isnan(cesm['aice'][2,:,:]), 100, 0)
masked_data = np.ma.masked_where(~mask, masked_data2)
masked_data_filled = masked_data/100 * tarea
areaSverdrup = np.nansum(masked_data_filled,axis=(0,1))*10**(-6) # In km^2
print(areaSverdrup)

# Plot the masked data
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-90)})
scat = ax.pcolormesh(cesm_grid.xc[:,:], cesm_grid.yc[:,:], masked_data, transform=ccrs.PlateCarree())

ax.gridlines(draw_labels=True)
ax.set_extent([-140, -60, 65, 90], crs=ccrs.PlateCarree()) # CAA

#Plot the polygon
path_patch = mpatches.PathPatch(contour_path, facecolor='#EEB479', alpha=0.6, edgecolor='#EEB479', lw=2, transform=ccrs.PlateCarree(),zorder=1)
ax.add_patch(path_patch)
# ax.add_feature(cfeature.LAND, color='lightgray',zorder=2)
ax.add_feature(cfeature.COASTLINE,color='gray',linewidth=0.3,zorder=2)

#  Save the masked Array of the LIA
masked_data_array = xarray.DataArray(mask)
# Save the dataset to a NetCDF file - change the name
output_file = "./masked_data_QEI23May.nc"
masked_data_array.to_netcdf(output_file)
