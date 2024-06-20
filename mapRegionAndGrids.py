#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import xarray
from matplotlib.colors import ListedColormap
from ice_flux_algo import computeGatesOnce, findGateNormal, findSlopeFromCoord
from iHESP_helper import callGatesCESM_HR

# %% Plot with LIA and gates CESM
# FIGURE 1 

# Open grids and masks
cesm_grid = xarray.open_dataset(r'/aos/home/mfol/Data/CESM/domain.ocn.tx0.1v2_090218.nc', decode_times=False)
lia_mask_qei = xarray.open_dataarray(r'./masked_data_QEI23May.nc')
lia_mask_LiaN = xarray.open_dataarray(r'/aos/home/mfol/Results/IHESP/masked_data_LIANorth.nc')
lia_mask_CAA = xarray.open_dataarray(r'./masked_data_CAA23May.nc')
cesm = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.185001-185912.nc', decode_times=True)

# Get gate segments index
gates = callGatesCESM_HR()
ulon = cesm_grid['xv'][:,:,3]
ulat = cesm_grid['yv'][:,:,3]

# Projection parameters for the figure
min_lon, max_lon, min_lat, max_lat = -150, 15, 60, 90
central_lon, central_lat = (min_lon+ max_lon)/2, (min_lat+ max_lat)/2
central_lon = -75
projection= ccrs.LambertConformal(central_longitude=central_lon, central_latitude=central_lat)
crs = ccrs.PlateCarree()

# Colors of regions
colorThermo = '#298c8c'
colorDyn = '#ea801c'
colorRidge = '#f2c45f'
fig, ax = plt.subplots(figsize=(10, 12), subplot_kw={'projection': projection})


# Add LIA-N, QEI and CAA-S contours
contourf = ax.contourf(cesm_grid.xc, cesm_grid.yc, lia_mask_LiaN, levels=[0.5,1], colors=colorThermo,alpha=0.7, transform=ccrs.PlateCarree(), zorder=0)
contourf = ax.contourf(cesm_grid.xc, cesm_grid.yc, lia_mask_CAA, levels=[0.5,1], colors=colorDyn,alpha=0.7, transform=ccrs.PlateCarree(), zorder=0)
contourf = ax.contourf(cesm_grid.xc, cesm_grid.yc, lia_mask_qei, levels=[0.5,1], colors=colorRidge,alpha=0.9, transform=ccrs.PlateCarree(), zorder=0)

# Cut domain zoomed on the CAA + Fram Strait
n = 20
aoi = mpath.Path(
    list(zip(np.linspace(max_lon,min_lon, n), np.full(n, min_lat))) + \
    list(zip(np.full(n, min_lon), np.linspace(min_lat, max_lat, n))) + \
    list(zip(np.linspace(min_lon, max_lon, n), np.full(n, max_lat))) + \
    list(zip(np.full(n, max_lon), np.linspace(max_lat, min_lat, n)))
)

# Add features to the map
ax.add_feature(cfeature.LAND, color='#e7dfcc', zorder=2)
ax.add_feature(cfeature.OCEAN, color='#9cb4c3')
ax.coastlines(resolution='10m', color='#796d5a', zorder=2)
ax.set_boundary(aoi, transform=ccrs.PlateCarree())
ax.set_extent((max_lon, min_lon, max_lat, min_lat))
grdl = ax.gridlines(
        draw_labels=True, rotate_labels=False,
        x_inline=False, y_inline=False, color='lightgray', alpha=0.5,
    )
grdl.xlabel_style = {'size': 10, 'color': 'gray'}
grdl.ylabel_style = {'size': 10, 'color': 'gray'}
grdl.xlocator = mticker.FixedLocator(np.arange(-170, 181, 20))
grdl.ylocator = mticker.FixedLocator(np.arange(-60, 90, 5))
grdl.top_labels = False

# Text for all straits, channels, and oceans
ax.text(-81, 73.9, 'Lanc.',fontsize=10, fontstyle='italic',
        transform=ccrs.PlateCarree(), fontname='Serif')
ax.text(-80, 75.7, 'Jones',fontsize=10, fontstyle='italic',
        transform=ccrs.PlateCarree(), fontname='Serif')
ax.text(-73, 69.0, 'Baffin Bay',fontsize=10, rotation=-410, fontstyle='italic', 
        transform=ccrs.PlateCarree(), fontname='Serif')
ax.text(-130.0, 70.4, 'Amund.',fontsize=10, fontstyle='italic',
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=-410,ha='center',va='bottom')
ax.text(-128.8, 74.0, "M'CLure",fontsize=10, fontstyle='italic',
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=-410,ha='center',va='bottom')
ax.text(-119.8, 77.0, 'Ball.',fontsize=10, fontstyle='italic',
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=-48,ha='center',va='bottom')
ax.text(-118.2, 77.5, 'Wilkins',fontsize=10, fontstyle='italic',
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=-48,ha='center',va='bottom')
ax.text(-111.8, 78.5, 'Pr.GA.',fontsize=10, fontstyle='italic',
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=-48,ha='center',va='bottom')
ax.text(-106.7, 79.4, 'Peary',fontsize=10, fontstyle='italic',
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=-48,ha='center',va='bottom')
ax.text(-100.5, 80.1, 'Sv.',fontsize=10, fontstyle='italic',
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=-48,ha='center',va='bottom')
ax.text(-0.2, 80.0, 'Fram',fontsize=10, fontstyle='italic',
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=0,ha='center',va='bottom')
# ax.text(-97.4, 81.6, 'Nan.',fontsize=10, fontstyle='italic',
#         transform=ccrs.PlateCarree(), fontname='Serif',rotation=-48,ha='center',va='bottom')
ax.text(-84.0, 78.7, 'Eur.',fontsize=10, fontstyle='italic',bbox=dict(facecolor='grey',alpha=0.5, edgecolor='white', boxstyle='round'),
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=0,ha='center',va='bottom')
ax.text(-55.0, 82.2, 'Nares',fontsize=10, fontstyle='italic',
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=45,ha='center',va='bottom')
ax.text(-84.0, 76.5, 'H',fontsize=10, fontstyle='italic',bbox=dict(facecolor='grey',alpha=0.5, edgecolor='white', boxstyle='round'),
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=0,ha='center',va='bottom')
ax.text(-96.0, 74.9, 'Pen.',fontsize=10, fontstyle='italic',bbox=dict(facecolor='grey',alpha=0.5, edgecolor='white', boxstyle='round'),
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=0,ha='center',va='bottom')
ax.text(-104.6, 73.6, 'BM.',fontsize=10, fontstyle='italic',bbox=dict(facecolor='grey',alpha=0.5, edgecolor='white', boxstyle='round'),
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=-0,ha='center',va='bottom')
ax.text(-115.0, 74.4, 'Fit.',fontsize=10, fontstyle='italic',bbox=dict(facecolor='grey',alpha=0.5, edgecolor='white', boxstyle='round'),
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=0,ha='center',va='bottom')
ax.text(-70, 88.7, 'Arctic Ocean',fontsize=10, fontstyle='italic',
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=-0,ha='center',va='bottom')
ax.text(-85, 85, 'LIA-N',fontsize=16, fontstyle='italic', 
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=-0,bbox=dict(facecolor='grey',alpha=0.5, edgecolor='white', boxstyle='round'),color='white',ha='center',va='bottom')
ax.text(-100, 71, 'CAA-S',fontsize=16, fontstyle='italic', 
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=-0,bbox=dict(facecolor='grey',alpha=0.5, edgecolor='white', boxstyle='round'),color='white',ha='center',va='bottom')
ax.text(-100, 77, 'QEI',fontsize=16, fontstyle='italic', 
        transform=ccrs.PlateCarree(), fontname='Serif',rotation=-0,bbox=dict(facecolor='grey',alpha=0.5, edgecolor='white', boxstyle='round'),color='white',ha='center',va='bottom')
# ax.text(-105, 73.8, 'Parry Channel',fontsize=10, fontstyle='italic',
#         transform=ccrs.PlateCarree(), fontname='Serif',rotation=1,ha='center',va='bottom', color='white')

# Identify gates to plot
gatesToPlot = [0,1,2,3,4,5,6,7,8,9,10,12,15,20,22]
for id, inter in enumerate(gates):
    if id in gatesToPlot:
        for i in range(len(inter['gate'])-1):
                ax.plot([ulon[inter['gate'][i]], ulon[inter['gate'][i+1]]], [ulat[inter['gate'][i]], ulat[inter['gate'][i+1]]], c='black',transform=crs, zorder=2)
for i in range(len(gates[23]['gate'])-1):
        ax.scatter([ulon[inter['gate'][i]], ulon[inter['gate'][i+1]]], [ulat[inter['gate'][i]], ulat[inter['gate'][i+1]]], c='black',s=0.8,transform=crs, zorder=2)

# c = ax.pcolormesh(cesm_grid['xc'], cesm_grid['yc'],  regions[0]['mask'][:,:],transform=ccrs.PlateCarree())
# c = ax.pcolormesh(cesm_grid['xc'], cesm_grid['yc'],  maskedaice,transform=ccrs.PlateCarree())

plt.tight_layout()
plt.show()

# output_file_path = './mapDomain.png'
# plt.savefig(output_file_path, dpi=100)
plt.show()
# %%
# %% Plot model grid
# FIGURE 2

# Create custom cmap without white
original_cmap = plt.get_cmap('Blues')
start_index = 0.
end_index = 0.9
new_cmap = ListedColormap(original_cmap(np.linspace(start_index, end_index, 256)))

# Open grids from 3 GCMs
cesm_grid = xarray.open_dataset(r'/aos/home/mfol/Data/CESM/domain.ocn.tx0.1v2_090218.nc', decode_times=False)
cesm = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.185001-185912.nc', decode_times=True)
ulon = cesm_grid['xv'][:,:,3]
ulat = cesm_grid['yv'][:,:,3]
cesm_LE = xarray.open_dataset(r'/mnt/qanik/CESM2-LE/aice/EM_1251.001.nc', decode_times=False)
cesm_LR = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/LR/sept/B.E.13.BHISTC5.ne30g16.sehires38.003.sunway.cice.h.1850-09.nc', decode_times=True)
gridLR = xarray.open_dataset(r'/aos/home/mfol/Data/CESM/domain.ocn.gx1v6.090206.nc', decode_times=True)

# Projection parameters
central_lon = -75
projection= ccrs.NorthPolarStereo(central_longitude=central_lon)
crs = ccrs.PlateCarree()

fig, ax = plt.subplots(1,3, figsize=(12,12), subplot_kw={'projection': projection})#, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
  
# steps of grill cells to be shown in each grid
step =26
stepLE = 4
stepZoom =12 
stepLEZoom = 2

# Show only NH
ax[0].set_extent([-120, -60, 65, 88], crs=ccrs.PlateCarree())
ax[1].set_extent([-120, -60, 65, 88], crs=ccrs.PlateCarree())
ax[2].set_extent([-120, -60, 65, 88], crs=ccrs.PlateCarree())

# CESM1.3-HR
ax[0].pcolormesh(ulon, ulat,  cesm['tmask'],cmap=new_cmap,transform=ccrs.PlateCarree(),zorder=1)
ax[0].pcolormesh(ulon[5::stepZoom,5::stepZoom],
                   ulat[5::stepZoom,5::stepZoom],
                   cesm['aice'][0,5::stepZoom,5::stepZoom], facecolor="none", edgecolor='lightgrey',  lw=0.5,
                      transform=ccrs.PlateCarree(),
                      antialiased=True,zorder=2)

# CESM2-LE
ax[2].pcolormesh(gridLR['xv'][:,:,3], gridLR['yv'][:,:,3],  cesm_LE['tmask'],cmap=new_cmap,transform=ccrs.PlateCarree(),zorder=1)
ax[2].pcolormesh(gridLR['xv'][::stepLEZoom,::stepLEZoom,3], gridLR['yv'][::stepLEZoom,::stepLEZoom,3],  cesm_LE['tmask'][::stepLEZoom,::stepLEZoom], facecolor="none", edgecolor='lightgrey',  lw=0.5,
                      transform=ccrs.PlateCarree(),
                      antialiased=True,zorder=2)

# CESM1.3-LR
ax[1].pcolormesh(gridLR['xv'][:,:,3], gridLR['yv'][:,:,3], cesm_LR['tmask'],cmap=new_cmap,transform=ccrs.PlateCarree(),zorder=1)
ax[1].pcolormesh(gridLR['xv'][::stepLEZoom,::stepLEZoom,3], gridLR['yv'][::stepLEZoom,::stepLEZoom,3],  cesm_LR['aice'][0,::stepLEZoom,::stepLEZoom], facecolor="none", edgecolor='lightgrey',  lw=0.5,
                      transform=ccrs.PlateCarree(),
                      antialiased=True,zorder=2)

# Add labels
ax[0].text(0.016, 0.930, 'a) CESM1.3-HR', bbox=dict(facecolor='white', alpha=0.8), transform=ax[0].transAxes, fontsize=12, zorder=4)
ax[1].text(0.016, 0.930, 'b) CESM1.3-LR',  bbox=dict(facecolor='white', alpha=0.8),transform=ax[1].transAxes, fontsize=12,  zorder=4)
ax[2].text(0.016, 0.930, 'c) CESM2-LE',  bbox=dict(facecolor='white', alpha=0.8),transform=ax[2].transAxes, fontsize=12,  zorder=4)

# fig.tight_layout()
plt.show()
