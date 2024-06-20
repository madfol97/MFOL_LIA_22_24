import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cmocean

def plotQuantityColormeshWithGrid(
        x,y,data,
        extent=[-120,-70, 65, 85],
        label='',
        stepOfGrid=1,
        crs = ccrs.PlateCarree(),
        projection=ccrs.NorthPolarStereo(central_longitude=-90)):
    maxValue = np.max([np.abs(np.max(data)), np.abs(np.min(data))])

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.COASTLINE,color='gray',linewidth=0.3)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
    ax.set_extent(extent, crs=crs)
    scat = ax.pcolormesh(x, y, data, transform=crs, cmap='RdBu',  vmin=-maxValue, vmax=maxValue)
    cbar = plt.colorbar(scat, ax=ax,fraction=0.045)
    cbar.set_label(label)

    ax.pcolormesh(x[::stepOfGrid,::stepOfGrid], y[::stepOfGrid,::stepOfGrid],  data[::stepOfGrid,::stepOfGrid], facecolor="none", edgecolor='lightgray',  lw=0.5,
                      transform=crs,
                      antialiased=True)

    plt.tight_layout()
    plt.show()

def plotQuantityColormesh(
        x,y,data,
        extent=[-120,-70, 65, 85],
        label='',
        crs = ccrs.PlateCarree(),
        projection=ccrs.NorthPolarStereo(central_longitude=-90)):
    maxValue = np.max([np.abs(np.max(data)), np.abs(np.min(data))])

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.COASTLINE,color='gray',linewidth=0.3)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
    ax.set_extent(extent, crs=crs)
    scat = ax.pcolormesh(x, y, data, transform=crs, cmap='RdBu',  vmin=-maxValue, vmax=maxValue)
    cbar = plt.colorbar(scat, ax=ax, fraction=0.045)
    
    cbar.set_label(label)

    plt.tight_layout()
    plt.show()

def plotQuantity(
        x,y,data,
        extent=[-120,-70, 65, 85],
        label='',
        crs = ccrs.PlateCarree(),
        cmap=cmocean.cm.ice,
        projection=ccrs.NorthPolarStereo(central_longitude=-90)):

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.COASTLINE,color='gray',linewidth=0.3)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
    ax.set_extent(extent, crs=crs)
    scat = ax.pcolormesh(x, y, data, transform=crs, cmap=cmap)
    cbar = plt.colorbar(scat, ax=ax, fraction=0.045)
    
    cbar.set_label(label)

    plt.tight_layout()
    plt.show()


def plotQuiver(
        lon,lat,u,v, angleWithEast, extent=[-120,-70,65,85],
        label='',
        title='',
        step=25,
        vmax=55,
        crs=ccrs.PlateCarree(),
        cmap=cmocean.cm.speed,
        projection=ccrs.NorthPolarStereo(central_longitude=-90)

):
    """
    Quiver plot for u,v at each lon, lat coordinate given.
    lon, lat, u, v must be of same sizes
    u and v signs should be aligned with grid axis: ths function projects u and v
    on North and East axis with angleWithEast. Here angleWithEast is the angle
    between the x axis (u) and East (in radians)
    """
    u_component = np.array(u)
    v_component = np.array(v)
    lon=np.array(lon)
    lat=np.array(lat)
    speed=np.array(np.sqrt(u**2 + v**2))
    
    # pi/2 must be added for angle between y axis(v) and East
    u_north = u_component* np.sin(angleWithEast) + v_component* np.sin(np.pi/2 + angleWithEast) 
    u_east = u_component* np.cos(angleWithEast) + v_component* np.cos(np.pi/2 + angleWithEast)

    v_src_crs = u_east[:,:] / np.cos(lat / 180 * np.pi)
    u_src_crs = u_north[:,:]
    magnitude = np.sqrt(u_east[:,:]**2 + u_north[:,:]**2)
    magn_src_crs = np.sqrt(u_src_crs**2 + v_src_crs**2)
    arrow_magnitudes = np.sqrt((u_src_crs * magnitude / magn_src_crs) ** 2 + (v_src_crs * magnitude / magn_src_crs) ** 2)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})
    ax.set_title(title)
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.COASTLINE,color='gray',linewidth=0.3)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
    ax.set_extent(extent, crs=crs)

    scat = ax.pcolormesh(lon,lat,speed, transform=crs, cmap=cmap,  vmin=0,vmax=vmax)
    
    cbar = plt.colorbar(scat, ax=ax,fraction=0.045)
    cbar.set_label(label)
    quiv = ax.quiver(
        np.array(lon)[::step, ::step],
        np.array(lat)[::step, ::step],
        np.array((v_src_crs*magnitude / magn_src_crs)/arrow_magnitudes)[::step, ::step],
        np.array((u_src_crs*magnitude / magn_src_crs)/arrow_magnitudes)[::step, ::step],
        angles='xy',
        pivot='mid',
        scale=60*arrow_magnitudes,headwidth=2,headlength=4, headaxislength=4,
        transform=crs, color='gray')
    plt.show()

def plotTwoQuiverDatasets(lon,lat,u1,v1, u2, v2, angleWithEast, extent=[-120,-70,65,85],
        label='',
        step=25,
        crs=ccrs.PlateCarree(),
        cmap=cmocean.cm.speed,
        projection=ccrs.NorthPolarStereo(central_longitude=-90)):
    """
    Quiver plot for u1,v1, u2,v2 at each lon, lat coordinate given.
    lon, lat, u1, v1, u2, v2 must be of same sizes
    u and v signs should be aligned with grid axis: ths function projects u and v
    on North and East axis with angleWithEast. Here angleWithEast is the angle
    between the x axis (u) and East (in radians)
    The colormesh is the norm of u1, v1
    """
    lon=np.array(lon)
    lat=np.array(lat)
    
    u_component1 = np.array(u1)
    v_component1 = np.array(v1)
    speed1=np.array(np.sqrt(u1**2 + v1**2))

    u_north1 = u_component1* np.sin(angleWithEast) + v_component1* np.sin(np.pi/2 + angleWithEast)
    u_east1 = u_component1* np.cos(angleWithEast) + v_component1* np.cos(np.pi/2 + angleWithEast)

    v_src_crs1 = u_east1[:,:] / np.cos(lat / 180 * np.pi)
    u_src_crs1 = u_north1[:,:]
    magnitude1 = np.sqrt(u_east1[:,:]**2 + u_north1[:,:]**2)
    magn_src_crs1 = np.sqrt(u_src_crs1**2 + v_src_crs1**2)
    arrow_magnitudes1 = np.sqrt((u_src_crs1 * magnitude1 / magn_src_crs1) ** 2 + (v_src_crs1 * magnitude1 / magn_src_crs1) ** 2)

    u_component2 = np.array(u2)
    v_component2 = np.array(v2)

    u_north2 = u_component2* np.sin(angleWithEast) + v_component2* np.sin(np.pi/2 + angleWithEast)
    u_east2 = u_component2* np.cos(angleWithEast) + v_component2* np.cos(np.pi/2 + angleWithEast)
    v_src_crs_2 = u_east2[:,:] / np.cos(lat / 180 * np.pi)
    u_src_crs_2 = u_north2[:,:]
    magnitude2 = np.sqrt(u_east2[:,:]**2 + u_north2[:,:]**2)
    magn_src_crs2 = np.sqrt(u_src_crs_2**2 + v_src_crs_2**2)
    arrow_magnitudes2 = np.sqrt((u_src_crs_2 * magnitude2 / magn_src_crs2) ** 2 + (v_src_crs_2 * magnitude2 / magn_src_crs2) ** 2)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.COASTLINE,color='gray',linewidth=0.3)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
    ax.set_extent(extent, crs=crs)

    scat = ax.pcolormesh(lon,lat,speed1, transform=crs, cmap=cmap,  vmin=0)

    cbar = plt.colorbar(scat, ax=ax,fraction=0.045)
    cbar.set_label(label)
    quiv = ax.quiver(
        np.array(lon)[::step, ::step],
        np.array(lat)[::step, ::step],
        np.array((v_src_crs1*magnitude1 / magn_src_crs1)/arrow_magnitudes1)[::step, ::step],
        np.array((u_src_crs1*magnitude1 / magn_src_crs1)/arrow_magnitudes1)[::step, ::step],
        angles='xy',
        scale=60,headwidth=2,headlength=4, headaxislength=4,
        transform=crs, color='gray')
    quiv = ax.quiver(
        np.array(lon)[::step, ::step],
        np.array(lat)[::step, ::step],
        np.array((v_src_crs_2*magnitude2 / magn_src_crs2)/arrow_magnitudes2)[::step, ::step],
        np.array((u_src_crs_2*magnitude2 / magn_src_crs2)/arrow_magnitudes2)[::step, ::step],
        angles='xy',
        scale=60,headwidth=2,headlength=4, headaxislength=4,
        transform=crs, color='red')
    plt.show()

def plotThreeQuiverDatasets(lon,lat,u1,v1, u2, v2,u3,v3, angleWithEast, extent=[-120,-70,65,85],
        label='',
        step=25,
        crs=ccrs.PlateCarree(),
        cmap=cmocean.cm.speed,
        projection=ccrs.NorthPolarStereo(central_longitude=-90),
        vmax=0.35):
    """
    Quiver plot for u1,v1, u2,v2 at each lon, lat coordinate given.
    lon, lat, u1, v1, u2, v2 must be of same sizes
    u and v signs should be aligned with grid axis: ths function projects u and v
    on North and East axis with angleWithEast. Here angleWithEast is the angle
    between the x axis (u) and East (in radians)
    The colormesh is the norm of u1, v1
    """
    lon=np.array(lon)
    lat=np.array(lat)
    
    u_component1 = np.array(u1)
    v_component1 = np.array(v1)
    speed1=np.array(np.sqrt(u1**2 + v1**2))

    u_north1 = u_component1* np.sin(angleWithEast) + v_component1* np.sin(np.pi/2 + angleWithEast)
    u_east1 = u_component1* np.cos(angleWithEast) + v_component1* np.cos(np.pi/2 + angleWithEast)

    v_src_crs1 = u_east1[:,:] / np.cos(lat / 180 * np.pi)
    u_src_crs1 = u_north1[:,:]
    magnitude1 = np.sqrt(u_east1[:,:]**2 + u_north1[:,:]**2)
    magn_src_crs1 = np.sqrt(u_src_crs1**2 + v_src_crs1**2)
    arrow_magnitudes1 = np.sqrt((u_src_crs1 * magnitude1 / magn_src_crs1) ** 2 + (v_src_crs1 * magnitude1 / magn_src_crs1) ** 2)

    u_component2 = np.array(u2)
    v_component2 = np.array(v2)

    u_north2 = u_component2* np.sin(angleWithEast) + v_component2* np.sin(np.pi/2 + angleWithEast)
    u_east2 = u_component2* np.cos(angleWithEast) + v_component2* np.cos(np.pi/2 + angleWithEast)
    v_src_crs_2 = u_east2[:,:] / np.cos(lat / 180 * np.pi)
    u_src_crs_2 = u_north2[:,:]
    magnitude2 = np.sqrt(u_east2[:,:]**2 + u_north2[:,:]**2)
    magn_src_crs2 = np.sqrt(u_src_crs_2**2 + v_src_crs_2**2)
    arrow_magnitudes2 = np.sqrt((u_src_crs_2 * magnitude2 / magn_src_crs2) ** 2 + (v_src_crs_2 * magnitude2 / magn_src_crs2) ** 2)


    u_component3 = np.array(u3)
    v_component3 = np.array(v3)

    u_north3 = u_component3* np.sin(angleWithEast) + v_component3* np.sin(np.pi/2 + angleWithEast)
    u_east3 = u_component3* np.cos(angleWithEast) + v_component3* np.cos(np.pi/2 + angleWithEast)
    v_src_crs_3 = u_east3[:,:] / np.cos(lat / 180 * np.pi)
    u_src_crs_3 = u_north3[:,:]
    magnitude3 = np.sqrt(u_east3[:,:]**2 + u_north3[:,:]**2)
    magn_src_crs3 = np.sqrt(u_src_crs_3**2 + v_src_crs_3**2)
    arrow_magnitudes3 = np.sqrt((u_src_crs_3 * magnitude3 / magn_src_crs3) ** 2 + (v_src_crs_3 * magnitude3 / magn_src_crs3) ** 2)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.COASTLINE,color='gray',linewidth=0.3)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.4, linestyle='--')
    ax.set_extent(extent, crs=crs)

    scat = ax.pcolormesh(lon,lat,speed1, transform=crs, cmap=cmap,  vmin=0, vmax=vmax)

    cbar = plt.colorbar(scat, ax=ax,fraction=0.045)
    cbar.set_label(label)
    quiv = ax.quiver(
        np.array(lon)[::step, ::step],
        np.array(lat)[::step, ::step],
        np.array((v_src_crs1*magnitude1 / magn_src_crs1)/arrow_magnitudes1)[::step, ::step],
        np.array((u_src_crs1*magnitude1 / magn_src_crs1)/arrow_magnitudes1)[::step, ::step],
        angles='xy',
        scale=60,headwidth=2,headlength=4, headaxislength=4,
        transform=crs, color='#7A4988')
    quiv = ax.quiver(
        np.array(lon)[::step, ::step],
        np.array(lat)[::step, ::step],
        np.array((v_src_crs_2*magnitude2 / magn_src_crs2)/arrow_magnitudes2)[::step, ::step],
        np.array((u_src_crs_2*magnitude2 / magn_src_crs2)/arrow_magnitudes2)[::step, ::step],
        angles='xy',
        scale=60,headwidth=2,headlength=4, headaxislength=4,
        transform=crs, color='#ED2939', alpha=0.5)
    quiv = ax.quiver(
        np.array(lon)[::step, ::step],
        np.array(lat)[::step, ::step],
        np.array((v_src_crs_3*magnitude3 / magn_src_crs3)/arrow_magnitudes3)[::step, ::step],
        np.array((u_src_crs_3*magnitude3 / magn_src_crs3)/arrow_magnitudes3)[::step, ::step],
        angles='xy',
        scale=60,headwidth=2,headlength=4, headaxislength=4,
        transform=crs, color='#588BAE', alpha=0.5)
    plt.show()