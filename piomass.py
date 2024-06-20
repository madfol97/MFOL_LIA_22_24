# %%  import numpy 
import numpy as np

# %%
# From Zach Labbe proposed by PIOMASS git
def readPiomas(directory,vari,years,thresh):
    """
    Reads binary PIOMAS data
    """
    
    ### Retrieve Grid
    grid = np.genfromtxt(directory + 'Thickness/' + 'grid.dat')
    grid = np.reshape(grid,(grid.size))  
    
    ### Define Lat/Lon
    lon = grid[:grid.size//2]   
    lons = np.reshape(lon,(120,360))
    lat = grid[grid.size//2:]
    lats = np.reshape(lat,(120,360))
    
    if vari == 'thick':
        files = 'heff.H'
        directory = directory + 'Thickness/'
    elif vari == 'sic':
        files = 'area.H'
        directory = directory + 'Concentration/'
    elif vari == 'gice':
        files = 'gice.H'
        directory = directory + 'distribution/'
    
    ### Read data from binary into numpy arrays
    var = np.empty((len(years),12,120,360))
    if vari == 'gice':
        var = np.empty((len(years),12,12,120,360))
    for i in range(len(years)):
        data = np.fromfile(directory + files + str(years[i]),
                           dtype = 'float32')

        ### Reshape into [year,month,lat,lon]
        months = int(data.shape[0]/(120*360))
        if months < 12: # IT WAS !=12
            lastyearq = np.reshape(data,(months,120,360))
            emptymo = np.empty((12-months,120,360))
            emptymo[:,:,:] = np.nan
            lastyear = np.append(lastyearq,emptymo,axis=0)
            var[i,:,:,:] = lastyear
        elif months > 12:
            months = int(months/12)
            dataq = np.reshape(data,(months,12,120,360))
            emptymo = np.empty((12-months,12, 120,360))
            emptymo[:,:,:,:] = np.nan
            lastyear = np.append(dataq,emptymo,axis=0)
            var[i,:,:,:] = lastyear
        else:
            dataq = np.reshape(data,(months,120,360))        
            var[i,:,:,:] = dataq
    
    ### Mask out threshold values
    var[np.where(var <= thresh)] = np.nan

    print('Completed: Read "%s" data!' % (vari))   
    
    return lats,lons,var

def getSITMasked():
    directorydata = '/storage/mfol/obs/PIOMAS/'
    years = [year for year in range(1979,2019)]
    # sit has a shape of (41 years ,12 months,120,360)
    lats,lons,sit = readPiomas(directorydata,'thick',years,0)
    lats,lons,sic = readPiomas(directorydata,'sic',years,0)
    lats,lons,gice = readPiomas(directorydata,'gice',years,0)
    # Already NH upper than 40N
    return np.where(sic[:,4,:,:] > 0.15, sit[:,4,:,:], np.nan),sit,sic
sitMasked, sit, sic = getSITMasked()
