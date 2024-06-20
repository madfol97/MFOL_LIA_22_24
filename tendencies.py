# %% import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray
import datetime
import pandas as pd
import matplotlib.colors as mcolors
from scipy import stats

# %%  get Grids and masks

cesm_grid = xarray.open_dataset(r'/aos/home/mfol/Data/CESM/domain.ocn.tx0.1v2_090218.nc', decode_times=False)
lia_mask_QEI = xarray.open_dataarray(r'./masked_data_QEI23May.nc')
lia_mask_CAA = xarray.open_dataarray(r'./masked_data_CAA23May.nc')
lia_mask_LIAN = xarray.open_dataarray(r'/aos/home/mfol/Results/IHESP/masked_data_LIANorth.nc')
cesmToKeep = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.185001-185912.nc', decode_times=True)

regions = [
    {'name': 'LIA-N','maskData': lia_mask_LIAN,'mask': None, 'nbPntsMask': None},
    {'name': 'QEI', 'maskData': lia_mask_QEI,'mask': None, 'nbPntsMask': None},
    {'name': 'CAA','maskData': lia_mask_CAA,'mask': None, 'nbPntsMask': None},
     ]

# Years have 365 days
def numberOfDaysPerMonth(month):
    thirdyDaysMonths = [4,6,9,11]
    thirdyOneDaysMonths = [1,3,5,7,8,10,12]
    if month in thirdyDaysMonths:
        return 30
    if month in thirdyOneDaysMonths:
        return 31
    return 28

# %% Find integrated tendencies (dynamic and thermodynamic) for each em per region, based on the melt season previously found

# Create obj with filename toward hist and proj with respective dates
# Comment and uncomment for each ensemble. Creates one excel per ensemble
allDatesHist = ['192001-192912','193001-193912','194001-194912','195001-195912','196001-196912','197001-197912','198001-198912','199001-199912','200001-200611']
allDatesProj = ['200601-201512','201601-202512','202601-203512','203601-204512','204601-205512','205601-206512','206601-207512','207601-208512','208601-209512','209601-210202']
allDates = [
            {'em': 'E1', 'emm': 'EM1', 'filedaidtt': 'storage/mfol/iHESP-HRCESM/EM1/hist/daidtt/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.daidtt.','filedaidtd': 'storage/mfol/iHESP-HRCESM/EM1/hist/daidtd/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.daidtd.','filedardg1dt': 'storage/mfol/iHESP-HRCESM/EM1/hist/dardg1dt/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.dardg1dt.','filedardg2dt': 'storage/mfol/iHESP-HRCESM/EM1/hist/dardg2dt/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.dardg2dt.','endfile':'.nc', 'fileMeltSeason':'./Results23May/CESM_HR_meltSeasonE1','dates':allDatesHist },
            {'em': 'E1', 'emm': 'EM1', 'filedaidtt': 'storage/mfol/iHESP-HRCESM/EM1/proj/daidtt/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.daidtt.','filedaidtd': 'storage/mfol/iHESP-HRCESM/EM1/proj/daidtd/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.daidtd.','filedardg1dt': 'storage/mfol/iHESP-HRCESM/EM1/proj/dardg1dt/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.dardg1dt.','filedardg2dt': 'storage/mfol/iHESP-HRCESM/EM1/proj/dardg2dt/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.dardg2dt.','endfile':'.nc', 'fileMeltSeason':'./Results23May/CESM_HR_meltSeasonE1','dates':allDatesProj },
            #{'em': 'E3', 'emm': 'EM3', 'filedaidtt': 'mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.','fileaice':'mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.','endfile':'-01.nc', 'fileMeltSeason':'./Results23May/CESM_HR_meltSeasonE3','dates':[str(year) for year in range(1920, 2006)]},
            #{'em': 'E3', 'emm': 'EM3', 'filedaidtt': 'mnt/qanik/iHESP/EM3/proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.31.003.cice.h.','fileaice':'mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.','endfile':'-01.nc', 'fileMeltSeason':'./Results23May/CESM_HR_meltSeasonE3','dates':[str(year) for year in range(2006, 2101)]},
            ]

meltSeasonFilePrefix = '/storage/mfol/iHESP-HRCESM/'
allYears = range(1920, 2101)

for region in regions: 
    # Apply ROI mask
    mask = region['mask']
    maskedData = np.ma.masked_where(mask, cesmToKeep['aice'][0,:,:])
    tareaMasked = np.ma.masked_where(mask, cesmToKeep['tarea'])
    daidtdTs = []
    daidttTs = []
    daidtdAdvOnlyTs = []
    meltStartTs = []
    freezeStartTs = []
    years = []
    filesList = []

    for obj in allDates :
        # Get melt season def
        meltSeasonDef = pd.read_excel(obj['fileMeltSeason'] + region['name']+ '.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)

        for datesString in obj['dates']:
            cesm = xarray.open_dataset(r'/'+ obj['filedaidtt'] + datesString + obj['endfile'], decode_times=True)
            daidtt = cesm['daidtt']
            cesm.close()

            time_bounds = daidtt.time[0].dt.year.item(), daidtt.time[-1].dt.year.item()
            if time_bounds[0] == time_bounds[1] and daidtt.time.shape[0] == 1:
                print(datesString)
                # get melt season start and end per year for the region
                filtered_df = meltSeasonDef[meltSeasonDef['Date'].dt.year == int(datesString)]
                start = filtered_df['meltStart'].values[0]
                end = filtered_df['freezeStart'].values[0]

                # create empty arrays
                thermoResults = np.full(daidtt[0,:,:].shape, 0.0)
                dynResults = np.full(daidtt[0,:,:].shape, 0.0)
                advResults = np.full(daidtt[0,:,:].shape, 0.0)
                ridgingLossResults = np.full(daidtt[0,:,:].shape, 0.0)
                ridgingGainResults = np.full(daidtt[0,:,:].shape, 0.0)
                
                if (not np.isnan(start) ) and ( start != 0):
                    for month in range(int(start),int(end)+1):
                        # Get tendencies
                        cesmMonth = xarray.open_dataset(r'/'+ obj['filedaidtt'] + datesString + '-' + str(month).zfill(2) + '.nc', decode_times=True)
                        daidttNa = cesmMonth['daidtt'][0,:,:].fillna(0.0)
                        daidtdNa = cesmMonth['daidtd'][0,:,:].fillna(0.0)
                        ridgingLossNa = cesmMonth['dardg1dt'][0,:,:].fillna(0.0)
                        ridgingGainNa = cesmMonth['dardg2dt'][0,:,:].fillna(0.0)
                        cesmMonth.close()

                        # apply ROI mask
                        daidttMonth = np.ma.masked_where(mask, daidttNa)
                        daidtdMonth = np.ma.masked_where(mask, daidtdNa)
                        ridgingLossMonth = np.ma.masked_where(mask, ridgingLossNa)
                        ridgingGainMonth = np.ma.masked_where(mask, ridgingGainNa)

                        # Get number of days to integrate over for each month. Partial month if start or end
                        if month == int(start):
                            ndays= numberOfDaysPerMonth(month)
                            nbDaysMonth = (1-(start - int(start))) * ndays
                        if month == int(end):
                            ndays= numberOfDaysPerMonth(month)
                            nbDaysMonth = (end - int(end)) * ndays
                        else: 
                            nbDaysMonth = numberOfDaysPerMonth(month)

                        # Tendency is % per day. convert to km2/month
                        thermoResults += daidttMonth/100 * tareaMasked * nbDaysMonth
                        dynResults += daidtdMonth/100 * tareaMasked * nbDaysMonth
                        ridgingLossResults += ridgingLossMonth/100 * tareaMasked * nbDaysMonth
                        ridgingGainResults += ridgingGainMonth/100 * tareaMasked * nbDaysMonth
                        
                    # Compute sum and store in excel eventually
                    sumthermoResults = np.nansum(thermoResults[:,:], axis =(0,1))
                    sumdynResults= np.nansum(dynResults[:,:], axis =(0,1))
                    sumadvResults= np.nansum(dynResults[:,:]+ridgingLossResults[:,:]-ridgingGainResults[:,:], axis =(0,1))

                    daidttTs.append(sumthermoResults)
                    daidtdTs.append(sumdynResults)
                    daidtdAdvOnlyTs.append(sumadvResults)
                else: 
                    daidttTs.append(np.nan)
                    daidtdTs.append(np.nan)
                    daidtdAdvOnlyTs.append(np.nan)

                years.append(pd.to_datetime(f"{datesString}"))
                filesList.append(datesString)


            else:
                cesmd = xarray.open_dataset(r'/'+ obj['filedaidtd'] + datesString + obj['endfile'], decode_times=True)
                daidtd = cesmd['daidtd']
                cesmd = xarray.open_dataset(r'/'+ obj['filedardg1dt'] + datesString + obj['endfile'], decode_times=True)
                dardg1dt = cesmd['dardg1dt']
                cesmd = xarray.open_dataset(r'/'+ obj['filedardg2dt'] + datesString + obj['endfile'], decode_times=True)
                dardg2dt = cesmd['dardg2dt']

                for year in range(time_bounds[0], time_bounds[1]):
                    print(year)
                    if year < 2100:
                        filtered_df = meltSeasonDef[meltSeasonDef['Date'].dt.year == year]
                        start = filtered_df['meltStart'].values[0]
                        end = filtered_df['freezeStart'].values[0]

                        thermoResults = np.full(daidtt[0,:,:].shape, 0.0)
                        dynResults = np.full(daidtt[0,:,:].shape, 0.0)
                        advResults = np.full(daidtt[0,:,:].shape, 0.0)
                        ridgingLossResults = np.full(daidtt[0,:,:].shape, 0.0)
                        ridgingGainResults = np.full(daidtt[0,:,:].shape, 0.0)

                        # Sum over melt season
                        if (not np.isnan(start)) and (start != 0):
                            for month in range(int(start),int(end)+1):
                                # Get tendencies
                                if month == 12 and year < 2100: # December is next year first month
                                    print('should not happen')
                                    daidttMonth = daidtt.sel(time=f'{year+1}-{1:02d}')[0,:,:].fillna(0.0)
                                    daidtdMonth = daidtd.sel(time=f'{year+1}-{1:02d}')[0,:,:].fillna(0.0)
                                    dardg1dtMonth = dardg1dt.sel(time=f'{year+1}-{1:02d}')[0,:,:].fillna(0.0)
                                    dardg2dtMonth = dardg2dt.sel(time=f'{year+1}-{1:02d}')[0,:,:].fillna(0.0)
                                elif month != 12:
                                    daidttMonth = daidtt.sel(time=f'{year}-{month+1:02d}')[0,:,:].fillna(0.0)
                                    daidtdMonth = daidtd.sel(time=f'{year}-{month+1:02d}')[0,:,:].fillna(0.0)
                                    dardg1dtMonth = dardg1dt.sel(time=f'{year}-{month+1:02d}')[0,:,:].fillna(0.0)
                                    dardg2dtMonth = dardg2dt.sel(time=f'{year}-{month+1:02d}')[0,:,:].fillna(0.0)
                                # apply ROI mask
                                daidttMonth = np.ma.masked_where(mask, daidttMonth)
                                daidtdMonth = np.ma.masked_where(mask, daidtdMonth)
                                ridgingLossMonth = np.ma.masked_where(mask, dardg1dtMonth)
                                ridgingGainMonth = np.ma.masked_where(mask, dardg2dtMonth)

                                # Get number of days to integrate over for each month. Partial month if start or end
                                if month == int(start):
                                    ndays= numberOfDaysPerMonth(month)
                                    nbDaysMonth = (1-(start - int(start))) * ndays
                                if month == int(end):
                                    ndays= numberOfDaysPerMonth(month)
                                    nbDaysMonth = (end - int(end)) * ndays
                                else: 
                                    nbDaysMonth = numberOfDaysPerMonth(month)

                                # Tendency is % per day. convert to km2/month
                                thermoResults += daidttMonth/100 * tareaMasked * nbDaysMonth
                                dynResults += daidtdMonth/100 * tareaMasked * nbDaysMonth
                                ridgingLossResults += ridgingLossMonth/100 * tareaMasked * nbDaysMonth
                                ridgingGainResults += ridgingGainMonth/100 * tareaMasked * nbDaysMonth
                        
                    # Compute sum and store in excel eventually
                    yAverageThermo = np.nansum(thermoResults[:,:], axis =(0,1))
                    yAverageDyn= np.nansum(dynResults[:,:], axis =(0,1))
                    sumadvResults= np.nansum(dynResults[:,:]+ridgingLossResults[:,:]-ridgingGainResults[:,:], axis =(0,1))
                    
                    daidttTs.append(yAverageThermo)
                    daidtdTs.append(yAverageDyn)
                    daidtdAdvOnlyTs.append(sumadvResults)
                    years.append(pd.to_datetime(f"{year}"))
                    filesList.append(datesString)
                cesmd.close()

    output_file = './Results23May/CESM_HR_tendencies'+ obj['em']+ region['name']
    df = pd.DataFrame(columns=['Date','daidtd', 'daidtdAdv','daidtt','file'])
    for i in range(len(years)):
        row = pd.Series({
                'Date': years[i].strftime('%Y-%m-%d %H:%M:%S'),
                'daidtd': daidtdTs[i],
                'daidtdAdv': daidtdAdvOnlyTs[i],
                'daidtt': daidttTs[i],
                'file':  filesList[i]})
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(output_file+'.xlsx')


# %% Plot one region tendencies SIA - Figure 12
regions = [
     {'region': 'LIA-N', 'data': [
            {'em': '1','file':'./Results23May/CESM_HR_tendenciesE1LIA-N.xlsx','tt': None, 'td': None},
            {'em': '3','file':'./Results23May/CESM_HR_tendenciesE3LIA-N.xlsx','tt': None, 'td': None},
            ],'ttdt':None,'tddt':None, 'ttmean':None,'ttmin':None,'ttmax':None,'tdmean':None,'tdmin':None,'tdmax':None,'freezeStart': None,'meltStart': None},
            {'region': 'QEI', 'data': [
            {'em': '1','file':'./Results23May/CESM_HR_tendenciesE1QEI.xlsx','tt': None, 'td': None},
            {'em': '3','file':'./Results23May/CESM_HR_tendenciesE3QEI.xlsx','tt': None, 'td': None},
            ], 'ttdt':None,'tddt':None, 'ttmean':None,'ttmin':None,'ttmax':None,'tdmean':None,'tdmin':None,'tdmax':None,'freezeStart': None,'meltStart': None},
            {'region': 'CAA', 'data': [
            {'em': '1','file':'./Results23May/CESM_HR_tendenciesE1CAA.xlsx','tt': None, 'td': None},
            {'em': '3','file':'./Results23May/CESM_HR_tendenciesE3CAA.xlsx','tt': None, 'td': None},
            ],'ttdt':None,'tddt':None, 'ttmean':None,'ttmin':None,'ttmax':None,'tdmean':None,'tdmin':None,'tdmax':None,'freezeStart': None,'meltStart': None},
    ]

fig, axs = plt.subplots(3,1,figsize=(12, 12))
plt.rc('font', size=14)
i = 0
colorThermo = '#f79920'
colorDyn = '#bd24c8'
colorThermo = '#298c8c'
colorDyn = '#ea801c'
colorRidge = '#f2c45f'
for region in regions: 
    dataOfAllEm = []
    for em in region['data']:
        dataOfAllEm.append(pd.read_excel(em['file'],  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime))
    merged_df = pd.concat(dataOfAllEm, ignore_index=True)  
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df['tt'] = pd.to_numeric(merged_df['daidtt'], errors='coerce')
    merged_df['td'] = pd.to_numeric(merged_df['daidtd'], errors='coerce')
    merged_df['tdAdv'] = pd.to_numeric(merged_df['daidtdAdv'], errors='coerce')
    
    region['ttmean'] = merged_df.groupby('Date')['tt'].mean().reset_index()['tt']
    region['ttmin'] = merged_df.groupby('Date')['tt'].min().reset_index()['tt']
    region['ttmax'] = merged_df.groupby('Date')['tt'].max().reset_index()['tt']
    region['tdmean'] = merged_df.groupby('Date')['td'].mean().reset_index()['td']
    region['tdmin'] = merged_df.groupby('Date')['td'].min().reset_index()['td']
    region['tdmax'] = merged_df.groupby('Date')['td'].max().reset_index()['td']
    region['tdtdAdvmean'] = merged_df.groupby('Date')['tdAdv'].mean().reset_index()['tdAdv']
    region['tdtdAdvmin'] = merged_df.groupby('Date')['tdAdv'].min().reset_index()['tdAdv']
    region['tdtdAdvmax'] = merged_df.groupby('Date')['tdAdv'].max().reset_index()['tdAdv']
    
    region['ridgingmean'] = region['tdmean']- region['tdtdAdvmean']
    dates = merged_df['Date'][0:181]
    
    axs[i].plot(dates,region['tdtdAdvmean'][0:181]/10**12, label='Advection',color=colorDyn, linewidth=2)
    axs[i].plot(dates,region['tdmean'][0:181]/10**12 - region['tdtdAdvmean'][0:181]/10**12 , label='Ridging',color=colorRidge, linewidth=2)
    axs[i].plot(dates,region['ttmean'][0:181]/10**12, label='Thermodynamic', color=colorThermo, linewidth=2)
    axs[i].fill_between(dates[0:np.shape(region['ttmean'][0:181])[0]], region['ttmin'][0:181]/10**12, region['ttmax'][0:181]/10**12, where=(region['ttmin'][0:181]/10**12< region['ttmax'][0:181]/10**12), interpolate=True, color=colorThermo, alpha=0.4)
    axs[i].fill_between(dates[0:np.shape(region['tdmean'][0:181])[0]], region['tdmin'][0:181]/10**12 - region['tdtdAdvmin'][0:181]/10**12 , region['tdmax'][0:181]/10**12 - region['tdtdAdvmax'][0:181]/10**12 , where=(region['tdmin'][0:181]/10**12 - region['tdtdAdvmin'][0:181]/10**12 < region['tdmax'][0:181]/10**12 - region['tdtdAdvmax'][0:181]/10**12 ), interpolate=True, color=colorDyn, alpha=0.4)
    axs[i].fill_between(dates[0:np.shape(region['ttmean'][0:181])[0]], region['tdtdAdvmin'][0:181]/10**12, region['tdtdAdvmax'][0:181]/10**12, where=(region['tdtdAdvmin'][0:181]/10**12< region['tdtdAdvmax'][0:181]/10**12), interpolate=True, color=colorDyn, alpha=0.4)
    
    axs[i].grid(linewidth=3, color='lightgrey')
    axs[i].axhline(y=0.0, color='grey', linestyle='-', linewidth=2)
    axs[i].set_xlim(1920,2080)
    axs[i].set_xlim(datetime.datetime(1920, 1, 1), datetime.datetime(2100, 1, 1))
    axs[i].xaxis.set_major_locator(mdates.YearLocator(base=20)) 
    axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


    i += 1
   
plt.xlim(1920, 2080)
plt.xlabel('Time')
plt.xlim(datetime.datetime(1920, 1, 1), datetime.datetime(2100, 1, 1))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=20)) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs[0].text(0.01, 0.9, 'a) LIA-N',  transform=axs[0].transAxes,fontsize=16)
axs[1].text(0.01, 0.9, 'b) QEI',  transform=axs[1].transAxes,fontsize=16)
axs[2].text(0.01, 0.9, 'c) CAA-S', transform=axs[2].transAxes,fontsize=16)
fig.text(0.02, 0.5, 'SIA ($10^6 km^2$)', va='center', rotation='vertical', fontsize=16)
legend = axs[0].legend()

plt.show()


# %% Scatter plot of thermodynamic and dynamic SIA loss integrated over the melt season
# Computes the linear correlation - Figure 13
fig, axs = plt.subplots(1,3,figsize=(18,10), facecolor='none')
i=0

plt.rc('font', size=14)
rs = []
rsend = []

# Creates a colorbar from blue to red for 20 years period
colors=[years for years in range(1920,2101)]
colorsForCmap=['#009392','#55c28f','#9CCB86','#C3D791','#E9E29C','#EEB479','#E88471','#DC6F78', '#CF597E' ]
cmap = mcolors.LinearSegmentedColormap.from_list('myCmap', colorsForCmap, N=9)
plt.rc('font', size=14)

for region in regions: 
    scat = axs[i].scatter(regions[i]['ttmean'][0:181]/10**12, regions[i]['tdmean'][0:181]/10**12, c=colors,s=30, cmap=cmap)
    axs[i].set_aspect('equal')
    tdmean = np.nan_to_num(np.array(regions[i]['tdmean'][0:181]/10**12))
    ttmean = np.nan_to_num(np.array(regions[i]['ttmean'][0:181]/10**12))
    r = stats.pearsonr(ttmean,tdmean)
    rs.append(r)
    if i == 0: 
        rpost2040 = stats.pearsonr(ttmean[120:155],tdmean[120:155])
    else:
        rpost2040 = stats.pearsonr(ttmean[120:170],tdmean[120:170])

    rsend.append(rpost2040)
    axs[i].grid()
    
    axs[i].axhline(0,  color = 'black', linewidth=1.5, zorder=0)
    axs[i].axvline(0,  color = 'black', linewidth=1.5, zorder=0)
    i+=1

c = fig.colorbar(scat, ax=axs.ravel().tolist(), fraction=0.015)
c.set_ticks([1920,1940,1960,1980,2000,2020,2040,2060,2080,2100])

axs[0].text(0.04, 0.94, 'a) LIA-N',  transform=axs[0].transAxes)
axs[1].text(0.04, 0.94, 'b) QEI',  transform=axs[1].transAxes)
axs[2].text(0.04, 0.94, 'c) CAA-S',  transform=axs[2].transAxes)
axs[0].text(0.04, 0.03, '$r_{1920-2100}$ = '+str(np.round(rs[0].statistic,2))+'\n $r_{2040-2075}$ = ' + str(np.round(rsend[0].statistic,2)),  transform=axs[0].transAxes)
axs[1].text(0.04, 0.03, '$r_{1920-2100}$ = ' + str(np.round(rs[1].statistic,2))+'\n $r_{2040-2090}$ = ' + str(np.round(rsend[1].statistic,2)),  transform=axs[1].transAxes)
axs[2].text(0.04, 0.03, '$r_{1920-2100}$ = '+str(np.round(rs[2].statistic,2))+'\n $r_{2040-2090}$ = ' + str(np.round(rsend[2].statistic,2)),  transform=axs[2].transAxes)

axs[0].set_xlim([-1.0, 0.1])
axs[0].set_ylim([-1.0, 0.1])
axs[1].set_xlim([-0.2, 0.1])
axs[1].set_ylim([-0.2, 0.1])
axs[2].set_xlim([-0.5, 0.1])
axs[2].set_ylim([-0.5, 0.1])
axs[0].set_ylabel('Dynamic SIA \n (10$^6$ km$^2$)')
axs[1].set_xlabel('Thermodynamic SIA (10$^6$ km$^2$)')
# %% FFT from detrended thermodynamic and dynamic tendencies and from fluxes
# Figure 14. FFT from mean.

def rolling_mean_detrend(data, window_size):
    kernel = np.ones(window_size) / window_size
    data = np.pad(data, window_size//2, mode='edge')
    rolling_mean = np.convolve(data, kernel, mode='same')[window_size//2+1:-window_size//2] 
    rolling_mean[0:20] = rolling_mean[20:40]
    detrended_data = data[window_size//2+1:-window_size//2] - rolling_mean
    
    return detrended_data, rolling_mean

detrendPeriod=20
for region in regions:
    region['ttdt'],region['trendtt'] = rolling_mean_detrend(region['ttmean'][~np.isnan(region['ttmean'])]/10**12,detrendPeriod)
    region['ridge'],region['trendridge'] = rolling_mean_detrend(region['ridgingmean'][~np.isnan(region['ridgingmean'])]/10**12, detrendPeriod)
    region['adv'],region['trendtdtdAdv'] = rolling_mean_detrend(region['tdtdAdvmean'][~np.isnan(region['tdtdAdvmean'])]/10**12, detrendPeriod)
    

fig, axs = plt.subplots(3,1, figsize=(12, 12), sharex=True)
i=0
j=0
ddfData = 2
ddfRednoise = 100

# From fluxes QEI
n=181
fluxesDivergence = np.load('./Results23May/fluxesQEIDiv.npy')
detrendfluxes, meanfluxes = rolling_mean_detrend(fluxesDivergence, 20)
fluxesfft = np.fft.fft((detrendfluxes-np.mean(detrendfluxes))[0:n])[0:(n//2)]
fluxesPower = np.abs(fluxesfft)**2
fluxesTotal=np.sum(fluxesPower)

# From fluxes CAA-S
n=181
fluxesDivergence = np.load('./Results23May/fluxesDivergenceCAAS.npy')
detrendfluxes, meanfluxes = rolling_mean_detrend(fluxesDivergence, 20)
fluxesfft = np.fft.fft((detrendfluxes-np.mean(detrendfluxes))[0:n])[0:(n//2)]
fluxesPowerCAA = np.abs(fluxesfft)**2
fluxesTotalCAA=np.sum(fluxesPowerCAA)

for region in regions: 
    advnoNan = region['adv'][~np.isnan(region['adv'])]
    ridgenoNan = region['ridge'][~np.isnan(region['ridge'])]
    ttnoNan = region['ttdt'][~np.isnan(region['ttdt'])]

    totaltd = region['adv'] + region['ridge']
    tdnoNan = totaltd[~np.isnan(totaltd)]

    advnoNanMean = np.mean(advnoNan)
    ridgenoNanMean = np.mean(ridgenoNan)
    tdnoNanMean = np.mean(tdnoNan)
    ttnoNanMean = np.mean(ttnoNan)
    n = len(advnoNan)

    alpha = 0.5     #red noise lag-one autocorrelation
    beta = np.sqrt(1.-alpha**2)  #beta, red noise parameter
    # Gilman et al. expression for the power spectrum of a red noise process
    rspec = []
    for h in np.arange(0,n//2,1):
        rspec.append((1.-alpha**2)/(1.-2.*alpha*np.cos(np.pi*(h)/(n//2))+alpha**2))
    fstat = stats.f.ppf(.99,ddfData,1000)
    spec99 = [fstat*m for m in rspec]
    fstat = stats.f.ppf(.95,ddfData,1000)
    spec95 = [fstat*m for m in rspec]
    
    # FTT
    ftadv = np.fft.fft((advnoNan))[0:(n//2)]
    ftridge = np.fft.fft((ridgenoNan))[0:(n//2)]
    fttd = np.fft.fft((tdnoNan))[0:(n//2)]
    fttt = np.fft.fft((ttnoNan))[0:(n//2)]
    # Power spectral density
    poweradv = np.abs(ftadv)**2
    powerridge = np.abs(ftridge)**2
    powertd = np.abs(fttd)**2
    powertt = np.abs(fttt)**2
    totaladv=np.sum(poweradv)
    totalridge=np.sum(powerridge)
    totaltd=np.sum(powertd)
    totaltt=np.sum(powertt)

    freqstd = np.fft.fftfreq(len(advnoNan),1)[0:n//2]

    period = 1/freqstd
    axs[i].plot(period, powertd/totaltd, color=colorDyn, linewidth=2,label='ridge')
    axs[i].plot(period, powertt/totaltt, color=colorThermo, linewidth=2,label='thermo')
    axs[i].plot(period,rspec/np.sum(rspec),'-', label = 'Red-noise fit', color = 'grey')
    axs[i].plot(period,spec99/np.sum(rspec),'-', label = '99% confidence', color = 'red')
    axs[i].set_ylabel('Power spectral density')
    axs[i].set_ylim([0,0.12])
    axs[i].set_xlim([2,8])

    axs[i].grid(True)
    i+=1

line5, =axs[1].plot(period,fluxesPower/fluxesTotal,linewidth=2, color=colorDyn, linestyle='dashed',label='Dynamic-Fluxes')
axs[2].plot(period,fluxesPowerCAA/fluxesTotalCAA,linewidth=2, color=colorDyn, linestyle='dashed',label='Dynamic-Fluxes')

       
axs[0].text(0.01, 0.9, 'a) LIA-N',transform=axs[0].transAxes, fontsize=14)
axs[1].text(0.01, 0.9, 'b) QEI', transform=axs[1].transAxes, fontsize=14)
axs[2].text(0.01, 0.9, 'c) CAA-S', transform=axs[2].transAxes, fontsize=14)

axs[2].set_xlabel('Years/Cycle')
handles, labels = axs[1].get_legend_handles_labels()
handles = [handles[0], handles[1],line5,handles[2],handles[3]]
fig.legend(handles=handles,labels=['Dynamic-MS','Thermodynamic-MS','Fluxes - Annual', 'Red-noise fit', '99% confidence'],loc='lower center',ncol=3,fontsize=12)
plt.show()

