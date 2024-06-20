# %% Import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import xarray
from datetime import datetime
import pandas as pd
import os
from piomass_hi import readPiomas


# %% Open results and plot SIE per region - Figure 5
# Define color code for each dataset
colorHR = '#2576af'
colorLR = '#37CAEC'
colorLE = '#c63e47'
colorObs = 'black'
colorObs2 = '#4F4F4F'

# Figure 
colorsForCmap=['#CF597E','#E88471','#EEB479','#E9E29C','#9CCB86',  '#55c28f','#598BAF']
regions = [{'region': 'a) LIA-N', 'data': [
            {'em': '1','file':'./Results23May/CESM_HR_sieE1LIA-N.xlsx','fileMarch':'./Results23May/CESM_HR_sieMarchE1LIA-N.xlsx',},
            #{'em': '2','file':'/aos/home/mfol/Data/CESM/CESM_HR/CESM_HR_SIA_SIE_Sept/CESM_HR_sieE2LIA-N.xlsx','fileMarch':'/aos/home/mfol/Data/CESM/CESM_HR/CESM_HR_SIA_SIE_March/CESM_HR_sieE2LIA-N.xlsx'},
            {'em': '3','file':'./Results23May/CESM_HR_sieE3LIA-N.xlsx','fileMarch':'./Results23May/CESM_HR_sieMarchE3LIA-N.xlsx'},
            ],'sia_mean': None,'sie_mean': None,'sie_min': None,'sie_max': None, 'color': colorsForCmap[6]},
            {'region': 'b) QEI', 'data': [
            {'em': '1','file':'./Results23May/CESM_HR_sieE1QEI.xlsx','fileMarch':'./Results23May/CESM_HR_sieMarchE1QEI.xlsx'},
             #{'em': '2','file':'/aos/home/mfol/Data/CESM/CESM_HR/CESM_HR_SIA_SIE_Sept/CESM_HR_sieE2QEI.xlsx','fileMarch':'/aos/home/mfol/Data/CESM/CESM_HR/CESM_HR_SIA_SIE_March/CESM_HR_sieE2QEI.xlsx'},
            {'em': '3','file':'./Results23May/CESM_HR_sieE3QEI.xlsx', 'fileMarch':'./Results23May/CESM_HR_sieMarchE3QEI.xlsx'},
            ],'sia_mean': None,'sie_mean': None,'sie_min': None,'sie_max': None, 'color': colorsForCmap[2]},
            {'region': 'c) CAA-S', 'data': [
            {'em': '1','file':'./Results23May/CESM_HR_sieE1CAA.xlsx','fileMarch':'./Results23May/CESM_HR_sieMarchE1CAA.xlsx'},
            #{'em': '2','file':'/aos/home/mfol/Data/CESM/CESM_HR/CESM_HR_SIA_SIE_Sept/CESM_HR_sieE2CAA.xlsx','fileMarch':'/aos/home/mfol/Data/CESM/CESM_HR/CESM_HR_SIA_SIE_March/CESM_HR_sieE2CAA.xlsx'},
            {'em': '3','file':'./Results23May/CESM_HR_sieE3CAA.xlsx','fileMarch':'./Results23May/CESM_HR_sieMarchE3CAA.xlsx'},
            ], 'sia_mean': None,'sie_mean': None,'sie_min': None,'sie_max': None, 'color': '#8B4513'},
    ]

# OBS NSIDC G02135 https://nsidc.org/data/g02135/versions/3n - 5 day extent and area
obs = pd.read_excel('/aos/home/mfol/Results/IHESP/CAA_NSDIC_SIE.xlsx',  sheet_name='Sheet1', parse_dates=['Years'], date_parser=pd.to_datetime)
obs['Year'] = pd.to_datetime(obs['Years'])
obs_mean = obs['ExtentMean']
obs_min = obs['ExtentMin']
obs_max = obs['ExtentMax']

# OBS CIS ICE CHARTS
obsCIS_CAA = pd.read_excel('./Results23May/CIS_marchSept_1982_1990_sie_CAA.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
obsCIS_CAA['Date'] = pd.to_datetime(obsCIS_CAA['Date'])
obsCIS_CAA_March_SIE = obsCIS_CAA['SIE'][2::12]
obsCIS_CAA_March_SIA = obsCIS_CAA['SIA'][2::12]
obsCIS_CAA_Sept_SIE = obsCIS_CAA['SIE'][8::12]
obsCIS_CAA_Sept_SIA = obsCIS_CAA['SIA'][8::12] 

obsCIS_QEI = pd.read_excel('./Results23May/CIS_marchSept_1982_1990_sie_QEI.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
obsCIS_QEI['Date'] = pd.to_datetime(obsCIS_QEI['Date'])

# # OBS FROM NSDIC V2 PRODUCTS OF AICE - NIMBUS 7
obsNSDIC_LIAN = pd.read_excel('/aos/home/mfol/Results/NSDICCDR/NSDICCDR_1979_2023_sia_LIA.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
obsNSDIC_LIAN['Date'] = pd.to_datetime(obsNSDIC_LIAN['Date'])
obsNSDIC_LIAN_March_SIA = obsNSDIC_LIAN['SIA_03']
obsNSDIC_LIAN_Sept_SIA = obsNSDIC_LIAN['SIA_09']
obsNSDIC_QEI = pd.read_excel('./Results23May/NSDICCDR_1979_2023_sia_QEI.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
obsNSDIC_QEI['Date'] = pd.to_datetime(obsNSDIC_QEI['Date'])
obsNSDIC_QEI_March_SIA = obsNSDIC_QEI['SIA_03']
obsNSDIC_QEI_Sept_SIA = obsNSDIC_QEI['SIA_09']
obsNSDIC_CAA = pd.read_excel('./Results23May/NSDICCDR_1979_2023_sia_CAA.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
obsNSDIC_CAA['Date'] = pd.to_datetime(obsNSDIC_CAA['Date'])
obsNSDIC_CAA_March_SIA = obsNSDIC_CAA['SIA_03']
obsNSDIC_CAA_Sept_SIA = obsNSDIC_CAA['SIA_09']

# LR data
septLR_CAA = pd.read_excel('./Results23May/CESM_LR_sept_CAA.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
septLR_CAA['Date'] = pd.to_datetime(septLR_CAA['Date'])
LR_CAA_Sept_SIE = septLR_CAA['SIE']
LR_CAA_Sept_SIA = septLR_CAA['AICE']
septLR_QEI = pd.read_excel('./Results23May/CESM_LR_sept_QEI.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
septLR_QEI['Date'] = pd.to_datetime(septLR_QEI['Date'])
LR_QEI_Sept_SIE = septLR_QEI['SIE']
LR_QEI_Sept_SIA = septLR_QEI['AICE']
septLR_LIAN = pd.read_excel('/aos/home/mfol/Results/CESM_LR/CESM_LR_sept_LIAN.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
septLR_LIAN['Date'] = pd.to_datetime(septLR_LIAN['Date'])
LR_LIAN_Sept_SIE = septLR_LIAN['SIE']
LR_LIAN_Sept_SIA = septLR_LIAN['AICE']


# CESM2LE 
directories = [ '/aos/home/mfol/Results/CESM_LE/LIAN_sept/', './Results23May/CESMLE/CAA_sept/', './Results23May/CESMLE/QEI_sept/']
sie_LE = {'LIA-N_sept': None, 'CAA_sept': None,  'QEI_sept': None}
indexes = [ 'LIA-N_sept',  'CAA_sept','QEI_sept']
i = 0
for directory in directories:
    ensembles = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            ensmbl = pd.read_excel(f,  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
            ensembles.append(ensmbl)
    merged_dfLE = pd.concat(ensembles, ignore_index=True)
    merged_dfLE['Date'] = pd.to_datetime(merged_dfLE['Date'])
    merged_dfLE['SIE'] = pd.to_numeric(merged_dfLE['SIE'], errors='coerce')
    mean_sie_LE = merged_dfLE.groupby('Date')['SIE'].mean().reset_index()['SIE']
    min_sie_LE = merged_dfLE.groupby('Date')['SIE'].min().reset_index()['SIE']
    max_sie_LE = merged_dfLE.groupby('Date')['SIE'].max().reset_index()['SIE']
    sie_LE[indexes[i]] = {'mean_sie_LE': mean_sie_LE, 'min_sie_LE': min_sie_LE, 'max_sie_LE': max_sie_LE}
    i += 1

# PLOT
fig, axs = plt.subplots(3,1, figsize=(12, 16))
plt.rc('font', size=16)
# HR
i=0
for region in regions: 
    dataOfAllEm = []
    dataOfAllEmMarch = []
    for em in region['data']:
        dataOfAllEm.append(pd.read_excel(em['file'],  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime))
        dataOfAllEmMarch.append(pd.read_excel(em['fileMarch'],  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime))
    merged_df = pd.concat(dataOfAllEm, ignore_index=True)  
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df['SIE'] = pd.to_numeric(merged_df['SIE'], errors='coerce')
    merged_df['SIA'] = pd.to_numeric(merged_df['SIA'], errors='coerce')
    region['sie_mean'] = merged_df.groupby('Date')['SIE'].mean().reset_index()['SIE']
    region['sie_min'] = merged_df.groupby('Date')['SIE'].min().reset_index()['SIE']
    region['sie_max'] = merged_df.groupby('Date')['SIE'].max().reset_index()['SIE']
    region['sia_mean'] = merged_df.groupby('Date')['SIA'].mean().reset_index()['SIA']
    region['sia_min'] = merged_df.groupby('Date')['SIA'].min().reset_index()['SIA']
    region['sia_max'] = merged_df.groupby('Date')['SIA'].max().reset_index()['SIA']
    
    merged_dfMarch = pd.concat(dataOfAllEmMarch, ignore_index=True)  
    merged_dfMarch['Date'] = pd.to_datetime(merged_dfMarch['Date'])
    merged_dfMarch['SIE'] = pd.to_numeric(merged_dfMarch['SIE'], errors='coerce')
    merged_dfMarch['SIA'] = pd.to_numeric(merged_dfMarch['SIA'], errors='coerce')
    region['sie_mean_march'] = merged_dfMarch.groupby('Date')['SIE'].mean().reset_index()['SIE']
    region['sie_min_march'] = merged_dfMarch.groupby('Date')['SIE'].min().reset_index()['SIE']
    region['sie_max_march'] = merged_dfMarch.groupby('Date')['SIE'].max().reset_index()['SIE']
    region['sia_mean_march'] = merged_dfMarch.groupby('Date')['SIA'].mean().reset_index()['SIA']
    region['sia_min_march'] = merged_dfMarch.groupby('Date')['SIA'].min().reset_index()['SIA']
    region['sia_max_march'] = merged_dfMarch.groupby('Date')['SIA'].max().reset_index()['SIA']
    
    # Plot Sept SIA
    axs[i].fill_between(merged_df['Date'][0:180], region['sie_min'][0:180]/10**12, region['sie_max'][0:180]/10**12, where=(region['sie_min'][0:180]/10**12 < region['sie_max'][0:180]/10**12), interpolate=True, color=colorHR, alpha=0.6)
    axs[i].plot(merged_df['Date'][0:180],region['sie_mean'][0:180]/10**12, label='CESM1.3-HR', color=colorHR, linewidth=2) #color=region['color'],linestyle='dotted'
    axs[i].text(0.01, 0.93,region['region'], transform=axs[i].transAxes)

    # axs[i].set_xlim(1920, 2100)
    axs[i].set_xlim(datetime(1920, 1, 1), datetime(2100, 1, 1))
    axs[i].xaxis.set_major_locator(mdates.YearLocator(base=20)) 
    axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    i+=1

# Add LR SIE
#shiftsLR = [1+(maxGridHR[0]-np.max(LR_LIAN_March_SIE/10**12))/(np.max(LR_LIAN_March_SIA)/10**12),1+(maxGridHR[1]- np.max(LR_QEI_March_SIA/10**12))/(np.max(LR_QEI_March_SIA)/10**12), 1+(maxGridHR[2]- np.max(LR_CAA_March_SIA/10**12))/(np.max(LR_CAA_March_SIA)/10**12)]
axs[2].plot(septLR_CAA['Date'], (np.array(LR_CAA_Sept_SIE)/10**12), color=colorLR,linewidth=2, label='CESM1.3-LR')
axs[1].plot(septLR_QEI['Date'], (np.array(LR_QEI_Sept_SIE)/10**12) , color=colorLR,linewidth=2)
axs[0].plot(septLR_LIAN['Date'], (np.array(LR_LIAN_Sept_SIE)/10**12) , color=colorLR,linewidth=2)

# CESM2LE
#shiftsLE = [1+(maxGridHR[0]-np.max(np.array(sie_LE['LIA-N_march']['mean_sie_LE'])/10**12))/(np.max(np.array(sie_LE['LIA-N_march']['mean_sie_LE']))/10**12),1+(maxGridHR[1]- np.max(np.array(sie_LE['QEI_march']['min_sie_LE'])/10**12))/(np.max(np.array(sie_LE['QEI_march']['min_sie_LE']))/10**12), 1+(maxGridHR[2]- np.max(np.array(sie_LE['CAA_march']['mean_sie_LE'])/10**12))/(np.max(np.array(sie_LE['CAA_march']['mean_sie_LE']))/10**12)]
axs[0].plot(merged_dfLE['Date'][0:250], np.array(sie_LE['LIA-N_sept']['mean_sie_LE'])/10**12 , color=colorLE,linewidth=2, label='CESM2-LE',zorder=-1)
axs[0].fill_between(merged_dfLE['Date'][0:250], np.array(sie_LE['LIA-N_sept']['min_sie_LE'])/10**12, np.array(sie_LE['LIA-N_sept']['max_sie_LE'])/10**12 , where=(np.array(sie_LE['LIA-N_sept']['min_sie_LE'])/10**12 < np.array(sie_LE['LIA-N_sept']['max_sie_LE'])/10**12), interpolate=True, color=colorLE, alpha=0.6,zorder=-1)
axs[1].plot(merged_dfLE['Date'][0:250], np.array(sie_LE['QEI_sept']['mean_sie_LE'])/10**12 , color=colorLE,linewidth=2, label='CESM2-LE',zorder=-1)
axs[1].fill_between(merged_dfLE['Date'][0:250], np.array(sie_LE['QEI_sept']['min_sie_LE'])/10**12 , np.array(sie_LE['QEI_sept']['max_sie_LE'])/10**12 , where=(np.array(sie_LE['QEI_sept']['min_sie_LE'])/10**12 < np.array(sie_LE['QEI_sept']['max_sie_LE'])/10**12), interpolate=True, color=colorLE, alpha=0.6,zorder=-1)
axs[2].plot(merged_dfLE['Date'][0:250], np.array(sie_LE['CAA_sept']['mean_sie_LE'])/10**12 , color=colorLE,linewidth=2, label='CESM2-LE',zorder=-1)
axs[2].fill_between(merged_dfLE['Date'][0:250], np.array(sie_LE['CAA_sept']['min_sie_LE'])/10**12 , np.array(sie_LE['CAA_sept']['max_sie_LE'])/10**12 , where=(np.array(sie_LE['CAA_sept']['min_sie_LE'])/10**12< np.array(sie_LE['CAA_sept']['max_sie_LE'])/10**12), interpolate=True, color=colorLE, alpha=0.6,zorder=-1)

# Add Observations NSIDC 
axs[2].plot(obsNSDIC_CAA['Date'], np.array(obsNSDIC_CAA_Sept_SIA)/10**12, color=colorObs,linewidth=2, label='CDR') #'#8B4513'
axs[1].plot(obsNSDIC_CAA['Date'], np.array(obsNSDIC_QEI_Sept_SIA)/10**12, color=colorObs,linewidth=2)
axs[0].plot(obsNSDIC_LIAN['Date'], np.array(obsNSDIC_LIAN_Sept_SIA)/10**12, color=colorObs,linewidth=2) #'#8B4513'


# # Add Observations CIS SIE
# #cisshift = [1+(maxGridHR[1] - np.max(np.array(obsCIS_QEI['SIA'][1::2])/10**6))/np.max(np.array(obsCIS_QEI['SIA'][1::2])/10**6),1+(maxGridHR[2] - np.max(np.array(obsCIS_CAA['SIA'][1::2])/10**6))/np.max(np.array(obsCIS_CAA['SIA'][1::2])/10**6)]
# axs[2].plot(obsCIS_CAA['Date'][1::2], np.array(obsCIS_CAA['SIE'][1::2])/10**6, color='red',linewidth=2, linestyle='dashed') #'#8B4513'
axs[2].plot(obsCIS_CAA['Date'][0::2], np.array(obsCIS_CAA['SIE'][0::2])/10**6, color=colorObs,linestyle='dashed',linewidth=2, label='CIS') #'#8B4513'
# axs[1].plot(obsCIS_QEI['Date'][1::2], np.array(obsCIS_QEI['SIE'][1::2])/10**6, color='red', linewidth=2,linestyle='dashed')
axs[1].plot(obsCIS_QEI['Date'][0::2], np.array(obsCIS_QEI['SIE'][0::2])/10**6, color=colorObs,linestyle='dashed',linewidth=2)


plt.xlim(1920, 2100)
plt.xlabel('Time',fontsize=14)
plt.xlim(datetime(1920, 1, 1), datetime(2100, 1, 1))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=20)) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs[0].set_ylim(0,1.25)
axs[1].set_ylim(0,0.2)
axs[2].set_ylim(0,0.55)

axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.text(0.04, 0.5, 'September SIE ($10^6 km^2$)', va='center', rotation='vertical')
plt.legend()
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)
plt.show()

# %% PLOT SIE PAN ARCTIC - ARTICLE Figure 3


HR_hist_E1 = pd.read_excel('/aos/home/mfol/Data/CESM/CESM_HR/CESM_HR_septsie_hist_E1.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
HR_hist_E2 = pd.read_excel('/aos/home/mfol/Data/CESM/CESM_HR/CESM_HR_septsie_hist_E2.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
HR_hist_E3 = pd.read_excel('/aos/home/mfol/Data/CESM/CESM_HR/CESM_HR_septsie_hist_E3.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
HR_proj_E1 = pd.read_excel('/aos/home/mfol/Data/CESM/CESM_HR/CESM_HR_septsie_proj_E1.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
HR_proj_E2 = pd.read_excel('/aos/home/mfol/Data/CESM/CESM_HR/CESM_HR_septsie_proj_E2.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
HR_proj_E3 = pd.read_excel('/aos/home/mfol/Data/CESM/CESM_HR/CESM_HR_septsie_proj_E3.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
cesm_HR_proj = [HR_proj_E1, HR_proj_E2, HR_proj_E3]
merged_df = pd.concat(cesm_HR_proj, ignore_index=True)
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
mean_sie_HR_proj = merged_df.groupby('Date')['SIE'].mean().reset_index()
min_sie_HR_proj = merged_df.groupby('Date')['SIE'].min().reset_index()
max_sie_HR_proj = merged_df.groupby('Date')['SIE'].max().reset_index()
mean_sia_HR_proj = merged_df.groupby('Date')['AICE'].mean().reset_index()
min_sia_HR_proj = merged_df.groupby('Date')['AICE'].min().reset_index()
max_sia_HR_proj = merged_df.groupby('Date')['AICE'].max().reset_index()

cesm_HR_hist= [HR_hist_E1, HR_hist_E2, HR_hist_E3]
merged_df = pd.concat(cesm_HR_hist, ignore_index=True)
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
mean_sie_HR_hist = merged_df.groupby('Date')['SIE'].mean().reset_index()
min_sie_HR_hist = merged_df.groupby('Date')['SIE'].min().reset_index()
max_sie_HR_hist = merged_df.groupby('Date')['SIE'].max().reset_index()
mean_sia_HR_hist = merged_df.groupby('Date')['AICE'].mean().reset_index()
min_sia_HR_hist = merged_df.groupby('Date')['AICE'].min().reset_index()
max_sia_HR_hist = merged_df.groupby('Date')['AICE'].max().reset_index()
crossover_date_HR = None
indexCrossover_HR = None
indexCrossover_HR  = np.min(np.where(mean_sie_HR_proj['SIE']/10**12 <= 1.0)[0])
crossover_date_HR = mean_sie_HR_proj['Date'][indexCrossover_HR]

# CESM LE
ensembles = []
directory = '/aos/home/mfol/Data/CESM/CESM_LE/CESM_LE_NH2'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        ensmbl = pd.read_excel(f,  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
        ensembles.append(ensmbl)
    else: print(filename + 'not a file')
merged_df = pd.concat(ensembles, ignore_index=True)
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
mean_sie_LE = merged_df.groupby('Date')['SIE'].mean().reset_index()
min_sie_LE = merged_df.groupby('Date')['SIE'].min().reset_index()
max_sie_LE = merged_df.groupby('Date')['SIE'].max().reset_index()
mean_sia_LE = merged_df.groupby('Date')['AICE'].mean().reset_index()
min_sia_LE = merged_df.groupby('Date')['AICE'].min().reset_index()
max_sia_LE = merged_df.groupby('Date')['AICE'].max().reset_index()
crossover_date_LE = None
indexCrossover_LE = None
indexCrossover_LE  = np.min(np.where(mean_sie_LE['SIE']/10**12 <= 1.0)[0])
crossover_date_LE = mean_sie_LE['Date'][indexCrossover_LE]

# OBS NSIDC from V2 PRODUCTS OF AICE - NIMBUS 7 aice concentration I computed pan arctic in nsdic_aice.py
obs2 = pd.read_excel('/aos/home/mfol/Results/NSDICCDR/NSDICCDR_1979_2023_sia_PanArctic.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
obs2['Date'] = pd.to_datetime(obs2['Date'])
obs2_mean = obs2['SIE_09']

# Add CESMHR
HR_e1 = pd.read_excel('/aos/home/mfol/Results/IHESP/ThicknessAveraged/CESM_HR_SIT/CESM_HR_sitPanArcticE1.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
HR_e2 = pd.read_excel('/aos/home/mfol/Results/IHESP/ThicknessAveraged/CESM_HR_SIT/CESM_HR_sitPanArcticE2.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
HR_e3 = pd.read_excel('/aos/home/mfol/Results/IHESP/ThicknessAveraged/CESM_HR_SIT/CESM_HR_sitPanArcticE3.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
cesm_HR = [HR_e1, HR_e2, HR_e3]
merged_df = pd.concat(cesm_HR, ignore_index=True)
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
mean_hi_HR_proj_sit = merged_df.groupby('Date')['hi'].mean().reset_index()['hi'][0:250]
min_hi_HR_proj_sit = merged_df.groupby('Date')['hi'].min().reset_index()['hi'][0:250]
max_hi_HR_proj_sit = merged_df.groupby('Date')['hi'].max().reset_index()['hi'][0:250]

# Add CESMLE
ensembles = []
for filename in os.listdir('/aos/home/mfol/Results/CESM_LE/PanArctic_May/'):
    f = os.path.join('/aos/home/mfol/Results/CESM_LE/PanArctic_May/', filename)
    if os.path.isfile(f):
        ensmbl = pd.read_excel(f,  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
        ensembles.append(ensmbl)
merged_dfLE = pd.concat(ensembles, ignore_index=True)
merged_dfLE['Date'] = pd.to_datetime(merged_dfLE['Date'])
merged_dfLE['hi'] = pd.to_numeric(merged_dfLE['hi'], errors='coerce')
mean_sit_LE = merged_dfLE.groupby('Date')['hi'].mean().reset_index()['hi']
min_sit_LE = merged_dfLE.groupby('Date')['hi'].min().reset_index()['hi']
max_sit_LE = merged_dfLE.groupby('Date')['hi'].max().reset_index()['hi']

# Add CESMLR 
cesmsie =  pd.read_excel('/aos/home/mfol/Results/CESM_LR/CESM_LR_septsie_proj.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
LR_e1_h = pd.read_excel('/aos/home/mfol/Results/IHESP/ThicknessAveraged/CESM_HR_SIT/CESM_LR_sitPanArctic.xlsx',  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)
meanLR_h = LR_e1_h['hi'][0:250]
fig, axs = plt.subplots(3,1,figsize=(12,16), facecolor='none')

plt.rcParams['font.size'] = 16

# GET PIOMAS SIT masked for SIC 15%
directorydata = '/storage/mfol/obs/PIOMAS/'
years = [year for year in range(1978,2021)]
# sit has a shape of (41 years ,12 months,120,360)
lats,lons,sit = readPiomas(directorydata,'thick',years,0)
lats,lons,sic = readPiomas(directorydata,'sic',years,0)
sitMasked = np.where(sic[:,4,:,:] > 0.15, sit[:,4,:,:], np.nan)


# PAN ARCTIC SIE
# HR Hist
axs[0].plot(mean_sie_HR_hist['Date'], mean_sie_HR_hist['SIE']/10**12, color=colorHR, linewidth=2)
axs[0].fill_between(min_sie_HR_hist['Date'], min_sie_HR_hist['SIE']/10**12, max_sie_HR_hist['SIE']/10**12, where=(min_sie_HR_hist['SIE']/10**12 < max_sie_HR_hist['SIE']/10**12), interpolate=True, color=colorHR, alpha=0.6)
# # HR proj
axs[0].plot(mean_sie_HR_proj['Date'], mean_sie_HR_proj['SIE']/10**12, color=colorHR,label='CESM1.3-HR', linewidth=2)
axs[0].fill_between(min_sie_HR_proj['Date'], min_sie_HR_proj['SIE']/10**12, max_sie_HR_proj['SIE']/10**12, where=(min_sie_HR_proj['SIE']/10**12 < max_sie_HR_proj['SIE']/10**12), interpolate=True, color=colorHR, alpha=0.6)
#CESM1.3 LE JUMEAU
axs[0].plot(cesmsie['Date'], cesmsie['SIE']/10**12, color=colorLR, label='CESM1.3-LR')
# # LE
axs[0].plot(mean_sie_LE['Date'], mean_sie_LE['SIE']/10**12, color=colorLE,label='CESM2-LE', linewidth=2,zorder=-1)
axs[0].fill_between(min_sie_LE['Date'], min_sie_LE['SIE']/10**12, max_sie_LE['SIE']/10**12, where=(min_sie_LE['SIE']/10**12 < max_sie_LE['SIE']/10**12), interpolate=True, color=colorLE, alpha=0.6,zorder=-1)
# # Ob
axs[0].plot(obs2['Date'], obs2_mean/10**12, color=colorObs,label='CDR', linewidth=2)
axs[0].axhline(y=1.0, color='black', linestyle='--', linewidth=1)
axs[0].set_ylim(0,10)
axs[0].set_ylabel('September SIE ($10^6 km^2$)',fontsize=16)
axs[0].grid(zorder=4,color='lightgrey')
axs[0].legend()
axs[0].text(0.01, 0.93,'a)', transform=axs[0].transAxes)
axs[0].set_xlim(datetime(1850, 1, 1), datetime(2100, 1, 1))
axs[0].xaxis.set_major_locator(mdates.YearLocator(base=20))  # Adjust the base value to control the tick intervals
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the ticks as years


# PAN ARCTIC SIA
# HR Hist
axs[1].plot(mean_sie_HR_hist['Date'], mean_sia_HR_hist['AICE']/10**14, color=colorHR, linewidth=2)
axs[1].fill_between(min_sie_HR_hist['Date'], min_sia_HR_hist['AICE']/10**14, max_sia_HR_hist['AICE']/10**14, where=(min_sia_HR_hist['AICE']/10**14 < max_sia_HR_hist['AICE']/10**14), interpolate=True, color=colorHR, alpha=0.6)
# # HR proj
axs[1].plot(mean_sie_HR_proj['Date'], mean_sia_HR_proj['AICE']/10**14, color=colorHR,label='CESM1.3-HR', linewidth=2)
axs[1].fill_between(min_sie_HR_proj['Date'], min_sia_HR_proj['AICE']/10**14, max_sia_HR_proj['AICE']/10**14, where=(min_sia_HR_proj['AICE']/10**14 < max_sia_HR_proj['AICE']/10**14), interpolate=True, color=colorHR, alpha=0.6)
#CESM1.3 LE JUMEAU
axs[1].plot(cesmsie['Date'], cesmsie['AICE']/10**14, color=colorLR, label='CESM1.3-LR')
# # LE
axs[1].plot(mean_sie_LE['Date'], mean_sia_LE['AICE']/10**12, color=colorLE,label='CESM2-LE', linewidth=2,zorder=-1)
axs[1].fill_between(min_sie_LE['Date'], min_sia_LE['AICE']/10**12, max_sia_LE['AICE']/10**12, where=(min_sia_LE['AICE']/10**12 < max_sia_LE['AICE']/10**12), interpolate=True, color=colorLE, alpha=0.6,zorder=-1)
# # Obs
axs[1].plot(obs2['Date'], obs2['SIA_09']/10**12, color=colorObs,label='CDR', linewidth=2)
axs[1].set_ylim(0,10)
axs[1].set_ylabel('September SIA ($10^6 km^2$)',fontsize=16)
axs[1].grid(zorder=4,color='lightgrey')
axs[1].legend()
axs[1].text(0.01, 0.93,'b)', transform=axs[1].transAxes)
axs[1].set_xlim(datetime(1850, 1, 1), datetime(2100, 1, 1))
axs[1].xaxis.set_major_locator(mdates.YearLocator(base=20))  # Adjust the base value to control the tick intervals
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the ticks as years

# PAN ARCTIC SIT
axs[2].text(0.01, 0.93,'c)', transform=axs[2].transAxes)
# # CESMHR
axs[2].plot(merged_df['Date'][0:250], mean_hi_HR_proj_sit, color=colorHR, label='CESM1.3-HR',linewidth=2)
axs[2].fill_between(merged_df['Date'][0:250],min_hi_HR_proj_sit,max_hi_HR_proj_sit, where=(min_hi_HR_proj_sit < max_hi_HR_proj_sit), interpolate=True, color=colorHR, alpha=0.6)
# #CESMLR
axs[2].plot(merged_df['Date'][0:250], meanLR_h, color=colorLR, label='CESM1.3-LR',linewidth=2)
# # CESMLE
axs[2].plot(merged_dfLE['Date'][0:251], mean_sit_LE, color=colorLE, label='CESM2-LE', zorder=-1,linewidth=2)
axs[2].fill_between(merged_dfLE['Date'][0:251], min_sit_LE,max_sit_LE, where=(min_sit_LE < max_sit_LE), interpolate=True, color=colorLE, alpha=0.6,zorder=-1)
# #PIOMAS
axs[2].plot(merged_df['Date'][128:171], np.nanmean(sitMasked, axis=(1,2)), color=colorObs, label='PIOMAS',linewidth=2)
axs[2].set_ylim([0,3.5])
axs[2].set_ylabel('May SIT (m)',fontsize=16)
axs[2].grid(zorder=4,color='lightgrey')
axs[2].legend()
axs[2].set_xlim(datetime(1850, 1, 1), datetime(2100, 1, 1))
axs[2].xaxis.set_major_locator(mdates.YearLocator(base=20))  # Adjust the base value to control the tick intervals
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the ticks as years

plt.show()


# %% PLOT FIGURE SIA thermo and dyn contribution Plot SIA march to sept
# with dynamic and thermodynamic contributions. Figure 11.
# 
regionstendency = [
     {'region': 'LIA-N', 'data': [
            {'em': '1','file':'./Results23May/CESM_HR_tendenciesE1LIA-N.xlsx','fileMarch':'./Results23May/CESM_HR_sieMarchE1LIA-N.xlsx','tt': None, 'td': None},
            {'em': '3','file':'./Results23May/CESM_HR_tendenciesE3LIA-N.xlsx','fileMarch':'./Results23May/CESM_HR_sieMarchE3LIA-N.xlsx','tt': None, 'td': None},
            ],'ttdt':None,'tddt':None, 'ttmean':None,'ttmin':None,'ttmax':None,'tdmean':None,'tdmin':None,'tdmax':None,'freezeStart': None,'meltStart': None, 'color': colorsForCmap[6]},
            {'region': 'QEI', 'data': [
            {'em': '1','file':'./Results23May/CESM_HR_tendenciesE1QEI.xlsx','tt': None, 'td': None,'fileMarch':'./Results23May/CESM_HR_sieMarchE1QEI.xlsx'},
            {'em': '3','file':'./Results23May/CESM_HR_tendenciesE3QEI.xlsx','tt': None, 'td': None,'fileMarch':'./Results23May/CESM_HR_sieMarchE3QEI.xlsx'},
            ], 'ttdt':None,'tddt':None, 'ttmean':None,'ttmin':None,'ttmax':None,'tdmean':None,'tdmin':None,'tdmax':None,'freezeStart': None,'meltStart': None, 'color': colorsForCmap[2]},
            {'region': 'CAA', 'data': [
            {'em': '1','file':'./Results23May/CESM_HR_tendenciesE1CAA.xlsx','tt': None, 'td': None,'fileMarch':'./Results23May/CESM_HR_sieMarchE1CAA.xlsx'},
            {'em': '3','file':'./Results23May/CESM_HR_tendenciesE3CAA.xlsx','tt': None, 'td': None,'fileMarch':'./Results23May/CESM_HR_sieMarchE3CAA.xlsx'},
            ],'ttdt':None,'tddt':None, 'ttmean':None,'ttmin':None,'ttmax':None,'tdmean':None,'tdmin':None,'tdmax':None,'freezeStart': None,'meltStart': None, 'color': '#8B4513'},
            
    ]

for region in regionstendency:
    dataOfAllEm = []
    dataOfAllEmMarch = []
    for em in region['data']:
        dataOfAllEm.append(pd.read_excel(em['file'],  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime))
        dataOfAllEmMarch.append(pd.read_excel(em['fileMarch'],  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime))
    
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
    dates = merged_df['Date'][0:181]

    merged_dfMarch = pd.concat(dataOfAllEmMarch, ignore_index=True) 
    merged_dfMarch['SIA'] = pd.to_numeric(merged_dfMarch['SIA'], errors='coerce')
    region['sie_meanMarch'] = merged_dfMarch.groupby('Date')['SIA'].mean().reset_index()['SIA']

fig, axs = plt.subplots(3,1, figsize=(12, 20))
plt.rc('font', size=14)
years = list(range(1920, 2101))
colorThermo = '#f79920'
colorDyn = '#bd24c8'
colorThermo = '#298c8c'
colorDyn = '#f1a226'
colorRidge = '#f2c45f'
i = 0
for region in regions: 
    dataDyn = regionstendency[i]['sie_meanMarch'][0:-1]/10**12 + regionstendency[i]['tdtdAdvmean'][0:181]/10**12
    dataDRidge = regionstendency[i]['sie_meanMarch'][0:-1]/10**12 + regionstendency[i]['tdmean'][0:181]/10**12- regionstendency[i]['tdtdAdvmean'][0:181]/10**12 
    dataTherm = regionstendency[i]['sie_meanMarch'][0:-1]/10**12 + regionstendency[i]['tdmean'][0:181]/10**12 + regionstendency[i]['ttmean'][0:181]/10**12
    
    # Plot SIA march and sept per region
    axs[i].plot(dates[0:180],regions[i]['sia_mean'][0:180]/10**12, color='black', linewidth=2)
    axs[i].plot(dates[0:180],regions[i]['sie_mean_march'][0:180]/10**12, color='black',linewidth=2)

    # Plot SIA march + adv loss
    axs[i].plot(dates[0:180],dataDyn[0:180], label='March SIA + Adv.', color=colorDyn, linewidth=2)
    axs[i].fill_between(dates[0:180], regions[i]['sie_mean_march'][0:180]/10**12, dataDyn[0:180], where=(dataDyn[0:180] < regionstendency[i]['sie_meanMarch'][0:180]/10**12), interpolate=True, color=colorDyn, alpha=0.6)
    #+ Ridge
    axs[i].plot(dates[0:180],dataDRidge[0:180], label='March SIA + Adv. + Ridging.',color=colorRidge, linewidth=2)
    axs[i].fill_between(dates[0:180], dataDyn[0:180], dataDRidge[0:180], where=(dataDRidge[0:180] < dataDyn[0:180]), interpolate=True, color=colorRidge, alpha=0.6)
    # #+ Thermo
    axs[i].plot(dates[0:180],dataTherm[0:180], label='March SIA + Adv. + Ridging+ Thermo.',color=colorThermo, linewidth=2)
    axs[i].fill_between(dates[0:180], dataDRidge[0:180], dataTherm[0:180], where=(dataTherm[0:180] < (dataDRidge[0:180])), interpolate=True, color=colorThermo, alpha=0.6)

    axs[i].set_xlim(datetime(1920, 1, 1), datetime(2100, 1, 1))
    axs[i].xaxis.set_major_locator(mdates.YearLocator(base=20))  # Adjust the base value to control the tick intervals
    axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the ticks as years
    axs[i].set_xticklabels(labels=years[::20]) 
    axs[i].grid(linewidth=1, color='lightgrey', zorder=0)
    axs[i].axhline(y=0.0, color='grey', linestyle='-', linewidth=2)
    axs[i].set_facecolor('none')
    axs[i].text(0.01, 0.95,region['region'], transform=axs[i].transAxes)
    
    i+=1
    
plt.xlim(1920, 2100)
plt.xlabel('Time',fontsize=14)
axs[0].set_ylim(-0.20,1.4)
axs[1].set_ylim(-0.05,0.2)
axs[2].set_ylim(-0.1,0.7)
plt.xlim(datetime(1920, 1, 1), datetime(2100, 1, 1))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=20)) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.text(0.04, 0.5, 'SIA ($10^6 km^2$)', va='center', rotation='vertical')
plt.legend()
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)

plt.show()


