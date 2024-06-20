#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from scipy.stats import linregress
from scipy import stats
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedFormatter

# %% Functions
def translateMonthsToWords(monthInNumber):
    datesDict = { 'Jan': 1, 'Feb': 2, 'March': 3, 'Mar': 3, 'Apr':4, 'April': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12 } 
    return datesDict[monthInNumber]

def translateMonthsStringsToWords(monthInNumber):
    datesDict = {  '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7, '08': 8, '09': 9, '10': 10, '11': 11, '12': 12 } 
    return datesDict[monthInNumber]

def numberOfDaysPerMonth(month):
    thirdyDaysMonths = [4,6,9,11]
    thirdyOneDaysMonths = [1,3,5,7,8,10,12]
    if month in thirdyDaysMonths:
        return 30
    if month in thirdyOneDaysMonths:
        return 31
    return 28

# Linear regression of scatter plot
def computeLinRegressionOfScatterPlotData(x, y):
    x = np.array(x)
    y = np.array(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
    r_squared = r_value**2
    return slope, intercept, r_squared

# %% OBSERVATIONS
# Open excel file and put each sheet in a dict
howell_sheet_all = pd.read_excel('/aos/home/mfol/Data/Observations/Howell 1997_2022/RADARSAT-Iceflux_1997-2022-forMF.xlsx', sheet_name=None)

# Create dict with one object per gate, monthly, yearly  fluxes
gate_dict_obs = {}

for gate_name, df in howell_sheet_all.items():
    months = df.columns[1:]

    gate_dict_obs[gate_name] = {}

    for _, row in df.iterrows():
        year = int(row.iloc[0])
        if year not in gate_dict_obs[gate_name]:
            gate_dict_obs[gate_name][year] = {'months': [], 'fluxes': [], 'annualFlux': []}

        for month, flux in zip(months, row[1:]):
            gate_dict_obs[gate_name][year]['months'].append(translateMonthsToWords(month))
            gate_dict_obs[gate_name][year]['fluxes'].append(flux/10**3)

for gate_name, gate_data in gate_dict_obs.items():
    for year, data in gate_data.items():
        gate_dict_obs[gate_name][year]['annualFlux'] = np.nansum(gate_dict_obs[gate_name][year]['fluxes'])

# Combine south and mid 
gate_dict_obs['QEI-South-Mid'] = {}

average = {}
for year, data in gate_dict_obs['QEI-South'].items():
    averageYear = []
    sorted_months = []
    gate_dict_obs['QEI-South-Mid'][year] = {}
    i = 0
    for month in data['months']:
        sorted_months.append(month)
        monthData = [
            gate_dict_obs['QEI-South'][year]['fluxes'][i],
            gate_dict_obs['QEI-Mid'][year]['fluxes'][i]
        ]
        averageYear.append(np.sum(monthData))
        i +=1
    gate_dict_obs['QEI-South-Mid'][year]['fluxes'] = averageYear
    gate_dict_obs['QEI-South-Mid'][year]['months'] = sorted_months
    gate_dict_obs['QEI-South-Mid'][year]['annualFlux'] = np.nansum(averageYear)


# Howell Jones and Lancaster
howell_sheet_JandL = pd.read_excel('/aos/home/mfol/Data/Observations/Howell 1997_2022/Lancaster_Sound__and_Jones_Sound_Howell.xlsx', sheet_name=None)

for gate_name, df in howell_sheet_JandL.items():
    months = df.columns[1:]

    gate_dict_obs[gate_name] = {}

    for _, row in df.iterrows():
        year = int(row.iloc[0])
        if year not in gate_dict_obs[gate_name]:
            gate_dict_obs[gate_name][year] = {'months': [], 'fluxes': [], 'annualFlux': []}

        for month, flux in zip(months, row[1:]):
            gate_dict_obs[gate_name][year]['months'].append(translateMonthsToWords(month))
            gate_dict_obs[gate_name][year]['fluxes'].append(flux/10**3)

for gate_name, gate_data in gate_dict_obs.items():
    for year, data in gate_data.items():
        gate_dict_obs[gate_name][year]['annualFlux'] = np.nansum(gate_dict_obs[gate_name][year]['fluxes'])

# KWOK
kwok_sheet_all = pd.read_excel(r"/aos/home/mfol/Data/Observations/Kwok2006/Kwok2006.xlsx", sheet_name=None)
kwok_sheet_all2 = pd.read_excel(r"/aos/home/mfol/Data/Observations/Smerdrud/Smerdrud2016.xlsx", sheet_name=None)
kwok = [kwok_sheet_all, kwok_sheet_all2]
gate_dict_obs_kwok = {}
for elem in kwok:

    for gate_name, df in elem.items():
        months = df.columns[1:]

        gate_dict_obs_kwok[gate_name] = {}

        for _, row in df.iterrows():
            year = int(row.iloc[0])
            if year not in gate_dict_obs_kwok[gate_name]:
                gate_dict_obs_kwok[gate_name][year] = {'months': [], 'fluxes': [], 'annualFlux': []}

            for month, flux in zip(months, row[1:]):
                if gate_name == 'Fram':
                    gate_dict_obs_kwok[gate_name][year]['fluxes'].append(flux)
                else: 
                    gate_dict_obs_kwok[gate_name][year]['fluxes'].append(flux/10**3)
                gate_dict_obs_kwok[gate_name][year]['months'].append(month)

    for gate_name, gate_data in gate_dict_obs_kwok.items():
        # print(gate_name)
        for year, data in gate_data.items():
            gate_dict_obs_kwok[gate_name][year]['annualFlux'] = np.nansum(gate_dict_obs_kwok[gate_name][year]['fluxes'])

# Kwok Nares annual budget
kwok_sheet_all = pd.read_excel(r"/aos/home/mfol/Data/Observations/Kwok2010/data.xlsx", sheet_name=None)
for gate_name, df in kwok_sheet_all.items():
    gate_dict_obs_kwok['Nares'] = {}
    for _, row in df.iterrows():
        year = int(row.iloc[0])
        gate_dict_obs_kwok['Nares'][year] = {'months': [], 'fluxes': [], 'annualFlux': []}
        gate_dict_obs_kwok['Nares'][year]['annualFlux'] = -row[1]


# %% RESULTS
ensembles = ['EM3','EM1'] 
filesMeltSeasonQEI = ['/aos/home/mfol/Results/CESM_HR_meltSeason/CESM_HR_meltSeasonE1QEIshort.xlsx'
            ,'/aos/home/mfol/Results/CESM_HR_meltSeason/CESM_HR_meltSeasonE3QEIshort.xlsx'
            ]

# Create dict with one object per gate, monthly, yearly and melt season fluxes
allEnsemblResults = {}
j = 0
for em in ensembles:
    cesmHR_fluxes = pd.read_excel('/aos/home/mfol/Results/IHESP/ice_flux_iHESP_'+em+'.xlsx', sheet_name=None)
    cesmHR_fluxes = cesmHR_fluxes['Sheet1']
    meltSeasonDefQEI = pd.read_excel(filesMeltSeasonQEI[j],  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime)

    # Create an empty dictionary to store the gate information
    gate_dict = {}

    # Iterate over each row in the DataFrame
    for _, row in cesmHR_fluxes.iterrows():
        year_month = row['Date']
        year, month = year_month.split('-')
        year = int(year)

        filtered_df = meltSeasonDefQEI[meltSeasonDefQEI['Date'].dt.year == year]
        start = filtered_df['meltStart'].values[0]
        end = filtered_df['freezeStart'].values[0]
        
        numberOfDaysInMonth = numberOfDaysPerMonth(translateMonthsStringsToWords(month))

        for col in cesmHR_fluxes.columns[2:-1]:
            gate_name = col
            
            if gate_name not in gate_dict:
                gate_dict[gate_name] = {}
            
            if year not in gate_dict[gate_name]:
                gate_dict[gate_name][year] = {'months': [], 'fluxes': [], 'annualFlux': [], 'meltSeasonFluxes': []}

            gate_dict[gate_name][year]['months'].append(translateMonthsStringsToWords(month))
            # m/s to monthly flux in 10^3km2
            gate_dict[gate_name][year]['fluxes'].append((row[col]*24*3600*10**-3)*numberOfDaysPerMonth(translateMonthsStringsToWords(month))/(10**6))
            
            if (not np.isnan(start) ) and ( start != 0):
                monthInInt = translateMonthsStringsToWords(month)
                if monthInInt > (int(start)) and monthInInt < (int(end)):
                    gate_dict[gate_name][year]['meltSeasonFluxes'].append((row[col]*24*3600*10**-3)*numberOfDaysPerMonth(translateMonthsStringsToWords(month))/(10**6))
                if monthInInt == (int(start)):
                    ndays= numberOfDaysPerMonth(translateMonthsStringsToWords(month))
                    nbDaysMonth = (1-(start - int(start))) * ndays
                    gate_dict[gate_name][year]['meltSeasonFluxes'].append((row[col]*24*3600*10**-3)*nbDaysMonth/(10**6))
                if monthInInt == (int(end)):
                    ndays= numberOfDaysPerMonth(translateMonthsStringsToWords(month))
                    nbDaysMonth = (end - int(end)) * ndays
                    gate_dict[gate_name][year]['meltSeasonFluxes'].append((row[col]*24*3600*10**-3)*nbDaysMonth/(10**6))
                
    # Sort the data
    for gate_name, gate_data in gate_dict.items():
        for year, data in gate_data.items():
            copyFluxes = gate_dict[gate_name][year]['fluxes']
            months = gate_dict[gate_name][year]['months']
            sortedData = sorted(zip(months, copyFluxes))
            sorted_months, sorted_fluxes = zip(*sortedData)
            gate_dict[gate_name][year]['months'] = sorted_months
            gate_dict[gate_name][year]['fluxes'] = sorted_fluxes
            
            gate_dict[gate_name][year]['annualFlux'] = np.nansum(gate_dict[gate_name][year]['fluxes'])
            gate_dict[gate_name][year]['annualFluxMeltSeason'] = np.nansum(gate_dict[gate_name][year]['meltSeasonFluxes'])
            
    # RESULTS CREATE QEI GATE (SUM)
    QEIGates = ['Ballantyne', 'Wilkins', 'Gust', 'Peary', 'Sverdrup', 'Eureka']
    gate_dict['QEI-in'] = {}
    gate_dict['QEI-South'] = {} # Ball, Wilk, Pr Gust.
    gate_dict['QEI-North'] = {} # Peary, Sv

    average = {}
    for year, data in gate_dict['Ballantyne'].items():
        averageYear = []
        averageYearSouth = []
        averageYearNorth = []
        averageYearMeltSeason = []
        sorted_months = []
        gate_dict['QEI-in'][year] = {}
        gate_dict['QEI-South'][year] = {}
        gate_dict['QEI-North'][year] = {}
        i = 0
        for month in data['months']:
            sorted_months.append(month)
            monthData = [
                gate_dict['Ballantyne'][year]['fluxes'][i],
                gate_dict['Wilkins'][year]['fluxes'][i],
                gate_dict['Gust'][year]['fluxes'][i],
                gate_dict['Peary'][year]['fluxes'][i],
                gate_dict['Sverdrup'][year]['fluxes'][i],
                gate_dict['Eureka'][year]['fluxes'][i]
            ]
            averageYear.append(np.nansum(monthData))
            averageYearSouth.append(np.nansum([
                gate_dict['Ballantyne'][year]['fluxes'][i],
                gate_dict['Wilkins'][year]['fluxes'][i],
                gate_dict['Gust'][year]['fluxes'][i]]))
            averageYearNorth.append(np.nansum([
                gate_dict['Peary'][year]['fluxes'][i],
                gate_dict['Sverdrup'][year]['fluxes'][i]]))
            i +=1
        gate_dict['QEI-in'][year]['fluxes'] = averageYear
        gate_dict['QEI-in'][year]['months'] = sorted_months
        gate_dict['QEI-in'][year]['annualFlux'] = np.nansum(averageYear)
        gate_dict['QEI-in'][year]['annualFluxMeltSeason'] = np.nansum([gate_dict['Ballantyne'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Wilkins'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Gust'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Peary'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Sverdrup'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Eureka'][year]['annualFluxMeltSeason']])

        gate_dict['QEI-South'][year]['fluxes'] = averageYearSouth
        gate_dict['QEI-South'][year]['months'] = sorted_months
        gate_dict['QEI-South'][year]['annualFlux'] = np.nansum(averageYearSouth)
        gate_dict['QEI-North'][year]['fluxes'] = averageYearNorth
        gate_dict['QEI-North'][year]['months'] = sorted_months
        gate_dict['QEI-North'][year]['annualFlux'] = np.nansum(averageYearNorth)


    # QEI OUT - Hell gate not resolved
    QEIGates = ['Fitzwilliam', 'BMC', 'Cardigan', 'Penny']
    gate_dict['QEI-out'] = {}
    average = {}
    for year, data in gate_dict['Fitzwilliam'].items():
        averageYear = []
        sorted_months = []
        gate_dict['QEI-out'][year] = {}
        i = 0
        for month in data['months']:
            sorted_months.append(month)
            monthData = [
                gate_dict['Fitzwilliam'][year]['fluxes'][i],
                gate_dict['BMC'][year]['fluxes'][i],
                gate_dict['Cardigan'][year]['fluxes'][i],
                gate_dict['Penny'][year]['fluxes'][i]
            ]
            averageYear.append(np.nansum(monthData))
            i +=1
        gate_dict['QEI-out'][year]['fluxes'] = averageYear
        gate_dict['QEI-out'][year]['months'] = sorted_months
        gate_dict['QEI-out'][year]['annualFlux'] = np.nansum(averageYear)
        gate_dict['QEI-out'][year]['annualFluxMeltSeason'] = np.nansum([gate_dict['Fitzwilliam'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['BMC'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Cardigan'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Penny'][year]['annualFluxMeltSeason']])
    # QEI In-Out - Hell gate not resolved
    QEIGates = ['Ballantyne', 'Wilkins', 'Gust', 'Peary', 'Sverdrup', 'Eureka','Fitzwilliam', 'BMC', 'Cardigan', 'Penny']
    gate_dict['QEI'] = {}
    average = {}
    for year, data in gate_dict['Fitzwilliam'].items():
        averageYear = []
        sorted_months = []
        gate_dict['QEI'][year] = {}
        i = 0
        for month in data['months']:
            sorted_months.append(month)
            monthData = [
                gate_dict['Ballantyne'][year]['fluxes'][i],
                gate_dict['Wilkins'][year]['fluxes'][i],
                gate_dict['Gust'][year]['fluxes'][i],
                gate_dict['Peary'][year]['fluxes'][i],
                gate_dict['Sverdrup'][year]['fluxes'][i],
                gate_dict['Eureka'][year]['fluxes'][i],
                gate_dict['Fitzwilliam'][year]['fluxes'][i],
                gate_dict['BMC'][year]['fluxes'][i],
                gate_dict['Cardigan'][year]['fluxes'][i],
                gate_dict['Penny'][year]['fluxes'][i],
            ]
            averageYear.append(np.nansum(monthData))
            i +=1
        gate_dict['QEI'][year]['fluxes'] = averageYear
        gate_dict['QEI'][year]['months'] = sorted_months
        gate_dict['QEI'][year]['annualFlux'] = np.nansum(averageYear)
        gate_dict['QEI'][year]['annualFluxMeltSeason'] = np.nansum([gate_dict['Fitzwilliam'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['BMC'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Cardigan'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Penny'][year]['annualFluxMeltSeason'],gate_dict['Ballantyne'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Wilkins'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Gust'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Peary'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Sverdrup'][year]['annualFluxMeltSeason'],
                                                                       gate_dict['Eureka'][year]['annualFluxMeltSeason']])
    
    allEnsemblResults[em] = gate_dict
    j +=1


# %% Compute seasonal cycle

# Compute seasonal cycle average for the stated gates
periods = [(1921, 1980), (1981, 2000) ,(2017, 2021), (2000, 2014), (2001, 2020), (2021, 2040), (2041, 2060),(2061, 2080), (2081, 2100)]
gates = [ 'QEI-in','QEI-out', 'QEI','Nares', "Mclure",'Lancaster','Admundsen','Fram']
seasonal_cycle = {str(period): {gate: {} for gate in gates} for period in periods}


# Iterate through the specified periods
for period in periods:
    start_year, end_year = period
    print(start_year)
    print(end_year)

    monthly_values = {gate: [[] for _ in range(12)] for gate in gates}

    for gate in gates:
        for year in range(start_year, end_year + 1):
            for em in ensembles:
                fluxesForYear = allEnsemblResults[em][gate][year]['fluxes']
                for month in range(12):
                    monthly_values[gate][month].append(fluxesForYear[month])

        mean_flux = []
        min_flux=[]
        max_flux=[]
        for month in range(12):
            monthly_data = monthly_values[gate][month]

            mean_flux.append(sum(monthly_data) / len(monthly_data))
            min_flux.append(min(monthly_data))
            max_flux.append(max(monthly_data))
        seasonal_cycle[str(period)][gate] = {
            'mean': mean_flux,
            'min': min_flux,
            'max': max_flux
        }

# %% Compute seasonal cycle for observations. Each dataset have different observed periods.
# Create dict to store observed seasonal cycle
seasonal_cycle_obs = {str((2001, 2020)): {'QEI-in': {} },str((2017, 2021)): {'QEI-Out': {} },str((2017, 2021)): {'Nares': {} }, str((2017, 2021)): {'Lancaster': {} },str((2001, 2020)): {'Mclure': {} },str((2000, 2014)): {'Fram': {} },str((1997, 2002)): {'Admundsen': {} },str((1997, 2002)): {'Mclure': {} }}

# QEI-in and McLure
periods = [(2001, 2020)]
for period in periods:
    start_year, end_year = period

    monthly_values = {'QEI-in': [[] for _ in range(12)]}
    monthly_valuesMclure = {'Mclure': [[] for _ in range(12)]}

    for year in range(start_year, end_year + 1):
        fluxesForYear = []
        fluxesForYearMclure = []
        for i in range(12):
            fluxesForYear.append(gate_dict_obs['QEI-South-Mid'][year]['fluxes'][i] + gate_dict_obs['QEI-North'][year]['fluxes'][i])
            fluxesForYearMclure.append(gate_dict_obs['Mclure'][year]['fluxes'][i])
            
        for month in range(12):
            monthly_values['QEI-in'][month].append(fluxesForYear[month])
            monthly_valuesMclure['Mclure'][month].append(fluxesForYearMclure[month])
            
    mean_flux = []
    min_flux=[]
    max_flux=[]
    mean_fluxMclure = []
    min_fluxMclure=[]
    max_fluxMclure=[]
    for month in range(12):
        monthly_data = monthly_values['QEI-in'][month]
        monthly_dataMclure = monthly_valuesMclure['Mclure'][month]

        mean_flux.append(sum(monthly_data) / len(monthly_data))
        min_flux.append(min(monthly_data))
        max_flux.append(max(monthly_data))

        mean_fluxMclure.append(sum(monthly_dataMclure) / len(monthly_dataMclure))
        min_fluxMclure.append(min(monthly_dataMclure))
        max_fluxMclure.append(max(monthly_dataMclure))

    seasonal_cycle_obs[str(period)]['QEI-in'] = {
        'mean': mean_flux,
        'min': min_flux,
        'max': max_flux
    }
    seasonal_cycle_obs[str(period)]['Mclure'] = {
        'mean': mean_fluxMclure,
        'min': min_fluxMclure,
        'max': max_fluxMclure
    }

# Nares, Lancaster, QEI out and Amundsen
periods = [(2017, 2021)]
for period in periods:
    start_year, end_year = period

    monthly_values = {'Nares': [[] for _ in range(12)]}
    monthly_valuesLanc = {'Lancaster': [[] for _ in range(12)]}
    monthly_valuesAmund = {'Amundsen': [[] for _ in range(12)]}
    monthly_valuesQEIOut = {'QEI-Out': [[] for _ in range(12)]}

    for year in range(start_year, end_year + 1):
        fluxesForYear = []
        fluxesForYearLanc = []
        fluxesForYearQEIOut= []
        fluxesForYearAmund=[]
        fluxesForYear = gate_dict_obs['Nares'][year]['fluxes']
        fluxesForYearLanc = gate_dict_obs['Lancaster'][year]['fluxes']
        fluxesForYearAmund = gate_dict_obs['Amundsen'][year]['fluxes']
        fluxesForYearQEIOut = gate_dict_obs['QEI-Out'][year]['fluxes']
        for month in range(12):
            monthly_values['Nares'][month].append(fluxesForYear[month])
            monthly_valuesLanc['Lancaster'][month].append(fluxesForYearLanc[month])
            monthly_valuesAmund['Amundsen'][month].append(fluxesForYearAmund[month])
            monthly_valuesQEIOut['QEI-Out'][month].append(fluxesForYearQEIOut[month])

    mean_flux = []
    min_flux=[]
    max_flux=[]
    mean_fluxLanc = []
    min_fluxLanc=[]
    max_fluxLanc=[]
    mean_fluxAmund = []
    min_fluxAmund =[]
    max_fluxAmund =[]
    mean_fluxQEIOut = []
    min_fluxQEIOut=[]
    max_fluxQEIOut=[]
    for month in range(12):
        monthly_data = monthly_values['Nares'][month]
        monthly_dataLanc = monthly_valuesLanc['Lancaster'][month]
        monthly_dataAmund= monthly_valuesAmund['Amundsen'][month]
        monthly_dataQEIOut = monthly_valuesQEIOut['QEI-Out'][month]

        mean_flux.append(sum(monthly_data) / len(monthly_data))
        min_flux.append(min(monthly_data))
        max_flux.append(max(monthly_data))

        mean_fluxLanc.append(sum(monthly_dataLanc) / len(monthly_dataLanc))
        min_fluxLanc.append(min(monthly_dataLanc))
        max_fluxLanc.append(max(monthly_dataLanc))

        mean_fluxAmund.append(sum(monthly_dataAmund) / len(monthly_dataAmund))
        min_fluxAmund.append(min(monthly_dataAmund))
        max_fluxAmund.append(max(monthly_dataAmund))

        mean_fluxQEIOut.append(sum(monthly_dataQEIOut) / len(monthly_dataQEIOut))
        min_fluxQEIOut.append(min(monthly_dataQEIOut))
        max_fluxQEIOut.append(max(monthly_dataQEIOut))
    seasonal_cycle_obs[str(period)]['Nares'] = {
        'mean': mean_flux,
        'min': min_flux,
        'max': max_flux
    }
    seasonal_cycle_obs[str(period)]['Lancaster'] = {
        'mean': mean_fluxLanc,
        'min': min_fluxLanc,
        'max': max_fluxLanc
    }
    seasonal_cycle_obs[str(period)]['Amundsen'] = {
        'mean': mean_fluxAmund,
        'min': min_fluxAmund,
        'max': max_fluxAmund
    }
    seasonal_cycle_obs[str(period)]['QEI-Out'] = {
        'mean': mean_fluxQEIOut,
        'min': min_fluxQEIOut,
        'max': max_fluxQEIOut
    }

# Fram Strait
periods = [(2000, 2014)]
for period in periods:
    start_year, end_year = period

    monthly_values = {'Fram': [[] for _ in range(12)]}

    for year in range(start_year, end_year + 1):
        fluxesForYear = []
        fluxesForYear = gate_dict_obs_kwok['Fram'][year]['fluxes']
        for month in range(12):
            monthly_values['Fram'][month].append(fluxesForYear[month])

    mean_flux = []
    min_flux=[]
    max_flux=[]
    for month in range(12):
        monthly_data = monthly_values['Fram'][month]

        mean_flux.append(sum(monthly_data) / len(~np.isnan(monthly_data)))
        min_flux.append(min(monthly_data))
        max_flux.append(max(monthly_data))
    seasonal_cycle_obs[str(period)]['Fram'] = {
        'mean': mean_flux,
        'min': min_flux,
        'max': max_flux
    }

# Amundsen, McLure, QEI-In
periods = [(1997, 2002)]
for period in periods:
    start_year, end_year = period

    monthly_values = {'Admundsen': [[] for _ in range(12)]}
    monthly_valuesLanc = {'Mclure': [[] for _ in range(12)]}
    monthly_valuesQEI = {'QEI-In': [[] for _ in range(12)]}

    for year in range(start_year, end_year + 1):
        fluxesForYear = []
        fluxesForYearLanc = []
        fluxesForYear = gate_dict_obs_kwok['Admundsen'][year]['fluxes']
        fluxesForYearLanc = gate_dict_obs_kwok['McLure'][year]['fluxes']
        fluxesForYearQEI = gate_dict_obs_kwok['QEI-N'][year]['fluxes'] + gate_dict_obs_kwok['QEI-S'][year]['fluxes']
        for month in range(12):
            monthly_values['Admundsen'][month].append(fluxesForYear[month])
            monthly_valuesLanc['Mclure'][month].append(fluxesForYearLanc[month])
            monthly_valuesQEI['QEI-In'][month].append(fluxesForYearQEI[month])

    mean_flux = []
    min_flux=[]
    max_flux=[]
    mean_fluxLanc = []
    min_fluxLanc=[]
    max_fluxLanc=[]
    mean_fluxQEI = []
    min_fluxQEI=[]
    max_fluxQEI=[]
    for month in range(12):
        monthly_data = monthly_values['Admundsen'][month]
        monthly_dataLanc = monthly_valuesLanc['Mclure'][month]
        monthly_dataQEI= monthly_valuesQEI['QEI-In'][month]
        mean_flux.append(np.nansum(monthly_data) / len(~np.isnan(monthly_data)))
        min_flux.append(min(monthly_data))
        max_flux.append(max(monthly_data))

        mean_fluxLanc.append(np.nansum(monthly_dataLanc) / len(~np.isnan(monthly_data)))
        min_fluxLanc.append(min(monthly_dataLanc))
        max_fluxLanc.append(max(monthly_dataLanc))

        mean_fluxQEI.append(np.nansum(monthly_dataQEI) / len(~np.isnan(monthly_dataQEI)))
        min_fluxQEI.append(min(monthly_dataQEI))
        max_fluxQEI.append(max(monthly_dataQEI))
    seasonal_cycle_obs[str(period)]['Admundsen'] = {
        'mean': mean_flux,
        'min': min_flux,
        'max': max_flux
    }
    seasonal_cycle_obs[str(period)]['Mclure'] = {
        'mean': mean_fluxLanc,
        'min': min_fluxLanc,
        'max': max_fluxLanc
    }
    seasonal_cycle_obs[str(period)]['QEI-In'] = {
        'mean': mean_fluxQEI,
        'min': min_fluxQEI,
        'max': max_fluxQEI
    }

# %% 
# Reverse QEI out to have positive fluxes
for i, gate in enumerate(['QEI-out']):
    for j,period in enumerate([(1921, 1980), (1981, 2000) ,(2017, 2021), (2000, 2014), (2001, 2020), (2021, 2040), (2041, 2060),(2061, 2080), (2081, 2100)]):
        for i, elem in enumerate(seasonal_cycle[str(period)][gate]['mean']):
            seasonal_cycle[str(period)][gate]['mean'][i] = -elem
# %%
# Reverse Lancaster to have positive fluxes
for i, gate in enumerate(['Lancaster']):
    for j,period in enumerate([(1921, 1980), (1981, 2000) ,(2017, 2021), (2000, 2014), (2001, 2020), (2021, 2040), (2041, 2060),(2061, 2080), (2081, 2100)]):
        for i, elem in enumerate(seasonal_cycle[str(period)][gate]['mean']):
            seasonal_cycle[str(period)][gate]['mean'][i] = -elem

# %% PLOT SEASONAL CYCLE - Figure 8

plt.rc('font', size=16)
colors=[  'dimgrey','#9CCB86','#EE7600','#CF597E',]
colors=['grey','#009392','#9CCB86','#C3D791','#EEB479','#E88471','#CF597E']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
months_short = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

fig, axs = plt.subplots(4,2, figsize=(16,16))
axs = axs.flatten()

# plot observations
obs1 = axs[0].plot(months, seasonal_cycle_obs[str((2001, 2020))]['QEI-in']['mean'], linewidth=4,c='#9CCB86',linestyle='dashed', label='Obs')       
obs1 = axs[0].plot(months, seasonal_cycle_obs[str((1997, 2002))]['QEI-In']['mean'], linewidth=4,c='#9CCB86',linestyle='dotted', label='Obs')       
obs4 = axs[1].plot(months, seasonal_cycle_obs[str((2017, 2021))]['QEI-Out']['mean'], linewidth=4,c='#9CCB86',linestyle='dashed', label='Obs')       
obs2 = axs[2].plot(months, seasonal_cycle_obs[str((2017, 2021))]['Nares']['mean'], linewidth=4,c='#9CCB86',linestyle='dashed', label='Obs')       
obs1 = axs[3].plot(months, seasonal_cycle_obs[str((2001, 2020))]['Mclure']['mean'], linewidth=4,c='#9CCB86',linestyle='dashed', label='Obs')       
obs2 = axs[4].plot(months, -np.array(seasonal_cycle_obs[str((2017, 2021))]['Lancaster']['mean']), linewidth=4,c='#9CCB86',linestyle='dashed', label='Obs')       
obs3 = axs[6].plot(months, seasonal_cycle_obs[str((2000, 2014))]['Fram']['mean'], linewidth=4,c='#9CCB86', linestyle=(0,(3,1,1,1)), label='Obs')      
axs[3].plot(months, seasonal_cycle_obs[str((1997, 2002))]['Mclure']['mean'], linewidth=4,c='#9CCB86', linestyle='dotted', label='Obs')      
axs[5].plot(months, seasonal_cycle_obs[str((1997, 2002))]['Admundsen']['mean'], linewidth=4,c='#9CCB86', linestyle='dotted', label='Obs')      
axs[5].plot(months, seasonal_cycle_obs[str((2017, 2021))]['Amundsen']['mean'], linewidth=4,c='#9CCB86', linestyle='dashed', label='Obs')      

# plot CESM1.3-HR seasonal cycles
for i, gate in enumerate([ 'QEI-in','QEI-out', 'Nares','Mclure', 'Lancaster','Admundsen','Fram']):
    if gate == 'QEI-in' or gate =='Mclure':
        for j,period in enumerate([(1921, 1980), (1981, 2000), (2001, 2020), (2021, 2040), (2041, 2060),(2061, 2080), (2081, 2100)]):
            pl = axs[i].plot(months, seasonal_cycle[str(period)][gate]['mean'], linewidth=2,c=colors[j], label=str(period))
    elif gate == 'Nares' or gate =='Lancaster' or gate== 'QEI-out' or gate == 'Admundsen':
        for j,period in enumerate([(1921, 1980), (1981, 2000) ,(2017, 2021), (2021, 2040), (2041, 2060),(2061, 2080), (2081, 2100)]):
            pl = axs[i].plot(months, seasonal_cycle[str(period)][gate]['mean'], linewidth=2,c=colors[j], label=str(period))
    elif gate == 'Fram':
        for j,period in enumerate([(1921, 1980), (1981, 2000) ,(2000, 2014), (2021, 2040), (2041, 2060),(2061, 2080), (2081, 2100)]):
            pl = axs[i].plot(months, seasonal_cycle[str(period)][gate]['mean'], linewidth=2,c=colors[j], label=str(period))
    
    axs[i].set_xlim(xmin='Jan', xmax='Dec')
    axs[i].grid(color='lightgrey')


axs[0].text(0.01, 0.9, 'a) QEI in', transform=axs[0].transAxes,fontsize=14)
axs[1].text(0.01, 0.9, 'b) QEI out', transform=axs[1].transAxes,fontsize=14)
axs[2].text(0.01, 0.9, 'c) Nares',transform=axs[2].transAxes, fontsize=14)
axs[3].text(0.01, 0.9, 'd) McLure',transform=axs[3].transAxes, fontsize=14)
axs[4].text(0.01, 0.9, 'e) Lancaster',transform=axs[4].transAxes, fontsize=14)
axs[5].text(0.01, 0.9, 'f) Amundsen',transform=axs[5].transAxes, fontsize=14)
axs[6].text(0.01, 0.9, 'g) Fram',transform=axs[6].transAxes, fontsize=14)

fig.text(0.06, 0.5, 'Monthly SIA export ($10^3$ km$^2$)', va='center', rotation='vertical')

axs[0].set_yticks([0, 10, 20, 30])
axs[1].set_yticks([0, 10, 20, 30])
axs[2].set_yticks([0, 10, 20, 30])
axs[3].set_yticks([-10,0, 10, 20, 30])
axs[4].set_yticks([0, 20, 40, 60])
axs[5].set_yticks([-20, -10,0,10,20])
axs[6].set_yticks([160,120,80,40,0])
axs[6].set_xticks(months,labels=months_short)
axs[5].set_xticks(months,labels=months_short)
axs[0].set_xticks(months,labels=['','','','','','','','','','','',''])
axs[1].set_xticks(months,labels=['','','','','','','','','','','',''])
axs[2].set_xticks(months,labels=['','','','','','','','','','','',''])
axs[3].set_xticks(months,labels=['','','','','','','','','','','',''])
axs[4].set_xticks(months,labels=['','','','','','','','','','','',''])

fig.subplots_adjust(bottom=0.1)
handlesSmed, labelsSmed = axs[6].get_legend_handles_labels()
handles, labels = axs[3].get_legend_handles_labels()
bottom_right_ax = axs[7]
bottom_right_ax.clear() 
bottom_right_ax.set_axis_off() 
handles.insert(2,handlesSmed[0])
bottom_right_ax.legend(handles=handles,labels=['Obs. Howell','Obs. Kwok','Obs. Smedsrud',(1921, 1980), (1981, 2000), '(2001, 2020) for comparison', (2021, 2040), (2041, 2060),(2061, 2080), (2081, 2100)], loc='center',ncol=2)

plt.show()


# %% Plot annual SIA fluxes  QEI - Figure 9
#OBS Howell
in_howell = []
nares_howell = []
out_howell = []
#OBS Kwok}
in_kwok = []
nares_kwok = []
# MODEL
model = {}


plt.rcParams['font.size'] = 16
years = list(range(1920, 2101))
yearsObsHowell =[1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
yearsObsHowellNares =[2017,2018,2019,2020,2021]
yearsObsKowk = [1997,1998,1999,2000,2001,2002]

# MODEL
in_data = []
in_min = []
in_max = []
out_data = []
out_min = []
out_max = []
diff = []
nares_data = []
loss= []
# Compute EM mean
for year, yeardata in allEnsemblResults['EM3']['QEI-in'].items():
    if year in years:
        em1 = allEnsemblResults['EM1']['QEI-in'][year]['annualFlux']
        em3 = yeardata['annualFlux']
        in_data.append((em1 + em3)/2)
        in_min.append(np.min([em1,em3]))
        in_max.append(np.max([em1,em3]))
for year, yeardata in allEnsemblResults['EM3']['Nares'].items():
    if year in years:
        em1 = allEnsemblResults['EM1']['Nares'][year]['annualFlux']
        em3 = yeardata['annualFlux']
        nares_data.append((em1 + em3)/2)
for year, yeardata in allEnsemblResults['EM3']['QEI-out'].items():
    if year in years:
        em1 = allEnsemblResults['EM1']['QEI-out'][year]['annualFlux']
        em3 = yeardata['annualFlux']
        out_data.append((em1 + em3)/2)
        out_min.append(np.min([em1,em3]))
        out_max.append(np.max([em1,em3]))
        
for i in range(len(out_data)):
    diff.append((in_data[i]+out_data[i]))
    loss.append(-out_data[i]+nares_data[i])

# Store data in dict for the model
model = {'in_data':in_data,'out_data':out_data, 'nares_data':nares_data,'diff':diff,'loss':loss}

# Compute Obs
for year in yearsObsHowell:
    in_howell.append(
        gate_dict_obs['QEI-South-Mid'][year]['annualFlux'] + 
        gate_dict_obs['QEI-North'][year]['annualFlux'])
for year in yearsObsHowellNares:
    nares_howell.append(
        gate_dict_obs['Nares'][year]['annualFlux'])
    out_howell.append(gate_dict_obs['QEI-Out'][year]['annualFlux'])


width2Obs = 0.3  # Width of the bars
width = 0.6  # Width of the bars
width_model = 0.3

        
fig, axs = plt.subplots(figsize=(15,10), facecolor='none')
top_vals_nares = np.where(np.array(model['in_data']) < 0, np.array(model['in_data'])/10**3, 0)
top_vals = np.where(top_vals_nares < 0, np.array(top_vals_nares) +( np.array(model['out_data']))/10**3, np.array(model['out_data'])/10**3 )

bars1 = axs.bar(years, np.array(model['in_data'])/10**3, color='#5e4c5f', width=1, zorder=1, label='QEI-in')
axs.bar(np.array(yearsObsHowell), np.array(in_howell)/10**3, width, label='QEI-in Obs.', color='white', edgecolor='#5e4c5f', hatch='////', zorder=2)
bars2 = axs.bar(years, np.array(model['out_data'])/10**3,  bottom=top_vals_nares,color='#999999', label='QEI-out',width=1,zorder=1)
axs.bar(np.array(yearsObsHowellNares), -np.array(out_howell)/10**3, width, label='QEI-out Obs.', color='white',edgecolor='#999999', hatch='////', zorder=2)
bars3 = axs.bar(years, -np.array(model['nares_data'])/10**3,bottom=top_vals, label='Nares', color='#d49720', width=1, zorder=1)
axs.bar(np.array(yearsObsHowellNares), -np.array(nares_howell)/10**3, width, bottom=top_vals[97:102],label='Nares Obs.', color='white',edgecolor='#d49720', hatch='////', zorder=2)

axs.plot(years, np.array(diff)/10**3, color='black', label='Dynamic loss (QEI Out-In)',linewidth=2, zorder=2)

axs.grid( zorder=0, linewidth=1)
axs.set_ylim(ymin=-0.300, ymax=0.30)
axs.yaxis.set_major_formatter(FixedFormatter(['{:,.2f}'.format(val) for val in axs.get_yticks()]))

axs.set_xlim(xmin=1920.5, xmax=2022.5)
axs.set_xticks(years[::20])
axs.set_xticklabels(years[::20], rotation=45) 
axs.set_ylabel('Yearly SIA export ($10^6$ km$^2$)')
axs.set_facecolor('none')
legend = plt.legend(loc='upper left')
legend.legendHandles[0].set_linewidth(2.0)

plt.show()

# %% Plot annual SIA fluxes CAA S - Figure 10

plt.rcParams['font.size'] = 16
years = list(range(1920, 2101))

# Obs Lancaster and McLure
lanc_howell = []
amund_kwok = []
mclure_howell = []
amund_howell = []
for year in yearsObsHowellNares:
    lanc_howell.append(gate_dict_obs['Lancaster'][year]['annualFlux'])
    amund_howell.append(gate_dict_obs['Amundsen'][year]['annualFlux'])   

for year in yearsObsHowell[0:-1]:
    mclure_howell.append(gate_dict_obs['Mclure'][year]['annualFlux']) 

for year in yearsObsKowk:
    amund_kwok.append(gate_dict_obs_kwok['Admundsen'][year]['annualFlux'])

# MODEL
in_data = []
min_qei = []
max_qei = []
mclure = []
min_mclure = []
max_mclure = []
out_data_lancaster = []
min_lanc = []
max_lanc = []
out_data_admund = []
min_amund = []
max_amund = []
diff = []

# Compute em mean per gate
for year, yeardata in allEnsemblResults['EM1']['Fitzwilliam'].items():
    if year in years:
        em1 = allEnsemblResults['EM3']['Fitzwilliam'][year]['annualFlux'] + allEnsemblResults['EM3']['BMC'][year]['annualFlux'] +  allEnsemblResults['EM3']['Penny'][year]['annualFlux']
        em3 = allEnsemblResults['EM1']['Fitzwilliam'][year]['annualFlux'] + allEnsemblResults['EM1']['BMC'][year]['annualFlux'] + allEnsemblResults['EM1']['Penny'][year]['annualFlux']
        in_data.append((em1 + em3)/2)
        min_qei.append(np.min([em1,em3]))
        max_qei.append(np.max([em1,em3]))

for year, yeardata in allEnsemblResults['EM1']['Mclure'].items():
    if year in years:
        em1 = allEnsemblResults['EM3']['Mclure'][year]['annualFlux']
        em3 = yeardata['annualFlux']
        mclure.append((em1 + em3)/2)
        min_mclure.append(np.min([em1,em3]))
        max_mclure.append(np.max([em1,em3]))

for year, yeardata in allEnsemblResults['EM1']['Lancaster'].items():
    if year in years:
        em1 = allEnsemblResults['EM3']['Lancaster'][year]['annualFlux']
        em3 = allEnsemblResults['EM1']['Lancaster'][year]['annualFlux']
        out_data_lancaster.append((em1 + em3)/2)
        min_lanc.append(np.min([em1,em3]))
        max_lanc.append(np.max([em1,em3]))
for year, yeardata in allEnsemblResults['EM1']['Admundsen'].items():
    if year in years:
        em1 = allEnsemblResults['EM3']['Admundsen'][year]['annualFlux']
        em3 = allEnsemblResults['EM1']['Admundsen'][year]['annualFlux']
        out_data_admund.append((em1 + em3)/2)
        min_amund.append(np.min([em1,em3]))
        max_amund.append(np.max([em1,em3]))
        
for i in range(len(out_data)):
    diff.append((-in_data[i]+out_data_admund[i]+ out_data_lancaster[i]+mclure[i]))

model = {'in_data':in_data,'out_data_admund':out_data_admund,'out_data_lancaster':out_data_lancaster, 'mclure':mclure,'diff':diff}

width = 0.6
width_model = 0.3
 
fig, axs = plt.subplots(figsize=(15,10), facecolor='none')
top_vals_mclure = np.where(np.array(model['mclure']) > 0, np.array(model['mclure'])/10**3, 0)
top_vals_lanc = np.where(np.array(model['out_data_admund']) < 0, np.array(model['out_data_admund'])/10**3, 0)
top_vals_in = np.where(np.array(model['out_data_admund']) > 0, np.array(model['out_data_admund'])/10**3, 0)

bars1 = axs.bar(years, -np.array(model['in_data'])/10**3, bottom=top_vals_in+top_vals_mclure, color='#999999', width=1, zorder=1, label='QEI-out')
axs.bar(np.array(yearsObsHowellNares), np.array(out_howell)/10**3, width,bottom=top_vals_in[97:102]+top_vals_mclure[97:102], label='QEI-out Obs.', color='white',edgecolor='#999999', hatch='////', zorder=2)
bars3 = axs.bar(years, np.array(model['mclure'])/10**3,bottom=top_vals_in, label='Mclure', color='#3584cc', width=1, zorder=1)
axs.bar(np.array(yearsObsHowell)[0:-1], np.array(mclure_howell)/10**3, width,bottom=top_vals_in[77:102], label='McLure Obs.', color='white',edgecolor='#3584cc', hatch='////', zorder=2)
bars2 = axs.bar(years, np.array(model['out_data_admund'])/10**3, color='#00b0be', label='Amundsen',width=1,zorder=1)
axs.bar(np.array(yearsObsHowellNares), np.array(amund_howell)/10**3, width, color='white',edgecolor='#00b0be', hatch='////', zorder=2)
axs.bar(np.array(yearsObsKowk), np.array(amund_kwok)/10**3, width, label='Amundsen Obs.', color='white',edgecolor='#00b0be', hatch='////', zorder=2)
bars2 = axs.bar(years, np.array(model['out_data_lancaster'])/10**3,bottom=top_vals_lanc,color='#8fd7d7', label='Lancaster',width=1,zorder=1)
axs.bar(np.array(yearsObsHowellNares), np.array(lanc_howell)/10**3, width, bottom=top_vals_lanc[97:102],label='Lancaster Obs.', color='white',edgecolor='#8fd7d7', hatch='////', zorder=2)

axs.plot(years, np.array(diff)/10**3, color='black', label='Dynamic loss (CAA-S Out-In)',linewidth=2, zorder=2)

axs.grid( zorder=0, linewidth=1)
axs.set_ylim(ymin=-0.500, ymax=0.50)
axs.yaxis.set_major_formatter(FixedFormatter(['{:,.2f}'.format(val) for val in axs.get_yticks()]))

axs.set_xlim(xmin=1920.5, xmax=2022.5)
axs.set_xticks(years[::20])
axs.set_xticklabels(years[::20], rotation=45) 
axs.set_ylabel('Yearly SIA export ($10^6$ km$^2$)')
axs.set_facecolor('none')
legend = plt.legend(loc='upper left', ncol=2)
legend.legendHandles[0].set_linewidth(2.0)
legend.legendHandles[1].set_linewidth(10.0)

plt.show()
