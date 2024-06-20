# %% Import packages
import xarray
import os
from ice_flux_algo import getMeasureInMeters, computeGatesLenghtInModel, computeIceFluxesCAAFromGates
from iHESP_helper import create_excel_file, freeDrift, callGatesCESM_HR

# %% Get grid and measures of grid cells

cesm_new = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.200001-200611.nc', decode_times=True)
cesm_grid = xarray.open_dataset(r'/aos/home/mfol/Data/CESM/domain.ocn.tx0.1v2_090218.nc', decode_times=False)
gates = callGatesCESM_HR()
tlon = cesm_grid.xc
tlat = cesm_grid.yc

iceFluxesByMonth = {}
measures = {
    'htn': cesm_new['HTN'],
    'hte': cesm_new['HTE'],
    }

#%% Compute monthly SIA fluxes from 1920 to 2100 for CESM1.3-HR

# Link to needed files
ensemblesData = [
            {'em': 'E1', 'emm': 'EM1', 'main': ['/mnt/qanik/iHESP/EM1/hist/', '/mnt/qanik/iHESP/EM1/proj/'],
                'uvel': ['/mnt/qanik/iHESP/EM1/hist/uvel_hist/','/mnt/qanik/iHESP/EM1/proj/uvel_proj'],
                'vvel': ['/mnt/qanik/iHESP/EM1/hist/vvel_hist/', '/mnt/qanik/iHESP/EM1/proj/vvel_proj'],
                'aice': ['/mnt/qanik/iHESP/EM1/hist/aice_hist/', '/mnt/qanik/iHESP/EM1/proj/aice_proj'], 
                'filePrefix': ['B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.', 'B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.']},

              {'em': 'E3', 'emm': 'EM3', 'main':['/mnt/qanik/iHESP/EM3/hist/', '/mnt/qanik/iHESP/EM3/proj/'],
               'filePrefix': ['b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.','b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.31.003.cice.h.']},
            ]

#  This will create an excel file per ensemble
for em in ensemblesData: 
    iceFluxesByMonth = {}

    if em['em'] == 'E1':
        # For hist or proj
        for i, directory in enumerate(em['uvel']):
            # For all files in hist or proj
            for filename in os.listdir(directory):
                if 'cice.h.' in filename:
                    fuvel = os.path.join(em['uvel'][i], filename)
                    dates = filename.split('.')[-2]
                    fvvel = os.path.join(em['vvel'][i], em['filePrefix'][i]+'vvel.'+dates+'.nc')
                    faice = os.path.join(em['aice'][i], em['filePrefix'][i]+'aice.'+dates+'.nc')
                    
                    if dates not in ['185001-185912', '186001-186912', '187001-187912', '188001-188912', '189001-189912', '190001-190912', '191001-191912']:
                        if os.path.isfile(fuvel):
                            cesm_a_uvel = xarray.open_dataset(fuvel, decode_times=True)
                            cesm_a_vvel = xarray.open_dataset(fvvel, decode_times=True)
                            cesm_a_aice = xarray.open_dataset(faice, decode_times=True)

                            for ti, t in enumerate(cesm_a_uvel['uvel'].time):
                                year  = t.dt.year.item()
                                month  = t.dt.month.item()
                                month_string = str(year) + '-' + str(month)
                                print(month_string)

                                # Here month is 'mm-yyyy'
                                iceFluxesByMonth[month_string] = computeIceFluxesCAAFromGates(gates, cesm_a_uvel['uvel'][ti,:,:].values, cesm_a_vvel['vvel'][ti,:,:].values, cesm_a_aice['aice'][ti,:,:].values, "B2", measures)

                            cesm_a_uvel.close()
                            cesm_a_vvel.close()
                            cesm_a_aice.close()
                else: print(filename)

    else:
        for i, directory in enumerate(em['main']):
            # For all files in hist or proj
            for filename in os.listdir(directory):
                if 'cice.h.' in filename:
                    allVar = os.path.join(directory+ filename)

                    if os.path.isfile(allVar):
                        cesm_allVar = xarray.open_dataset(allVar, decode_times=True)

                        for ti, t in enumerate(cesm_allVar['uvel'].time):
                            year  = t.dt.year.item()
                            month  = t.dt.month.item()
                            month_string = str(year) + '-' + str(month)
                            print(month_string)

                            # Here month is 'mm-yyyy'
                            iceFluxesByMonth[month_string] = computeIceFluxesCAAFromGates(gates, cesm_allVar['uvel'][ti,:,:].values, cesm_allVar['vvel'][ti,:,:].values, cesm_allVar['aice'][ti,:,:].values, "B2", measures)

                        cesm_allVar.close()

    # Store monthly SIA fluxes in an excel file
    output_file = 'ice_flux_iHESP_' + em['em'] + '.xlsx'
    create_excel_file(iceFluxesByMonth, output_file)

# %% Compute Gates Lenght in model
cesm_new = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.200001-200611.nc', decode_times=True)
cesm_grid = xarray.open_dataset(r'/aos/home/mfol/Data/CESM/domain.ocn.tx0.1v2_090218.nc', decode_times=False)
gates = callGatesCESM_HR()
tlon = cesm_grid.xc
tlat = cesm_grid.yc

iceFluxesByMonth = {}
measures = {
    'htn': cesm_new['HTN'],
    'hte': cesm_new['HTE'],
    }
gatelenghts = computeGatesLenghtInModel(gates, measures)

# %% Compute physical lenght of gates
for gate in gates:
    [lon1, lat1, lon2, lat2] = [cesm_grid.xv[gate['gate'][0][0],gate['gate'][0][1],3].values,
                                cesm_grid.yv[gate['gate'][0][0],gate['gate'][0][1],3].values,
                                cesm_grid.xv[gate['gate'][-1][0],gate['gate'][-1][1],3].values,
                                cesm_grid.yv[gate['gate'][-1][0],gate['gate'][-1][1],3].values]
    measure = getMeasureInMeters(lat1, lon1, lat2, lon2)
    print(gate['name'] + ' measures :' + str(measure/1000))