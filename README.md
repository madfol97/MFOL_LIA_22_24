Here are all the scripts used to compute the results and figures of Fol et al., 2024.


FIGURES  -   FILE
_______________________________
Figure 1 - mapRegionAndGrids.py

Figure 2 - mapRegionAndGrids.py

Figure 3 - plotSIE.py

Figure 4 - plotSICSITSpatial.py

Figure 5 - plotSIE.py

Figure 6 - thickness.py

Figure 7 - meltSeason.py

Figure 8 - fluxesFigures.py

Figure 9 - fluxesFigures.py

Figure 10 - fluxesFigures.py

Figure 11 - plotSIE.py

Figure 12 - tendencies.py

Figure 13 - tendencies.py

Figure 14 - tendencies.py




_____________________MAPS_____________________________

mapRegionAndGrids.py: Used for Figure 1 and Figure 2. Map of regions of interest, the LIA-N, QEI and CAA-S. Grids of CESM1.3-HR, LR and CESM2.LE.

createMasks.py: Create nc files for regions (LIA-N, QEI, CAA-S) masked for CESM1.3-HR. Could be adjusted for other grids. 


____________________SIE_SIA_SIT_____________________________

CIS2aice.py: CIS gridded ice charts see data availability, analysis, computation and storing in excel file of SIE and SIA per region (QEI and CAA-S). The biggestmaskContinent is the most extensive landmark. See CISToRead.pdf for data availability matrix and sampling points for regions.

nsidcAice.py: Computes and save in excel file SIA and SIE per region (LIA-N, QEI, CAA-S and pan-Arctic) for NSIDC CDR data.

computeSIE.py: Compute pan-Arctic and regional (LIA-N, QEI, CAA-S) SIE and SIA for CESM1.3-HR, CESM2-LE and CESM1.3-LR and save in excel files. Mush change excel file names, dates to march or sept and applied mask when for a region.

plotSIE.py: Used for Figure 3,5 and 11: Pan artic SIE, SIA, mean May SIT, regional (LIA-N, QEI,CAA-S) SIE, and thermodynamic and dynamic SIA contribution to the melt season SIA loss. Read observed and simulated SIA and SIE sotred in excel files. Uses results from computeSIE.py, computeMeanSIT.py, tendency.py, nsidcAice.py and piomass.py.

plotSICSITSpatial.py: Used for Figure 4. Compute and plot 20 years average SIC and SIT for models and observation. Plot MIZ contours on SIC. 


piomass.py: Functions to read and retreive SIC, SIT and SIT distribution (gice) from PIOMAS datasets. 

computeMeanSIT.py: Compute mean pan Artic May SIT for CESM1.3-HR, LR and CESM2.LE. 

thickness.py: Used for Figure 6. Computes and plot 20-years average thickness distribution for CESM1.3-HR and PIOMAS. For CESM1.3-HR uses aicen001-2-3-4-5 categories and retreives frequencies by doing the ratio of each aicen00x/aice. For PIOMAS, the distribution is already computed, but capping of the thickest bins for easier comparision with CESM1.3-HR. Uses function from piomass.py.


____________________INTEGRATED_QUANTITIES________________________

meltSeason.py: Used for Figure 7. Compute simulated melt season from linear interpolation when the annual sum of the daidtt over a region (LIA-N, QEI, CAA-S) crosses 0. Plot observed and simulated melt season per region.

tendencies.py: Used for Figures 12,13 and 14. Computes the melt season integrated dynamic (ridging and advection) and thermodynamic SIA loss for each regions (LIA-N, QEI, CAA-S). Uses results computed in meltSeason.py. Plots the timeseries, computes the correlation between thermodynamic and dynamic tendencies. Computes the spectral analysis (FFT) of the 20-years detrended thermo and dyn tendencies SIA loss and SIA loss from fluxes.



_____________________________ICE FLUX_______________________________
ice_flux_algo.py: All functions used to automatically select gates from approximate edges coordinates [lon1,lat1,lon2,lat2]. Functions to select grid points between the two (DDO algirithm), then select edges of those grid points. Defined for curvilinear grids of type C grid (ORCA) and B grid (tripolar ne.120_t12).

fluxes_helper.py: Create excel dataFrame with one column per gates. Call ice_flux_algo functions to compute SIA fluxes for CESM1.3-HR and readjust the selected grid points to get rid of land segments.

computeIceFluxes.py: Call ice_flux_algo and fluxes_helper functions to compute SIA fluxes for CESM1.3-HR. Creates one excel file per esnemble with monthly values from 1920-2100. Verification of gates lenght in the model compared to a physical lenght.

fluxesFigures.py: Used for Figure 8,9,10. Read observed and simulated SIA fluxes stored in excel files produced from computeIceFluxes.py. Convert to 10^3km2/month units, combine gates into QEI-in, QEI-out, compute 20-year seasonal cycles (for common period for observed fluxes) and ensemble mean. Plot seasonal cycle, annual budget for the QEI and CAA-S.


______________________________________________
supplemental - plotsLibrary.py: Functions to plot projected quantities and quiver plots - not used for any figure in the manuscript, but usefull.
