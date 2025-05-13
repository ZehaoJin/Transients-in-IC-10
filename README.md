# Transients-in-IC-10
### [1. ZTF IC 10 variable catalog (Main product)](#1-ztf-ic-10-variable-catalog-main-product-1)
### [2. Cite this work](#2-Cite-this-work-1)
### [3. Non-variables, non-periodic variables, and periodic variables identified by this work](#3-Non-variables-non-periodic-variables-and-periodic-variables-identified-by-this-work-1)
### [4. Cross match with SIMBAD，X-ray and Gaia](#4-cross-match-with-simbadx-ray-and-gaia-1)
### [5. HR diagrams](#5-HR-diagrams-1)
### [6. Interesting lightcurves](#6-Interesting-lightcurves-1)
### [7. All periodic lightcurves](https://github.com/ZehaoJin/Transients-in-IC-10/blob/main/All_periodic_lightcurves.ipynb)


## 1. ZTF IC 10 variable catalog (Main product)
- official data archived on Zenodo: [https://zenodo.org/records/14979711](https://zenodo.org/records/14979711)
- ZTF r band catalog: [csv](df_zr.csv) / [ecsv (astropy Qtable)](t_zr_20240401.ecsv)
- ZTF g band catalog: [csv](df_zg.csv)  / [ecsv (astropy Qtable)](t_zg_20240401.ecsv)
- color catalog: [csv](df_color.csv) / [ecsv (astropy Qtable)](t_color_0912.ecsv)

## 2. Cite this work
If you use this repository or would like to refer the paper, please use the following BibTeX entry:

    @article{Jin_2025,
    doi = {10.3847/1538-4365/adc7fe},
    url = {https://dx.doi.org/10.3847/1538-4365/adc7fe},
    year = {2025},
    month = {may},
    publisher = {The American Astronomical Society},
    volume = {278},
    number = {1},
    pages = {31},
    author = {Jin, Zehao and Gelfand, Joseph D.},
    title = {ZTF IC 10 Variable Catalog},
    journal = {The Astrophysical Journal Supplement Series},
    abstract = {To study how massive variable stars affect their environment, we search for variability among Zwicky Transient Facility (ZTF) sources located within the optical extent of the nearby starburst galaxy IC 10. We present the ZTF IC 10 catalog, which classifies 1516 r-band sources and 864 g-band sources within a 225″ radius around IC 10 into three categories: 1388 (767) r- (g)-band nonvariables, 150 (85) r- (g)-band nonperiodic variables, and 37 (12) r- (g)-band periodic variables. Among them 101 (48) r- (g)-band nonperiodic variables and 22 (4) r- (g)-band periodic variables are inside IC 10. We verify our classification by crossmatching with previous variability catalogs and machine learning–powered classifications. Various analysis including population demographics, color–magnitude diagrams, and crossmatching with a set of different surveys and database such as Gaia, XMM-Newton, Chandra, and SIMBAD are also presented. Based on source density and parallax, we distinguish sources within IC 10 from non–IC 10 sources. For IC 10 sources, we highlight flaring supergiants, a source with a long secondary period, and periodic supergiants including a possible S Doradus luminous blue variable and candidate Miras. For non–IC 10 sources, we present superred sources and compact objects such as a possible long-period subdwarf and a periodic X-ray source. The catalog can serve as a useful database to study the connections between various types of massive stars and their host galaxies.}
    }


## 3. Non-variables, non-periodic variables, and periodic variables identified by this work
<img src="plots/cmd_r.png" width="500">  <img src="plots/cmd_g.png" width="500">

## 4. Cross match with SIMBAD，X-ray and Gaia
<img src="plots/simbad_r.png" width="330"> <img src="plots/xray_r.png" width="330"> <img src="plots/plx_r.png" width="330">

## 5. HR diagrams
<img src="plots/HRdiagram_IC10.png" width="500"> <img src="plots/HRdiagram_noneIC10.png" width="500">

## 6. Interesting lightcurves
### 6.1. Flaring Super Giants
<img src="plots/lc_128.png" width="500"> <img src="plots/lc_1108.png" width="500">

### 6.2. Luminous Blue Variable (LBV) with Long Secondary Period (LSP)
<img src="plots/sin_slope_193.png" width="500">

### 6.3. Periodic Super Giants
<img src="plots/folded_lc_90.png" width="330"> <img src="plots/folded_lc_158.png" width="330"> <img src="plots/folded_lc_223.png" width="330">
<img src="plots/folded_lc_225.png" width="330"> <img src="plots/folded_lc_354.png" width="330">

### 6.4. Luminous Blue Variable (LBV) Candidate
<img src="plots/lc_color_90.png" width="500"> <img src="plots/g_g-r_90.png" width="500">

### 6.5. Mira Candidates
<img src="plots/super_red_242.png" width="330"> <img src="plots/super_red_577.png" width="330"> <img src="plots/super_red_1259.png" width="330">

### 6.6. Possible Periodic Subdwarf
<img src="plots/lc_145.png" width="500"> <img src="plots/sin_145.png" width="500">


### 6.7. Low Mass X-ray Binary in the Milky Way
<img src="plots/lc_2008.png" width="500"> <img src="plots/sin_2008.png" width="500">
