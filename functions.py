from astropy.io import fits
import numpy as np
from astropy.table import QTable,vstack
import astropy.units as u
import astropy.utils
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.stats import chisquare
from scipy.stats import chi2
from scipy.stats import lognorm
from astropy.timeseries import LombScargle
import time
from astropy.visualization import hist
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from astropy.timeseries import BoxLeastSquares
from astropy.timeseries import TimeSeries
from astropy.time import Time
import george
from george.kernels import ExpSquaredKernel
from scipy.optimize import minimize
from tqdm import tqdm


t_zr=QTable.read('t_zr_0307.ecsv')
t_zg=QTable.read('t_zg_0307.ecsv')


def draw_lightcurve(SourceID):
    plt.figure(figsize=(20,15))
    plt.title('Source ID = '+str(SourceID),fontsize=40)
    #rbandcolor=['red','salmon','maroon']
    #gbandcolor=['green','limegreen','seagreen']
    rbandcolor=['red','black','maroon']
    gbandcolor=['green','blue','seagreen']
    ubandcolor=['violet']


    OIDs=t_zr[t_zr['SourceID']==SourceID]['OID']
    for i,OID in enumerate(OIDs):
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        
        #remove the bad one day
        t_ztf=t_ztf[t_ztf['catflags']==0]
        
        t_ztf=t_ztf[(t_ztf['mjd']<58481*u.day)|(t_ztf['mjd']>58482*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<58472*u.day)|(t_ztf['mjd']>58474*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<59032*u.day)|(t_ztf['mjd']>59033*u.day)]

        plt.errorbar(np.array(t_ztf['mjd']),np.array(t_ztf['mag']),yerr=np.array(t_ztf['magerr']),fmt='xb',ecolor='grey',capsize=2,label='r band',markeredgecolor=rbandcolor[i])

    OIDs=t_zg[t_zg['SourceID']==SourceID]['OID']
    for i,OID in enumerate(OIDs):
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        t_ztf=t_ztf[t_ztf['catflags']==0]
        plt.errorbar(np.array(t_ztf['mjd']),np.array(t_ztf['mag']),yerr=np.array(t_ztf['magerr']),fmt='xb',ecolor='grey',capsize=2,label='g band',markeredgecolor=gbandcolor[i])


    plt.xlabel('mjd',fontsize=40)
    plt.ylabel('mag',fontsize=40)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.gca().invert_yaxis()
    return None


def draw_fold_lightcurve(SourceID,force_period=None):
    plt.figure(figsize=(20,15))
    plt.title('Source ID = '+str(SourceID))


    OIDs_r=t_zr[t_zr['SourceID']==SourceID]['OID']
    OIDs_g=t_zg[t_zg['SourceID']==SourceID]['OID']
    length=max(len(OIDs_r),len(OIDs_g))
    for i,OID in enumerate(OIDs_r):
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)

        #remove the bad one day
        t_ztf=t_ztf[t_ztf['catflags']==0]
        
        t_ztf=t_ztf[(t_ztf['mjd']<58481*u.day)|(t_ztf['mjd']>58482*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<58472*u.day)|(t_ztf['mjd']>58474*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<59032*u.day)|(t_ztf['mjd']>59033*u.day)]


        ts=TimeSeries(time=Time(t_ztf['mjd'],format='mjd'),data={'mag': t_ztf['mag']})
        if force_period!=None:
            ts_folded=ts.fold(period=force_period*u.day)
        else:
            ts_folded=ts.fold(period=t_zr[t_zr['OID']==OID]['period']*u.day)

        plt.subplot(2,length,i+1)
        plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=ts.time.jd ,s=4)
        plt.colorbar()
        plt.xlabel('Time (days)')
        plt.ylabel('mag')
        plt.gca().invert_yaxis()
        plt.title('R band  OID = '+str(OID))


    for i,OID in enumerate(OIDs_g):
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        t_ztf=t_ztf[t_ztf['catflags']==0]
        ts=TimeSeries(time=Time(t_ztf['mjd'],format='mjd'),data={'mag': t_ztf['mag']})
        if force_period!=None:
            ts_folded=ts.fold(period=force_period*u.day)
        else:
            ts_folded=ts.fold(period=t_zg[t_zg['OID']==OID]['period']*u.day)

        plt.subplot(2,length,i+1+length)
        plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=ts.time.jd , s=4)
        plt.colorbar()
        plt.xlabel('Time (days)')
        plt.ylabel('mag')
        plt.gca().invert_yaxis()
        plt.title('G band  OID = '+str(OID))


    return None

def sinfit(SourceID,force_period=None):
    fs=20
    def sinfunc(t, A, w, p, c):
        return A * np.sin(w*t + p) + c
    plt.figure(figsize=(20,15))
    plt.title('Source ID = '+str(SourceID),fontsize=fs)


    OIDs_r=t_zr[t_zr['SourceID']==SourceID]['OID']
    OIDs_g=t_zg[t_zg['SourceID']==SourceID]['OID']
    length=max(len(OIDs_r),len(OIDs_g))
    for i,OID in enumerate(OIDs_r):
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)

        #remove the bad one day
        t_ztf=t_ztf[t_ztf['catflags']==0]
        
        t_ztf=t_ztf[(t_ztf['mjd']<58481*u.day)|(t_ztf['mjd']>58482*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<58472*u.day)|(t_ztf['mjd']>58474*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<59032*u.day)|(t_ztf['mjd']>59033*u.day)]


        ts=TimeSeries(time=Time(t_ztf['mjd'],format='mjd'),data={'mag': t_ztf['mag'],'magerr':t_ztf['magerr']})
        if force_period!=None:
            ts_folded=ts.fold(period=force_period*u.day)
            guess_period=force_period
        else:
            ts_folded=ts.fold(period=t_zr[t_zr['OID']==OID]['period']*u.day)
            guess_period=t_zr[t_zr['OID']==OID]['period']*u.day
            guess_period=float(guess_period.value)

        guess_freq = 1./guess_period
        guess_amp = np.std(ts_folded['mag'].value) * 2.**0.5
        guess_offset = np.mean(ts_folded['mag'].value)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])


        popt, pcov =curve_fit(sinfunc,ts_folded.time.jd,ts_folded['mag'],p0=guess)
        A, w, p, c = popt
        f = w/(2.*np.pi)
        fit_period=1./f
        print('L-S_period:',guess_period)
        print('fit_period:',fit_period)
        tt=np.linspace(ts_folded.time.jd.min(),ts_folded.time.jd.max(),100)
        yy=A*np.sin(w*tt+p)+c
        y=A*np.sin(w*ts_folded.time.jd+p)+c

        sin_chi2=(((ts_folded['mag'].value-y)**2)/(ts_folded['magerr'].value**2)).sum()/(len(ts_folded)-1)
        print('sin_chi2:',sin_chi2)


        plt.subplot(2,length,i+1)
        plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=ts.time.jd ,s=4)
        plt.plot(tt,yy,c='r',label='sinusoidal fit')
        plt.legend()
        plt.colorbar()
        plt.xlabel('Time (days)',fontsize=fs)
        plt.ylabel('mag',fontsize=fs)
        plt.gca().invert_yaxis()
        plt.title('R band  OID = '+str(OID),fontsize=fs*1.0)


    for i,OID in enumerate(OIDs_g):
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        t_ztf=t_ztf[t_ztf['catflags']==0]
        ts=TimeSeries(time=Time(t_ztf['mjd'],format='mjd'),data={'mag': t_ztf['mag'],'magerr':t_ztf['magerr']})
        if force_period!=None:
            ts_folded=ts.fold(period=force_period*u.day)
            guess_period=force_period
        else:
            ts_folded=ts.fold(period=t_zg[t_zg['OID']==OID]['period']*u.day)
            guess_period=t_zg[t_zg['OID']==OID]['period']*u.day
            guess_period=float(guess_period.value)

        guess_freq = 1./guess_period
        guess_amp = np.std(ts_folded['mag'].value) * 2.**0.5
        guess_offset = np.mean(ts_folded['mag'].value)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])


        popt, pcov =curve_fit(sinfunc,ts_folded.time.jd,ts_folded['mag'],p0=guess)
        A, w, p, c = popt
        f = w/(2.*np.pi)
        fit_period=1./f
        print('L-S_period:',guess_period)
        print('fit_period:',fit_period)
        tt=np.linspace(ts_folded.time.jd.min(),ts_folded.time.jd.max(),100)
        yy=A*np.sin(w*tt+p)+c
        y=A*np.sin(w*ts_folded.time.jd+p)+c

        sin_chi2=(((ts_folded['mag'].value-y)**2)/(ts_folded['magerr'].value**2)).sum()/(len(ts_folded)-1)
        print('sin_chi2:',sin_chi2)

        plt.subplot(2,length,i+1+length)
        plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=ts.time.jd , s=4)
        plt.plot(tt,yy,c='r',label='sinusoidal fit')
        plt.legend()
        plt.colorbar()
        plt.xlabel('Time (days)',fontsize=fs)
        plt.ylabel('mag',fontsize=fs)
        plt.gca().invert_yaxis()
        plt.title('G band  OID = '+str(OID),fontsize=fs*1.0)


    return None


def sinfit_single(SourceID,force_period=None):
    fs=20
    def sinfunc(t, A, w, p, c):
        return A * np.sin(w*t + p) + c
    plt.figure(figsize=(20,7.5))
    plt.title('Source ID = '+str(SourceID),fontsize=fs)


    OIDs_r=t_zr[t_zr['SourceID']==SourceID]
    OIDs_g=t_zg[t_zg['SourceID']==SourceID]
    length=1
    
    if len(OIDs_r)==1:
        OID=OIDs_r['OID']
    else:
        OID=OIDs_r[OIDs_r['numobs'].argmax()]['OID']
    
    OID_str=str(int(OID))
    t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)

    #remove the bad one day
    t_ztf=t_ztf[t_ztf['catflags']==0]
        
    t_ztf=t_ztf[(t_ztf['mjd']<58481*u.day)|(t_ztf['mjd']>58482*u.day)]
    t_ztf=t_ztf[(t_ztf['mjd']<58472*u.day)|(t_ztf['mjd']>58474*u.day)]
    t_ztf=t_ztf[(t_ztf['mjd']<59032*u.day)|(t_ztf['mjd']>59033*u.day)]


    ts=TimeSeries(time=Time(t_ztf['mjd'],format='mjd'),data={'mag': t_ztf['mag'],'magerr':t_ztf['magerr']})
    if force_period!=None:
        ts_folded=ts.fold(period=force_period*u.day)
        guess_period=force_period
    else:
        ts_folded=ts.fold(period=t_zr[t_zr['OID']==OID]['period']*u.day)
        guess_period=t_zr[t_zr['OID']==OID]['period']*u.day
        guess_period=float(guess_period.value)

    guess_freq = 1./guess_period
    guess_amp = np.std(ts_folded['mag'].value) * 2.**0.5
    guess_offset = np.mean(ts_folded['mag'].value)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])


    popt, pcov =curve_fit(sinfunc,ts_folded.time.jd,ts_folded['mag'],p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fit_period=1./f
    print('L-S_period:',guess_period)
    print('fit_period:',fit_period)
    tt=np.linspace(ts_folded.time.jd.min(),ts_folded.time.jd.max(),100)
    yy=A*np.sin(w*tt+p)+c
    y=A*np.sin(w*ts_folded.time.jd+p)+c

    sin_chi2=(((ts_folded['mag'].value-y)**2)/(ts_folded['magerr'].value**2)).sum()/(len(ts_folded)-1)
    print('sin_chi2:',sin_chi2)


    plt.subplot(1,2,1)
    plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=ts.time.jd ,s=4)
    plt.plot(tt,yy,c='r',label='sinusoidal fit')
    plt.legend()
    plt.colorbar()
    plt.xlabel('Time (days)',fontsize=fs)
    plt.ylabel('mag',fontsize=fs)
    plt.gca().invert_yaxis()
    plt.title('R band  OID = '+str(OID),fontsize=fs*1.0)

    if len(OIDs_g)==1:
        OID=OIDs_g['OID']
    else:
        OID=OIDs_g[OIDs_g['numobs'].argmax()]['OID']
    
    OID_str=str(int(OID))
    t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
    t_ztf=t_ztf[t_ztf['catflags']==0]
    ts=TimeSeries(time=Time(t_ztf['mjd'],format='mjd'),data={'mag': t_ztf['mag'],'magerr':t_ztf['magerr']})
    if force_period!=None:
        ts_folded=ts.fold(period=force_period*u.day)
        guess_period=force_period
    else:
        ts_folded=ts.fold(period=t_zg[t_zg['OID']==OID]['period']*u.day)
        guess_period=t_zg[t_zg['OID']==OID]['period']*u.day
        guess_period=float(guess_period.value)

    guess_freq = 1./guess_period
    guess_amp = np.std(ts_folded['mag'].value) * 2.**0.5
    guess_offset = np.mean(ts_folded['mag'].value)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])


    popt, pcov =curve_fit(sinfunc,ts_folded.time.jd,ts_folded['mag'],p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fit_period=1./f
    print('L-S_period:',guess_period)
    print('fit_period:',fit_period)
    tt=np.linspace(ts_folded.time.jd.min(),ts_folded.time.jd.max(),100)
    yy=A*np.sin(w*tt+p)+c
    y=A*np.sin(w*ts_folded.time.jd+p)+c

    sin_chi2=(((ts_folded['mag'].value-y)**2)/(ts_folded['magerr'].value**2)).sum()/(len(ts_folded)-1)
    print('sin_chi2:',sin_chi2)

    plt.subplot(1,2,2)
    plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=ts.time.jd , s=4)
    plt.plot(tt,yy,c='r',label='sinusoidal fit')
    plt.legend()
    plt.colorbar()
    plt.xlabel('Time (days)',fontsize=fs)
    plt.ylabel('mag',fontsize=fs)
    plt.gca().invert_yaxis()
    plt.title('G band  OID = '+str(OID),fontsize=fs*1.0)


    return None


def sinfit2(SourceID,force_period=None):
    def sinfunc(t, A, w, p, c):
        return A * np.sin(w*t + p) + c

    plt.figure(figsize=(20,15))
    plt.title('Source ID = '+str(SourceID))
    #rbandcolor=['red','salmon','maroon']
    #gbandcolor=['green','limegreen','seagreen']
    rbandcolor=['red','black','maroon']
    gbandcolor=['green','blue','seagreen']
    ubandcolor=['violet']


    OIDs=t_zr[t_zr['SourceID']==SourceID]['OID']
    for i,OID in enumerate(OIDs):
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        #remove the bad one day
        t_ztf=t_ztf[t_ztf['catflags']==0]
        
        t_ztf=t_ztf[(t_ztf['mjd']<58481*u.day)|(t_ztf['mjd']>58482*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<58472*u.day)|(t_ztf['mjd']>58474*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<59032*u.day)|(t_ztf['mjd']>59033*u.day)]

        if force_period!=None:
            guess_period=force_period
        else:
            guess_period=t_zr[t_zr['OID']==OID]['period']*u.day
            guess_period=float(guess_period.value)

        guess_freq = 1./guess_period
        guess_amp = np.std(t_ztf['mag'].value) * 2.**0.5
        guess_offset = np.mean(t_ztf['mag'].value)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])


        popt, pcov =curve_fit(sinfunc,t_ztf['mjd'].value,t_ztf['mag'].value,p0=guess)
        A, w, p, c = popt
        f = w/(2.*np.pi)
        fit_period=1./f
        print('L-S_period:',guess_period)
        print('fit_period:',fit_period)
        tt=np.linspace(t_ztf['mjd'].value.min(),t_ztf['mjd'].value.max(),500)
        yy=A*np.sin(w*tt+p)+c
        y=A*np.sin(w*t_ztf['mjd'].value+p)+c

        sin_chi2=(((t_ztf['mag'].value-y)**2)/(t_ztf['magerr'].value**2)).sum()/(len(t_ztf)-1)
        print('sin_chi2:',sin_chi2)


        plt.errorbar(np.array(t_ztf['mjd']),np.array(t_ztf['mag']),yerr=np.array(t_ztf['magerr']),fmt='xb',ecolor='grey',capsize=2,label='r band',markeredgecolor=rbandcolor[i])
        plt.plot(tt,yy,c='r',label='sinusoidal fit')

    OIDs=t_zg[t_zg['SourceID']==SourceID]['OID']
    for i,OID in enumerate(OIDs):
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        t_ztf=t_ztf[t_ztf['catflags']==0]

        if force_period!=None:
            guess_period=force_period
        else:
            guess_period=t_zg[t_zg['OID']==OID]['period']*u.day
            guess_period=float(guess_period.value)

        guess_freq = 1./guess_period
        guess_amp = np.std(t_ztf['mag'].value) * 2.**0.5
        guess_offset = np.mean(t_ztf['mag'].value)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])


        popt, pcov =curve_fit(sinfunc,t_ztf['mjd'].value,t_ztf['mag'].value,p0=guess)
        A, w, p, c = popt
        f = w/(2.*np.pi)
        fit_period=1./f
        print('L-S_period:',guess_period)
        print('fit_period:',fit_period)
        tt=np.linspace(t_ztf['mjd'].value.min(),t_ztf['mjd'].value.max(),500)
        yy=A*np.sin(w*tt+p)+c
        y=A*np.sin(w*t_ztf['mjd'].value+p)+c

        sin_chi2=(((t_ztf['mag'].value-y)**2)/(t_ztf['magerr'].value**2)).sum()/(len(t_ztf)-1)
        print('sin_chi2:',sin_chi2)

        plt.errorbar(np.array(t_ztf['mjd']),np.array(t_ztf['mag']),yerr=np.array(t_ztf['magerr']),fmt='xb',ecolor='grey',capsize=2,label='g band',markeredgecolor=gbandcolor[i])
        plt.plot(tt,yy,c='r',label='sinusoidal fit')


    plt.xlabel('mjd')
    plt.ylabel('mag')
    plt.legend()
    plt.gca().invert_yaxis()
    return None

def windowfunc(ztf_ID,high=2800,low=0.1,nan=False):
    t=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+str(ztf_ID))
    #remove bad days
    t=t[t['catflags']==0]
    
    t=t[(t['mjd']<58481*u.day)|(t['mjd']>58482*u.day)]
    t=t[(t['mjd']<58472*u.day)|(t['mjd']>58474*u.day)]
    t=t[(t['mjd']<59032*u.day)|(t['mjd']>59033*u.day)]

    con_mag=np.zeros(t['mag'].shape)+t['mag'].mean()
    con_err=np.zeros(t['magerr'].shape)+t['magerr'].mean()

    if nan==False:
        ls = LombScargle(t['mjd'],con_mag, con_err)
    else:
        # Probably need this if you see NaN
        ls = LombScargle(t['mjd'],con_mag, con_err,fit_mean=False, center_data=False)


    freq, power = ls.autopower(nyquist_factor=300,minimum_frequency=1/(high*u.day),maximum_frequency=1/(low*u.day))
    #LNP_SIG=ls.false_alarm_probability(power.max())
    plt.plot(1/freq, power,label='window function',alpha=0.5)
    plt.xlabel('period (days)')
    plt.ylabel('power')
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")

    ls = LombScargle(t['mjd'],t['mag'], t['magerr'])
    freq, power = ls.autopower(nyquist_factor=300,minimum_frequency=1/(high*u.day),maximum_frequency=1/(low*u.day))
    LNP_SIG=ls.false_alarm_probability(power.max())
    plt.plot(1/freq, power,label='L-S',alpha=0.5)
    plt.legend()
    print('period=',1/freq[power.argmax()],'    LNP_SIG=',ls.false_alarm_probability(power.max()))

    return None

def color(SourceID):
    plt.figure(figsize=(20,15))
    plt.title('Source ID = '+str(SourceID))

    t_zr_cand=t_zr[t_zr['SourceID']==SourceID]
    t_zg_cand=t_zg[t_zg['SourceID']==SourceID]
    OID_zr=t_zr_cand[t_zr_cand['numobs'].argmax()]['OID']
    OID_zg=t_zg_cand[t_zg_cand['numobs'].argmax()]['OID']
    OID_zr=str(int(OID_zr))
    OID_zg=str(int(OID_zg))

    #print(OID_zr,OID_zg)

    tt_zr=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_zr)
    tt_zg=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_zg)

    # clean data flag
    tt_zr=tt_zr[tt_zr['catflags']==0]
    tt_zg=tt_zg[tt_zg['catflags']==0]
    
    #remove the bad one day
    tt_zr=tt_zr[(tt_zr['mjd']<58481*u.day)|(tt_zr['mjd']>58482*u.day)]
    tt_zr=tt_zr[(tt_zr['mjd']<58472*u.day)|(tt_zr['mjd']>58474*u.day)]
    tt_zr=tt_zr[(tt_zr['mjd']<59032*u.day)|(tt_zr['mjd']>59033*u.day)]

    start=np.floor(min([tt_zr['mjd'].value.min(),tt_zg['mjd'].value.min()]))
    end=np.ceil(max([tt_zr['mjd'].value.max(),tt_zg['mjd'].value.max()]))

    days=np.arange(start,end)
    arr=np.zeros((3,len(days)))
    arr[0,:]=days
    ## arr[days,zr,zg]

    for i,day in enumerate(days):
        zr_cand=tt_zr[np.floor(tt_zr['mjd'].value)==day]
        zg_cand=tt_zg[np.floor(tt_zg['mjd'].value)==day]
        if len(zr_cand)==1 and len(zg_cand)==1:
            arr[1,i]=zr_cand['mag'].value
            arr[2,i]=zg_cand['mag'].value
        else:
            arr[1,i]=np.nan
            arr[2,i]=np.nan

    arr=arr[:,np.isnan(arr[1,:])==False]

    plt.scatter(arr[0],arr[2]-arr[1])  #G-R
    plt.ylabel('G-R',fontsize=20)
    plt.xlabel('mjd',fontsize=20)

    return None

#'''
def MGPRfit3(SourceID,force_period=None,errscale=1):
    #errscale=10
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    plt.figure(figsize=(20,15))
    plt.title('Source ID = '+str(SourceID))


    OIDs_r=t_zr[t_zr['SourceID']==SourceID]['OID']
    OIDs_g=t_zg[t_zg['SourceID']==SourceID]['OID']
    length=max(len(OIDs_r),len(OIDs_g))
    for i,OID in enumerate(OIDs_r):
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)

        #remove the bad one day
        t_ztf=t_ztf[t_ztf['catflags']==0]
        
        t_ztf=t_ztf[(t_ztf['mjd']<58481*u.day)|(t_ztf['mjd']>58482*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<58472*u.day)|(t_ztf['mjd']>58474*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<59032*u.day)|(t_ztf['mjd']>59033*u.day)]


        ts=TimeSeries(time=Time(t_ztf['mjd'],format='mjd'),data={'mag': t_ztf['mag'],'magerr':t_ztf['magerr']})
        if force_period!=None:
            ts_folded=ts.fold(period=force_period*u.day)
        else:
            ts_folded=ts.fold(period=t_zr[t_zr['OID']==OID]['period']*u.day)

        # Set up the Gaussian process.
        kernel = ExpSquaredKernel(np.var(ts_folded['mag'].value))
        gp = george.GP(kernel)
        # Pre-compute the factorization of the matrix.
        gp.compute(ts_folded.time.jd, ts_folded['magerr'].value*errscale)
        # Compute the log likelihood.
        # print(gp.lnlikelihood(y))

        tt = np.linspace(ts_folded.time.jd.min(),ts_folded.time.jd.max(),500)
        mu, cov = gp.predict(ts_folded['mag'].value, tt)
        std = np.sqrt(np.diag(cov))


        y=ts_folded['mag'].value
        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
        #print(result)

        gp.set_parameter_vector(result.x)
        #print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(ts_folded['mag'].value)))

        mu, cov = gp.predict(ts_folded['mag'].value, tt)
        std = np.sqrt(np.diag(cov))


        plt.subplot(2,length,i+1)
        plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=ts.time.jd ,s=4)
        plt.plot(tt, mu, lw=2, label='MGPR fit')
        plt.fill_between(tt, mu-std, mu+std, color="b", alpha=0.2)
        plt.legend()
        plt.colorbar()
        plt.xlabel('Time (days)')
        plt.ylabel('mag')
        plt.gca().invert_yaxis()
        plt.title('R band  OID = '+str(OID))


    for i,OID in enumerate(OIDs_g):
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        t_ztf=t_ztf[t_ztf['catflags']==0]
        ts=TimeSeries(time=Time(t_ztf['mjd'],format='mjd'),data={'mag': t_ztf['mag'],'magerr':t_ztf['magerr']})
        if force_period!=None:
            ts_folded=ts.fold(period=force_period*u.day)
        else:
            ts_folded=ts.fold(period=t_zg[t_zg['OID']==OID]['period']*u.day)

        # Set up the Gaussian process.
        kernel = ExpSquaredKernel(np.var(ts_folded['mag'].value))
        gp = george.GP(kernel)
        # Pre-compute the factorization of the matrix.
        gp.compute(ts_folded.time.jd, ts_folded['magerr'].value*errscale)
        # Compute the log likelihood.
        # print(gp.lnlikelihood(y))

        tt = np.linspace(ts_folded.time.jd.min(),ts_folded.time.jd.max(),500)
        mu, cov = gp.predict(ts_folded['mag'].value, tt)
        std = np.sqrt(np.diag(cov))

        y=ts_folded['mag'].value
        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
        #print(result)

        gp.set_parameter_vector(result.x)
        #print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(ts_folded['mag'].value)))

        mu, cov = gp.predict(ts_folded['mag'].value, tt)
        std = np.sqrt(np.diag(cov))


        plt.subplot(2,length,i+1+length)
        plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=ts.time.jd , s=4)
        plt.plot(tt, mu, lw=2, label='MGPR fit')
        plt.fill_between(tt, mu-std, mu+std, color="b", alpha=0.2)
        plt.legend()
        plt.colorbar()
        plt.xlabel('Time (days)')
        plt.ylabel('mag')
        plt.gca().invert_yaxis()
        plt.title('G band  OID = '+str(OID))


    return None
#'''


#def MGPRfit2(SourceID,force_period=None):
def MGPRfit(SourceID,force_period=None,errscale=1,ylim=None):
    #errscale=1
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    plt.figure(figsize=(20,15))
    plt.title('Source ID = '+str(SourceID))


    OIDs_r=t_zr[t_zr['SourceID']==SourceID]['OID']
    OIDs_g=t_zg[t_zg['SourceID']==SourceID]['OID']
    for i,OID in enumerate(OIDs_r):
        OID_str=str(int(OID))
        tt_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)

        #remove the bad one day
        tt_ztf=tt_ztf[tt_ztf['catflags']==0]
        tt_ztf=tt_ztf[(tt_ztf['mjd']<58481*u.day)|(tt_ztf['mjd']>58482*u.day)]
        tt_ztf=tt_ztf[(tt_ztf['mjd']<58472*u.day)|(tt_ztf['mjd']>58474*u.day)]
        tt_ztf=tt_ztf[(tt_ztf['mjd']<59032*u.day)|(tt_ztf['mjd']>59033*u.day)]

        if i==0:
            t_ztf=tt_ztf
        else:
            t_ztf=vstack([t_ztf,tt_ztf])



    try:
        ts=TimeSeries(time=Time(t_ztf['mjd'],format='mjd'),data={'mag': t_ztf['mag'],'magerr':t_ztf['magerr']})
        if force_period!=None:
            ts_folded=ts.fold(period=force_period*u.day)
        else:
            #ts_folded=ts.fold(period=t_zr[t_zr['OID']==OID]['period']*u.day)
            ts_folded=ts.fold(period=t_zr[t_zr['SourceID']==SourceID]['period'][t_zr[t_zr['SourceID']==SourceID]['numobs'].argmax()]*u.day)

        # Set up the Gaussian process.
        kernel = ExpSquaredKernel(np.var(ts_folded['mag'].value))
        gp = george.GP(kernel)
        # Pre-compute the factorization of the matrix.
        gp.compute(ts_folded.time.jd, ts_folded['magerr'].value*errscale)
        # Compute the log likelihood.
        # print(gp.lnlikelihood(y))

        tt = np.linspace(ts_folded.time.jd.min(),ts_folded.time.jd.max(),500)
        mu, cov = gp.predict(ts_folded['mag'].value, tt)
        std = np.sqrt(np.diag(cov))


        y=ts_folded['mag'].value
        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
        #print(result)

        gp.set_parameter_vector(result.x)
        #print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(ts_folded['mag'].value)))

        mu, cov = gp.predict(ts_folded['mag'].value, tt)
        std = np.sqrt(np.diag(cov))


        plt.subplot(2,1,1)
        plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=ts.time.jd ,s=4)
        plt.plot(tt, mu, lw=2, label='MGPR fit')
        plt.fill_between(tt, mu-std, mu+std, color="b", alpha=0.2)
        plt.legend()
        plt.colorbar()
        plt.xlabel('Time (days)')
        plt.ylabel('mag')
        plt.gca().invert_yaxis()
        plt.title('Source ID = '+str(SourceID)+', R band  OID = '+str(OID))
        if ylim!=None:
            plt.ylim(ylim[0])

    except:
        print('No data found in R band')


    for i,OID in enumerate(OIDs_g):
        OID_str=str(int(OID))
        tt_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        tt_ztf=tt_ztf[tt_ztf['catflags']==0]

        if i==0:
            t_ztf=tt_ztf
        else:
            t_ztf=vstack([t_ztf,tt_ztf])
    
    try:
        ts=TimeSeries(time=Time(t_ztf['mjd'],format='mjd'),data={'mag': t_ztf['mag'],'magerr':t_ztf['magerr']})
        if force_period!=None:
            ts_folded=ts.fold(period=force_period*u.day)
        else:
            #ts_folded=ts.fold(period=t_zg[t_zg['OID']==OID]['period']*u.day)
            ts_folded=ts.fold(period=t_zg[t_zg['SourceID']==SourceID]['period'][t_zg[t_zg['SourceID']==SourceID]['numobs'].argmax()]*u.day)

        # Set up the Gaussian process.
        kernel = ExpSquaredKernel(np.var(ts_folded['mag'].value))
        gp = george.GP(kernel)
        # Pre-compute the factorization of the matrix.
        gp.compute(ts_folded.time.jd, ts_folded['magerr'].value*errscale)
        # Compute the log likelihood.
        # print(gp.lnlikelihood(y))

        tt = np.linspace(ts_folded.time.jd.min(),ts_folded.time.jd.max(),500)
        mu, cov = gp.predict(ts_folded['mag'].value, tt)
        std = np.sqrt(np.diag(cov))

        y=ts_folded['mag'].value
        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
        #print(result)

        gp.set_parameter_vector(result.x)
        #print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(ts_folded['mag'].value)))

        mu, cov = gp.predict(ts_folded['mag'].value, tt)
        std = np.sqrt(np.diag(cov))


        plt.subplot(2,1,2)
        plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=ts.time.jd , s=4)
        plt.plot(tt, mu, lw=2, label='MGPR fit')
        plt.fill_between(tt, mu-std, mu+std, color="b", alpha=0.2)
        plt.legend()
        plt.colorbar()
        plt.xlabel('Time (days)')
        plt.ylabel('mag')
        plt.gca().invert_yaxis()
        plt.title('Source ID = '+str(SourceID)+',G band  OID = '+str(OID))
        if ylim!=None:
            plt.ylim(ylim[1])
    except:
        print('No data found in G band')


    return None

#def MGPRfit3(SourceID,force_period=None):
def MGPRfit2(SourceID,force_period=None,errscale=1):
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    plt.figure(figsize=(20,15))
    plt.title('Source ID = '+str(SourceID))
    #rbandcolor=['red','salmon','maroon']
    #gbandcolor=['green','limegreen','seagreen']
    rbandcolor=['red','black','maroon']
    gbandcolor=['green','blue','seagreen']
    ubandcolor=['violet']


    OIDs=t_zr[t_zr['SourceID']==SourceID]['OID']
    for i,OID in enumerate(OIDs):
        OID_str=str(int(OID))
        tt_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        #remove the bad one day
        tt_ztf=tt_ztf[tt_ztf['catflags']==0]
        tt_ztf=tt_ztf[(tt_ztf['mjd']<58481*u.day)|(tt_ztf['mjd']>58482*u.day)]
        tt_ztf=tt_ztf[(tt_ztf['mjd']<58472*u.day)|(tt_ztf['mjd']>58474*u.day)]
        tt_ztf=tt_ztf[(tt_ztf['mjd']<59032*u.day)|(tt_ztf['mjd']>59033*u.day)]

        if i==0:
            t_ztf=tt_ztf
        else:
            t_ztf=vstack([t_ztf,tt_ztf])

    # Set up the Gaussian process.
    kernel = ExpSquaredKernel(np.var(t_ztf['mag'].value))
    gp = george.GP(kernel)
    # Pre-compute the factorization of the matrix.
    gp.compute(t_ztf['mjd'].value,t_ztf['magerr'].value*errscale)
    # Compute the log likelihood.
    # print(gp.lnlikelihood(y))

    tt = np.linspace(t_ztf['mjd'].value.min(),t_ztf['mjd'].value.max(),500)
    mu, cov = gp.predict(t_ztf['mag'].value, tt)
    std = np.sqrt(np.diag(cov))


    y=t_ztf['mag'].value
    result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    #print(result)

    gp.set_parameter_vector(result.x)
    #print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(t_ztf['mag'].value)))

    mu, cov = gp.predict(t_ztf['mag'].value, tt)
    std = np.sqrt(np.diag(cov))


    plt.errorbar(np.array(t_ztf['mjd']),np.array(t_ztf['mag']),yerr=np.array(t_ztf['magerr']),fmt='xb',ecolor='grey',capsize=2,label='r band',markeredgecolor=rbandcolor[0])
    plt.plot(tt, mu, lw=2, label='MGPR fit')
    plt.fill_between(tt, mu-std, mu+std, color="b", alpha=0.2)




    OIDs=t_zg[t_zg['SourceID']==SourceID]['OID']
    for i,OID in enumerate(OIDs):
        OID_str=str(int(OID))
        tt_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        tt_ztf=tt_ztf[tt_ztf['catflags']==0]

        if i==0:
            t_ztf=tt_ztf
        else:
            t_ztf=vstack([t_ztf,tt_ztf])

    # Set up the Gaussian process.
    kernel = ExpSquaredKernel(np.var(t_ztf['mag'].value))
    gp = george.GP(kernel)
    # Pre-compute the factorization of the matrix.
    gp.compute(t_ztf['mjd'].value,t_ztf['magerr'].value*errscale)
    # Compute the log likelihood.
    # print(gp.lnlikelihood(y))

    tt = np.linspace(t_ztf['mjd'].value.min(),t_ztf['mjd'].value.max(),500)
    mu, cov = gp.predict(t_ztf['mag'].value, tt)
    std = np.sqrt(np.diag(cov))


    y=t_ztf['mag'].value
    result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    #print(result)

    gp.set_parameter_vector(result.x)
    #print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(t_ztf['mag'].value)))

    mu, cov = gp.predict(t_ztf['mag'].value, tt)
    std = np.sqrt(np.diag(cov))


    plt.errorbar(np.array(t_ztf['mjd']),np.array(t_ztf['mag']),yerr=np.array(t_ztf['magerr']),fmt='xb',ecolor='grey',capsize=2,label='g band',markeredgecolor=gbandcolor[0])
    plt.plot(tt, mu, lw=2, label='MGPR fit')
    plt.fill_between(tt, mu-std, mu+std, color="b", alpha=0.2)


    plt.xlabel('mjd')
    plt.ylabel('mag')
    plt.legend()
    plt.gca().invert_yaxis()
    return None

def query(ID):
    display(t_zr[t_zr['SourceID']==ID])
    display(t_zg[t_zg['SourceID']==ID])


def color_fold(SourceID,force_period):
    plt.figure(figsize=(20,15))
    plt.title('Source ID = '+str(SourceID))

    t_zr_cand=t_zr[t_zr['SourceID']==SourceID]
    t_zg_cand=t_zg[t_zg['SourceID']==SourceID]
    OID_zr=t_zr_cand[t_zr_cand['numobs'].argmax()]['OID']
    OID_zg=t_zg_cand[t_zg_cand['numobs'].argmax()]['OID']
    OID_zr=str(int(OID_zr))
    OID_zg=str(int(OID_zg))

    #print(OID_zr,OID_zg)

    tt_zr=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_zr)
    tt_zg=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_zg)

    #remove the bad one day
    tt_zr=tt_zr[tt_zr['catflags']==0]
    tt_zg=tt_zg[tt_zg['catflags']==0]
    
    tt_zr=tt_zr[(tt_zr['mjd']<58481*u.day)|(tt_zr['mjd']>58482*u.day)]
    tt_zr=tt_zr[(tt_zr['mjd']<58472*u.day)|(tt_zr['mjd']>58474*u.day)]
    tt_zr=tt_zr[(tt_zr['mjd']<59032*u.day)|(tt_zr['mjd']>59033*u.day)]

    start=np.floor(min([tt_zr['mjd'].value.min(),tt_zg['mjd'].value.min()]))
    end=np.ceil(max([tt_zr['mjd'].value.max(),tt_zg['mjd'].value.max()]))

    days=np.arange(start,end)
    arr=np.zeros((3,len(days)))
    arr[0,:]=days
    ## arr[days,zr,zg]

    for i,day in enumerate(days):
        zr_cand=tt_zr[np.floor(tt_zr['mjd'].value)==day]
        zg_cand=tt_zg[np.floor(tt_zg['mjd'].value)==day]
        if len(zr_cand)==1 and len(zg_cand)==1:
            arr[1,i]=zr_cand['mag'].value
            arr[2,i]=zg_cand['mag'].value
            arr[0,i]=(zr_cand['mjd'].value+zg_cand['mjd'].value)/2.
        else:
            arr[1,i]=np.nan
            arr[2,i]=np.nan

    arr=arr[:,np.isnan(arr[1,:])==False]

    ts=TimeSeries(time=Time(arr[0],format='mjd'),data={'mag': arr[2]-arr[1]})
    ts_folded=ts.fold(period=force_period*u.day)

    plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=ts.time.jd , s=20)  #G-R
    plt.ylabel('G-R',fontsize=20)
    plt.xlabel('mjd',fontsize=20)

    return None

'''
def fit_mag(ztf_ID,plot=False):
    def gaussian1(x,c,mu,sigma):
        return c*np.exp(-(x-mu)**2/(2*sigma**2))

    def gaussian2(x,c1,mu1,sigma1,c2,mu2,sigma2):
        return c1 * np.exp(-(x-mu1)**2/(2*sigma1**2))+c2*np.exp(-(x-mu2)**2/(2*sigma2**2))

    def lorentzian(x,c,cen,wid):
        return c*wid**2/((x-cen)**2+wid**2)
    
    
    try:
        t=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+str(ztf_ID))
    except:
        print('2nd download attempt in 30 seconds...')
        time.sleep(30)
        try:
            t=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+str(ztf_ID))
        except:
            print('3rd download attempt in 30 seconds...')
            time.sleep(30)
            try:
                t=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+str(ztf_ID))
            except:
                print('last download attempt failed')
                    
    
    #remove bad flags
    t=t[t['catflags']==0]
    # remove bad days
    t=t[(t['mjd']<58481*u.day)|(t['mjd']>58482*u.day)]
    t=t[(t['mjd']<58472*u.day)|(t['mjd']>58474*u.day)]
    t=t[(t['mjd']<59032*u.day)|(t['mjd']>59033*u.day)]

    mag=np.array(t['mag'])

    histo=np.histogram(mag,bins=10,density=1)
    x=(histo[1][:-1]+histo[1][1:])/2
    y=histo[0]
    
    if plot:
        plt.hist(mag,bins=10,density=1,alpha=0.5)
        xx=np.linspace(x.min(),x.max(),50)
    
    # 1 gaussian
    try:
        popt_g1,pcov_g1 = curve_fit(gaussian1,x,y,p0=[1,x.mean(),x.std()])
        
        if plot:
            plt.plot(xx,gaussian1(xx,popt_g1[0],popt_g1[1],popt_g1[2]),label='gaussian')
            
        sqerr_gaussian1=((y-gaussian1(x,popt_g1[0],popt_g1[1],popt_g1[2]))**2).sum()
        
    except:
        sqerr_gaussian1=np.nan
        
    # 2 gaussian
    try:
        popt_g2,pcov_g2 = curve_fit(gaussian2,x,y,p0=[1,x.mean(),x.std(),1,x.mean(),x.std()])
        
        if plot:
            plt.plot(xx,gaussian2(xx,popt_g2[0],popt_g2[1],popt_g2[2],popt_g2[3],popt_g2[4],popt_g2[5]),label='2 gaussian')
        
        sqerr_gaussian2=((y-gaussian2(x,popt_g2[0],popt_g2[1],popt_g2[2],popt_g2[3],popt_g2[4],popt_g2[5]))**2).sum()
        
    except:
        sqerr_gaussian2=np.nan
        
    # lorentzian    
    try:    
        popt_l,pcov_l = curve_fit(lorentzian,x,y,p0=[1,x.mean(),x.std()])
        
        if plot:
            plt.plot(xx,lorentzian(xx,popt_l[0],popt_l[1],popt_l[2]),label='lorentzian')
        
        sqerr_lorentzian=((y-lorentzian(x,popt_l[0],popt_l[1],popt_l[2]))**2).sum()
        
    except:
        sqerr_lorentzian=np.nan

    if plot:
        plt.xlabel('magnitude')
        plt.ylabel('probability density')
        plt.gca().invert_xaxis()
        plt.legend()
    return sqerr_gaussian1,sqerr_gaussian2,sqerr_lorentzian
 '''

def fit_mag(ztf_ID,plot=True):
    g2_flag=False
    def gaussian1(x,c,mu,sigma):
        return c*np.exp(-(x-mu)**2/(2*sigma**2))

    def gaussian2(x,c1,mu1,sigma1,c2,mu2,sigma2):
        return c1 * np.exp(-(x-mu1)**2/(2*sigma1**2))+c2*np.exp(-(x-mu2)**2/(2*sigma2**2))

    def lorentzian(x,c,cen,wid):
        return c*wid**2/((x-cen)**2+wid**2)
    
    def gumbel(x,c,mu,beta):
        z=(x-mu)/beta
        return c/beta*np.exp(-(z+np.exp(-z)))
    
    try:
        t=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+str(ztf_ID))
    except:
        print('2nd download attempt in 30 seconds...')
        time.sleep(30)
        try:
            t=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+str(ztf_ID))
        except:
            print('3rd download attempt in 30 seconds...')
            time.sleep(30)
            try:
                t=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+str(ztf_ID))
            except:
                print('last download attempt failed')
                    
    
    #remove bad flags
    t=t[t['catflags']==0]
    # remove bad days
    t=t[(t['mjd']<58481*u.day)|(t['mjd']>58482*u.day)]
    t=t[(t['mjd']<58472*u.day)|(t['mjd']>58474*u.day)]
    t=t[(t['mjd']<59032*u.day)|(t['mjd']>59033*u.day)]

    mag=np.array(t['mag'])
    magerr=np.array(t['magerr']).mean()

    histo=np.histogram(mag,bins=10,density=1)
    x=(histo[1][:-1]+histo[1][1:])/2
    y=histo[0]
    
    if plot:
        plt.hist(mag,bins=10,density=1,alpha=0.5)
        xx=np.linspace(x.min(),x.max(),50)
    
    # 1 gaussian
    try:
        popt_g1,pcov_g1 = curve_fit(gaussian1,x,y,bounds=([0,x.min(),magerr*0.7],[50,x.max(),x.max()-x.min()]))
        
        if plot:
            plt.plot(xx,gaussian1(xx,popt_g1[0],popt_g1[1],popt_g1[2]),label='gaussian')
            
        sqerr_gaussian1=((y-gaussian1(x,popt_g1[0],popt_g1[1],popt_g1[2]))**2).sum()/(10-3)
        
    except:
        sqerr_gaussian1=99999
        
    # 2 gaussian
    try:
        popt_g2,pcov_g2 = curve_fit(gaussian2,x,y,bounds=([0,x.min(),magerr*0.7,0,x.min(),magerr*0.5],[50,x.max(),x.max()-x.min(),50,x.max(),x.max()-x.min()]))
        if plot:
            plt.plot(xx,gaussian2(xx,popt_g2[0],popt_g2[1],popt_g2[2],popt_g2[3],popt_g2[4],popt_g2[5]),label='2 gaussian')
        
        sqerr_gaussian2=((y-gaussian2(x,popt_g2[0],popt_g2[1],popt_g2[2],popt_g2[3],popt_g2[4],popt_g2[5]))**2).sum()/(10-6)
        g2_flag=(popt_g2[2]+popt_g2[5])>np.abs(popt_g2[1]-popt_g2[4])
    except:
        sqerr_gaussian2=99999
        
    # lorentzian    
    try:    
        popt_gum,pcov_gum = curve_fit(gumbel,x,y,p0=[1,x.mean(),1],bounds=([-50,x.min()-magerr,-100],[50,x.max()+magerr,100]))
        
        if plot:
            plt.plot(xx,gumbel(xx,popt_gum[0],popt_gum[1],popt_gum[2]),label='gumbel')
        
        sqerr_gumbel=((y-gumbel(x,popt_gum[0],popt_gum[1],popt_gum[2]))**2).sum()/(10-3)
        
    except:
        sqerr_gumbel=99999

    if plot:
        plt.xlabel('magnitude')
        plt.ylabel('probability density')
        plt.gca().invert_xaxis()
        plt.legend()
    return sqerr_gaussian1,sqerr_gaussian2,sqerr_gumbel,g2_flag


def counterpart(ra,dec,t):
    c1 = SkyCoord(ra, dec,unit="deg")
    c2 = SkyCoord(t['RA'], t['DEC'],unit="deg")
    print(c1.separation(c2).min().to(u.arcsec))
    return t[c1.separation(c2).argmin()]