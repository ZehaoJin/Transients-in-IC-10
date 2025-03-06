from astropy.io import fits
import numpy as np
from astropy.table import QTable,vstack
import astropy.units as u
import astropy.utils
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.lines as mlines
from scipy.stats import chisquare
from scipy.stats import chi2
from scipy.stats import lognorm
from scipy.stats import pearsonr
from astropy.timeseries import LombScargle
import time
from astropy.visualization import hist
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from astropy.timeseries import BoxLeastSquares
from astropy.timeseries import TimeSeries
from astropy.time import Time
#import george
#from george.kernels import ExpSquaredKernel
from scipy.optimize import minimize
from tqdm import tqdm


# t_zr=QTable.read('t_zr_20240401.ecsv')
# t_zg=QTable.read('t_zg_20240401.ecsv')

t_zr=QTable.read('t_zr_20250225.ecsv')
t_zg=QTable.read('t_zg_20250225.ecsv')


def draw_lightcurve(SourceID):
    plt.figure(figsize=(20,15))
    #plt.title('Source ID = '+str(SourceID),fontsize=40)
    #rbandcolor=['red','salmon','maroon']
    #gbandcolor=['green','limegreen','seagreen']
    # rbandcolor=['red','black','maroon']
    # gbandcolor=['green','blue','seagreen']
    # ubandcolor=['violet']
    rbandcolor='red'
    gbandcolor='green'
    fmt=['x','o','s']
    rmarker=[]
    gmarker=[]
    #labels = ['Source ID = '+str(SourceID),'r band', 'g band']
    labels = ['Source ID = '+str(SourceID)]


    OIDs=t_zr[t_zr['SourceID']==SourceID]['OID']
    for i,OID in enumerate(OIDs):
        OID_str=str(int(OID))
        try:
            t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        except:
            time.sleep(1)
            t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        
        #remove the bad one day
        t_ztf=t_ztf[t_ztf['catflags']==0]
        
        t_ztf=t_ztf[(t_ztf['mjd']<58481*u.day)|(t_ztf['mjd']>58482*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<58472*u.day)|(t_ztf['mjd']>58474*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<59032*u.day)|(t_ztf['mjd']>59033*u.day)]

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

        marker = plt.errorbar(np.array(t_ztf['mjd']),np.array(t_ztf['mag']),yerr=np.array(t_ztf['magerr']),fmt=fmt[i],ecolor='grey',capsize=4,label='r band',markeredgecolor=rbandcolor,markersize=10,fillstyle='none')
        rmarker.append(marker)
    if len(OIDs)!=0:
        labels.append('r band')

    OIDs=t_zg[t_zg['SourceID']==SourceID]['OID']
    for i,OID in enumerate(OIDs):
        OID_str=str(int(OID))
        try:
            t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        except:
            time.sleep(1)
            t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        t_ztf=t_ztf[t_ztf['catflags']==0]

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

        marker = plt.errorbar(np.array(t_ztf['mjd']),np.array(t_ztf['mag']),yerr=np.array(t_ztf['magerr']),fmt=fmt[i],ecolor='grey',capsize=4,label='g band',markeredgecolor=gbandcolor,markersize=10,fillstyle='none')
        gmarker.append(marker)
    if len(OIDs)!=0:
        labels.append('g band')


    plt.xlabel('modified Julian days',fontsize=40)
    plt.ylabel('mag',fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    #plt.legend(fontsize=40)
    handles = [mlines.Line2D([], [], color='none'),tuple(rmarker), tuple(gmarker)]
    plt.legend(
        handles=handles,
        labels=labels,
        fontsize=40,
        handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)}
    )
    plt.gca().invert_yaxis()
    return None


def draw_lightcurve_OID(OID,band='r'):
    plt.figure(figsize=(20,15))
    plt.title('OID = '+str(OID),fontsize=40)

    OID_str=str(int(OID))
    t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
    
    if band=='r':
        #remove the bad one day
        t_ztf=t_ztf[t_ztf['catflags']==0]
        
        t_ztf=t_ztf[(t_ztf['mjd']<58481*u.day)|(t_ztf['mjd']>58482*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<58472*u.day)|(t_ztf['mjd']>58474*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<59032*u.day)|(t_ztf['mjd']>59033*u.day)]

    # up to DR15, Nov 2022, MJD 59892
    t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

    plt.errorbar(np.array(t_ztf['mjd']),np.array(t_ztf['mag']),yerr=np.array(t_ztf['magerr']),fmt='xb',ecolor='grey',capsize=2,label=band+' band')


    plt.xlabel('modified Julian days',fontsize=40)
    plt.ylabel('mag',fontsize=40)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.gca().invert_yaxis()
    return None


def draw_lightcurve_with_color(SourceID):
    plt.figure(figsize=(20,20))
    #plt.suptitle('Source ID = '+str(SourceID), fontsize=30, y=0.925)
    ax2=plt.subplot(3,1,3)
    ax1=plt.subplot(3,1,(1,2),sharex=ax2)
    plt.subplots_adjust(hspace=0)

    # rbandcolor=['red','black','maroon']
    # gbandcolor=['green','blue','seagreen']
    # ubandcolor=['violet']
    rbandcolor='red'
    gbandcolor='green'
    fmt=['x','o','s']
    rmarker=[]
    gmarker=[]
    labels=['Source ID = '+str(SourceID)]

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

    # up to DR15, Nov 2022, MJD 59892
    tt_zr=tt_zr[tt_zr['mjd']<59892*u.day]
    tt_zg=tt_zg[tt_zg['mjd']<59892*u.day]

    # tt_zr=tt_zr[tt_zr['mjd']<58750*u.day]
    # tt_zg=tt_zg[tt_zg['mjd']<58750*u.day]

    start=np.floor(min([tt_zr['mjd'].value.min(),tt_zg['mjd'].value.min()]))
    end=np.ceil(max([tt_zr['mjd'].value.max(),tt_zg['mjd'].value.max()]))

    days=np.arange(start,end)
    arr=np.zeros((5,len(days)))
    arr[0,:]=days
    ## arr[days,zr,zg]

    for i,day in enumerate(days):
        zr_cand=tt_zr[np.floor(tt_zr['mjd'].value)==day]
        zg_cand=tt_zg[np.floor(tt_zg['mjd'].value)==day]
        if len(zr_cand)==1 and len(zg_cand)==1:
            arr[1,i]=zr_cand['mag'].value
            arr[2,i]=zg_cand['mag'].value
            arr[3,i]=zr_cand['magerr'].value
            arr[4,i]=zg_cand['magerr'].value
        else:
            arr[1,i]=np.nan
            arr[2,i]=np.nan
            arr[3,i]=np.nan
            arr[4,i]=np.nan

    arr=arr[:,np.isnan(arr[1,:])==False]

    if SourceID==90:
        color_lim=(arr[2]-arr[1])<1.2
        arr=arr[:,color_lim]
        
    OIDs=t_zr[t_zr['SourceID']==SourceID]['OID']
    for i,OID in enumerate(OIDs):
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        
        #remove the bad one day
        t_ztf=t_ztf[t_ztf['catflags']==0]
        
        t_ztf=t_ztf[(t_ztf['mjd']<58481*u.day)|(t_ztf['mjd']>58482*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<58472*u.day)|(t_ztf['mjd']>58474*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<59032*u.day)|(t_ztf['mjd']>59033*u.day)]

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

        marker = ax1.errorbar(np.array(t_ztf['mjd']),np.array(t_ztf['mag']),yerr=np.array(t_ztf['magerr']),fmt=fmt[i],ecolor='grey',capsize=4,label='r band',markeredgecolor=rbandcolor,markersize=10,fillstyle='none')
        rmarker.append(marker)
    if len(OIDs)!=0:
        labels.append('r band')

    OIDs=t_zg[t_zg['SourceID']==SourceID]['OID']
    for i,OID in enumerate(OIDs):
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        t_ztf=t_ztf[t_ztf['catflags']==0]

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

        marker = ax1.errorbar(np.array(t_ztf['mjd']),np.array(t_ztf['mag']),yerr=np.array(t_ztf['magerr']),fmt=fmt[i],ecolor='grey',capsize=4,label='g band',markeredgecolor=gbandcolor,markersize=10,fillstyle='none')
        gmarker.append(marker)
    if len(OIDs)!=0:
        labels.append('g band')

    ax2.errorbar(arr[0],arr[2]-arr[1],yerr=np.sqrt(arr[3]**2+arr[4]**2),fmt='o',ecolor='grey',capsize=2,label='g-r',markersize=10)
    if SourceID==90:
        ax2.axvline(58750,linestyle='--',color='red')
        ax1.axvline(58750,linestyle='--',color='red')
    ax2.set_xlabel('modified Julian days',fontsize=40)
    ax1.set_ylabel('mag',fontsize=40)
    ax2.set_ylabel(r'$g-r$',fontsize=40)
    ax1.tick_params(labelbottom=False)
    ax1.tick_params(axis='y', labelsize=30)
    ax2.tick_params(axis='both', labelsize=30)
    #ax1.legend(fontsize=30)
    
    handles = [mlines.Line2D([], [], color='none'),tuple(rmarker), tuple(gmarker)]
    ax1.legend(
        handles=handles,
        labels=labels,
        fontsize=30,
        handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)}
    )
    ax1.invert_yaxis()
    #ax2.invert_yaxis()
    return None



def draw_fold_lightcurve(SourceID,force_period=None):
    fs=20
    plt.figure(figsize=(20,15))
    plt.suptitle('Source ID = '+str(SourceID),fontsize=fs)


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

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

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

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

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
    plt.suptitle('Source ID = '+str(SourceID),fontsize=fs)


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

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

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

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

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


def sinfit_single(SourceID,force_period=None,max_light=False, print_param=False, plot_fit_r=True, plot_fit_g=True):
    fs=20
    def sinfunc(t, A, w, p, c):
        return A * np.sin(w*t + p) + c
    plt.figure(figsize=(20,7.5))
    #plt.suptitle('Source ID = '+str(SourceID),fontsize=fs, y=0.93)


    OIDs_r=t_zr[t_zr['SourceID']==SourceID]
    OIDs_g=t_zg[t_zg['SourceID']==SourceID]
    length=1
    
    if len(OIDs_r)!=0:
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

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]


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
        perr = np.sqrt(np.diag(pcov))
        f = w/(2.*np.pi)
        fit_period=1./f
        print('L-S_period:',guess_period)
        print('fit_period:',fit_period,'±',(2. * np.pi / w**2) * perr[1])
        tt=np.linspace(ts_folded.time.jd.min(),ts_folded.time.jd.max(),100)
        yy=A*np.sin(w*tt+p)+c
        y=A*np.sin(w*ts_folded.time.jd+p)+c

        sin_chi2=(((ts_folded['mag'].value-y)**2)/(ts_folded['magerr'].value**2)).sum()/(len(ts_folded)-1)
        print('sin_chi2:',sin_chi2)


        plt.subplot(1,2,1)
        plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=t_ztf['mjd'].value,s=20,label='R band  OID = '+str(OID))
        if plot_fit_r==True:
            plt.plot(tt, yy, c='r', label='sinusoidal fit, period = {:.1f} days'.format(guess_period))
        plt.legend(fontsize=fs*0.8)
        #plt.colorbar()
        #colorbar tick size
        plt.colorbar().ax.tick_params(labelsize=fs*0.8)
        plt.xlabel('Time (days)',fontsize=fs*1.2)
        plt.ylabel(r'$m_r$',fontsize=fs*1.3)
        #ticksize
        plt.xticks(fontsize=fs*0.8)
        plt.yticks(fontsize=fs*0.8)
        plt.gca().invert_yaxis()
        #plt.title('R band  OID = '+str(OID),fontsize=fs*1.0)

        max_r=yy.max()
        min_r=yy.min()

        if print_param==True:
            print('A:',A,'±',perr[0])
            print('w:',w,'±',perr[1])
            print('p:',p,'±',perr[2])
            print('c:',c,'±',perr[3])
            print('')
    
    if len(OIDs_g)!=0:
        if len(OIDs_g)==1:
            OID=OIDs_g['OID']
        else:
            OID=OIDs_g[OIDs_g['numobs'].argmax()]['OID']
    
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        t_ztf=t_ztf[t_ztf['catflags']==0]

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

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
        perr = np.sqrt(np.diag(pcov))
        f = w/(2.*np.pi)
        fit_period=1./f
        print('L-S_period:',guess_period)
        print('fit_period:',fit_period, '±',(2. * np.pi / w**2) * perr[1])
        tt=np.linspace(ts_folded.time.jd.min(),ts_folded.time.jd.max(),100)
        yy=A*np.sin(w*tt+p)+c
        y=A*np.sin(w*ts_folded.time.jd+p)+c

        sin_chi2=(((ts_folded['mag'].value-y)**2)/(ts_folded['magerr'].value**2)).sum()/(len(ts_folded)-1)
        print('sin_chi2:',sin_chi2)

        plt.subplot(1,2,2)
        plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=t_ztf['mjd'].value, s=20,label='G band  OID = '+str(OID))
        if plot_fit_g==True:
            plt.plot(tt, yy, c='r', label='sinusoidal fit, period = {:.1f} days'.format(guess_period))
        plt.legend(fontsize=fs*0.8)
        #plt.colorbar()
        #colorbar tick size
        plt.colorbar().ax.tick_params(labelsize=fs*0.8)
        plt.xlabel('Time (days)',fontsize=fs*1.2)
        plt.ylabel(r'$m_g$',fontsize=fs*1.3)
        #ticksize
        plt.xticks(fontsize=fs*0.8)
        plt.yticks(fontsize=fs*0.8)
        plt.gca().invert_yaxis()
        #plt.title('G band  OID = '+str(OID),fontsize=fs*1.0)

        max_g=yy.max()
        min_g=yy.min()

        if print_param==True:
            print('A:',A,'±',perr[0])
            print('w:',w,'±',perr[1])
            print('p:',p,'±',perr[2])
            print('c:',c,'±',perr[3])
            print('')

    if max_light==True:
        if len(OIDs_r)==0:
            max_r=-999
            min_r=-999
        if len(OIDs_g)==0:
            max_g=-999
            min_g=-999
        return max_r,max_g,min_r,min_g
    else:
        return None
    
def sinfit_fix_period(SourceID,force_period=None,max_light=False, print_param=False, plot_fit_r=True, plot_fit_g=True):
    fs=20
    
    plt.figure(figsize=(20,7.5))
    plt.suptitle('Source ID = '+str(SourceID),fontsize=fs)


    OIDs_r=t_zr[t_zr['SourceID']==SourceID]
    OIDs_g=t_zg[t_zg['SourceID']==SourceID]
    length=1
    
    if len(OIDs_r)!=0:
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

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]


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
        guess = np.array([guess_amp, 0., guess_offset])
        #convert guess period to w
        w = 2.*np.pi / guess_period

        def sinfunc(t, A, p, c):
            return A * np.sin(w*t + p) + c

        popt, pcov =curve_fit(sinfunc,ts_folded.time.jd,ts_folded['mag'],p0=guess)
        A, p, c = popt
        perr = np.sqrt(np.diag(pcov))
        tt=np.linspace(ts_folded.time.jd.min(),ts_folded.time.jd.max(),100)
        yy=A*np.sin(w*tt+p)+c
        y=A*np.sin(w*ts_folded.time.jd+p)+c

        sin_chi2=(((ts_folded['mag'].value-y)**2)/(ts_folded['magerr'].value**2)).sum()/(len(ts_folded)-1)
        print('sin_chi2:',sin_chi2)


        plt.subplot(1,2,1)
        plt.scatter(ts_folded.time.jd/guess_period*2*np.pi, ts_folded['mag'],c=ts.time.jd ,s=4)
        if plot_fit_r==True:
            plt.plot(tt/guess_period*2*np.pi,yy,c='r',label='sinusoidal fit')
        plt.legend()
        plt.colorbar()
        plt.xlabel('phase',fontsize=fs)
        plt.ylabel('mag',fontsize=fs)
        plt.xticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'])
        plt.gca().invert_yaxis()
        plt.title('R band  OID = '+str(OID),fontsize=fs*1.0)

        max_r=yy.max()
        min_r=yy.min()

        if print_param==True:
            print('Period',guess_period)
            print('A:',A,'±',perr[0])
            print('w:',w)
            print('p:',p,'±',perr[1])
            print('c:',c,'±',perr[2])
            print('t0:',t_ztf['mjd'][0])
            print('')
    
    if len(OIDs_g)!=0:
        if len(OIDs_g)==1:
            OID=OIDs_g['OID']
        else:
            OID=OIDs_g[OIDs_g['numobs'].argmax()]['OID']
    
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        t_ztf=t_ztf[t_ztf['catflags']==0]

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

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
        guess = np.array([guess_amp, 0., guess_offset])
        #convert guess period to w
        w = 2.*np.pi / guess_period

        def sinfunc(t, A, p, c):
            return A * np.sin(w*t + p) + c


        popt, pcov =curve_fit(sinfunc,ts_folded.time.jd,ts_folded['mag'],p0=guess)
        A, p, c = popt
        perr = np.sqrt(np.diag(pcov))
        tt=np.linspace(ts_folded.time.jd.min(),ts_folded.time.jd.max(),100)
        yy=A*np.sin(w*tt+p)+c
        y=A*np.sin(w*ts_folded.time.jd+p)+c

        sin_chi2=(((ts_folded['mag'].value-y)**2)/(ts_folded['magerr'].value**2)).sum()/(len(ts_folded)-1)
        print('sin_chi2:',sin_chi2)

        plt.subplot(1,2,2)
        plt.scatter(ts_folded.time.jd/guess_period*2*np.pi, ts_folded['mag'],c=ts.time.jd , s=4)
        if plot_fit_g==True:
            plt.plot(tt/guess_period*2*np.pi,yy,c='r',label='sinusoidal fit')
        plt.legend()
        plt.colorbar()
        plt.xlabel('phase',fontsize=fs)
        plt.ylabel('mag',fontsize=fs)
        plt.xticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'])
        plt.gca().invert_yaxis()
        plt.title('G band  OID = '+str(OID),fontsize=fs*1.0)

        max_g=yy.max()
        min_g=yy.min()

        if print_param==True:
            print('Period',guess_period)
            print('A:',A,'±',perr[0])
            print('w:',w)
            print('p:',p,'±',perr[1])
            print('c:',c,'±',perr[2])
            print('t0:',t_ztf['mjd'][0])
            print('')

    if max_light==True:
        if len(OIDs_r)==0:
            max_r=-999
            min_r=-999
        if len(OIDs_g)==0:
            max_g=-999
            min_g=-999
        return max_r,max_g,min_r,min_g
    else:
        return None


def sinfit_fix_period_absolute(SourceID,force_period=None,max_light=False, print_param=False, plot_fit_r=True, plot_fit_g=True):
    fs=20
    
    E_BV = 0.98
    Av=3.1*E_BV
    Ag=1.19*Av
    Ar=0.834*Av
    IC10_distant_modulus = 24

    t0=58254.49

    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(10, 15), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle('Source ID = '+str(SourceID), fontsize=fs, y=0.91)

    OIDs_r=t_zr[t_zr['SourceID']==SourceID]
    OIDs_g=t_zg[t_zg['SourceID']==SourceID]
    length=1
    
    if len(OIDs_r)!=0:
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

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]


        #ts=TimeSeries(time=Time(t_ztf['mjd'],format='mjd'),data={'mag': t_ztf['mag']-Ar-IC10_distant_modulus,'magerr':t_ztf['magerr']})
        ts_folded={'time':t_ztf['mjd'].value,'mag':t_ztf['mag'].value-Ar-IC10_distant_modulus,'magerr':t_ztf['magerr'].value}
        if force_period!=None:
            ts_folded['time']=(ts_folded['time']-t0)%force_period
            guess_period=force_period
        else:
            ts_folded['time']=(ts_folded['time']-t0)%(t_zr[t_zr['OID']==OID]['period'].value)
            guess_period=t_zr[t_zr['OID']==OID]['period'].value

        guess_freq = 1./guess_period
        guess_amp = np.std(ts_folded['mag']) * 2.**0.5
        guess_offset = np.mean(ts_folded['mag'])
        guess = np.array([guess_amp, 0., guess_offset])
        #convert guess period to w
        w = 2.*np.pi / guess_period

        def sinfunc(t, A, p, c):
            return A * np.sin(w*t + p) + c

        popt, pcov =curve_fit(sinfunc,ts_folded['time'],ts_folded['mag'],p0=guess,bounds=([0,-np.pi,-np.inf],[np.inf,np.pi,np.inf]))
        A, p, c = popt
        perr = np.sqrt(np.diag(pcov))
        tt=np.linspace(ts_folded['time'].min(),ts_folded['time'].max(),100)
        yy=A*np.sin(w*tt+p)+c
        y=A*np.sin(w*ts_folded['time']+p)+c

        sin_chi2=(((ts_folded['mag']-y)**2)/(ts_folded['magerr']**2)).sum()/(len(ts_folded['mag'])-1)
        print('sin_chi2:',sin_chi2)


        sc2 = ax2.scatter(ts_folded['time']/guess_period*2*np.pi, ts_folded['mag'],c=t_ztf['mjd'].value,s=20,label='OID = '+str(OID))
        if plot_fit_r==True:
            ax2.plot(tt/guess_period*2*np.pi,yy,c='r',label='sinusoidal fit')
        ax2.legend(fontsize=fs)
        ax2.set_xlabel(r'$\phi$',fontsize=fs)
        ax2.set_ylabel(r'$M_r$',fontsize=fs)
        ax2.set_xticks([0,0.5*np.pi,np.pi,1.5*np.pi,2*np.pi])
        ax2.set_xticklabels(['0','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$'])
        ax2.tick_params(axis='both', which='major', labelsize=fs*0.9)

        
        ax2.invert_yaxis()

        max_r=yy.max()
        min_r=yy.min()

        if print_param==True:
            print('Period',guess_period)
            print('A:',A,'±',perr[0])
            print('w:',w)
            print('p:',p,'±',perr[1])
            print('c:',c,'±',perr[2])
            print('t0:',t_ztf['mjd'][0])
            print('')
    
    if len(OIDs_g)!=0:
        if len(OIDs_g)==1:
            OID=OIDs_g['OID']
        else:
            OID=OIDs_g[OIDs_g['numobs'].argmax()]['OID']
    
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        t_ztf=t_ztf[t_ztf['catflags']==0]

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

        #ts=TimeSeries(time=Time(t_ztf['mjd'],format='mjd'),data={'mag': t_ztf['mag']-Ag-IC10_distant_modulus,'magerr':t_ztf['magerr']})
        ts_folded={'time':t_ztf['mjd'].value,'mag':t_ztf['mag'].value-Ar-IC10_distant_modulus,'magerr':t_ztf['magerr'].value}
        if force_period!=None:
            ts_folded['time']=(ts_folded['time']-t0)%force_period
            guess_period=force_period
        else:
            ts_folded['time']=(ts_folded['time']-t0)%(t_zg[t_zg['OID']==OID]['period'].value)
            guess_period=t_zg[t_zg['OID']==OID]['period'].value

        guess_freq = 1./guess_period
        guess_amp = np.std(ts_folded['mag']) * 2.**0.5
        guess_offset = np.mean(ts_folded['mag'])
        guess = np.array([guess_amp, 0., guess_offset])
        #convert guess period to w
        w = 2.*np.pi / guess_period

        def sinfunc(t, A, p, c):
            return A * np.sin(w*t + p) + c


        popt, pcov =curve_fit(sinfunc,ts_folded['time'],ts_folded['mag'],p0=guess,bounds=([0,-np.pi,-np.inf],[np.inf,np.pi,np.inf]))
        A, p, c = popt
        perr = np.sqrt(np.diag(pcov))
        tt=np.linspace(ts_folded['time'].min(),ts_folded['time'].max(),100)
        yy=A*np.sin(w*tt+p)+c
        y=A*np.sin(w*ts_folded['time']+p)+c

        sin_chi2=(((ts_folded['mag']-y)**2)/(ts_folded['magerr']**2)).sum()/(len(ts_folded['mag'])-1)
        print('sin_chi2:',sin_chi2)

        sc1 = ax1.scatter(ts_folded['time']/guess_period*2*np.pi, ts_folded['mag'],c=t_ztf['mjd'].value,s=20,label='OID = '+str(OID))
        if plot_fit_g==True:
            ax1.plot(tt/guess_period*2*np.pi,yy,c='r',label='sinusoidal fit')
        ax1.legend(fontsize=fs)
        ax1.set_xlabel(r'$\phi$',fontsize=fs)
        ax1.set_ylabel(r'$M_g$',fontsize=fs)
        # set y tick font size
        ax1.tick_params(axis='both', which='major', labelsize=fs*0.9)
        #ax1.set_xticks([0,0.5*np.pi,np.pi,1.5*np.pi,2*np.pi])
        #ax1.set_xticklabels(['0','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$'])
        ax1.invert_yaxis()

        max_g=yy.max()
        min_g=yy.min()

        if print_param==True:
            print('Period',guess_period)
            print('A:',A,'±',perr[0])
            print('w:',w)
            print('p:',p,'±',perr[1])
            print('c:',c,'±',perr[2])
            print('t0:',t_ztf['mjd'][0])
            print('')

    # Create a single colorbar for both subplots
    cbar = fig.colorbar(sc1, ax=[ax1, ax2], orientation='horizontal', pad=0.07)
    cbar.set_label('MJD', fontsize=fs)
    cbar.ax.tick_params(labelsize=fs*0.8)

    if max_light==True:
        if len(OIDs_r)==0:
            max_r=-999
            min_r=-999
        if len(OIDs_g)==0:
            max_g=-999
            min_g=-999
        return max_r,max_g,min_r,min_g
    else:
        return None
    

def sinfit_fix_period_OID(OID,force_period=None, print_param=False):
    plt.figure(figsize=(12,6))
    fs=20
    
    OID=str(int(OID))
    entry = t_zr[t_zr['OID']==OID]
    if len(entry)==0:
        entry = t_zg[t_zg['OID']==OID]
    
    SourceID = entry['SourceID'][0]
    band = entry['filter'][0]

    if force_period == None:
        period = entry['period'][0]
    else:
        period = force_period

    # print('SourceID:',SourceID)
    # print('band:',band)
    
    try:
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID)
    except:
        try:
            time.sleep(1)
            t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID)
        except:
            time.sleep(1)
            t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID)
        
            

    if band=='zr':
        #remove the bad one day
        t_ztf=t_ztf[t_ztf['catflags']==0]
        
        t_ztf=t_ztf[(t_ztf['mjd']<58481*u.day)|(t_ztf['mjd']>58482*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<58472*u.day)|(t_ztf['mjd']>58474*u.day)]
        t_ztf=t_ztf[(t_ztf['mjd']<59032*u.day)|(t_ztf['mjd']>59033*u.day)]

    # up to DR15, Nov 2022, MJD 59892
    t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]


    ts=TimeSeries(time=Time(t_ztf['mjd'],format='mjd'),data={'mag': t_ztf['mag'],'magerr':t_ztf['magerr']})

    ts_folded=ts.fold(period=period*u.day)

    freq = 1./period
    amp = np.std(ts_folded['mag'].value) * 2.**0.5
    offset = np.mean(ts_folded['mag'].value)
    guess = np.array([amp, 0., offset])
    #convert guess period to w
    w = 2.*np.pi / period

    def sinfunc(t, A, p, c):
        return A * np.sin(w*t + p) + c

    popt, pcov =curve_fit(sinfunc,ts_folded.time.jd,ts_folded['mag'],p0=guess)
    A, p, c = popt
    perr = np.sqrt(np.diag(pcov))
    tt=np.linspace(ts_folded.time.jd.min(),ts_folded.time.jd.max(),100)
    yy=A*np.sin(w*tt+p)+c
    y=A*np.sin(w*ts_folded.time.jd+p)+c

    sin_chi2=(((ts_folded['mag'].value-y)**2)/(ts_folded['magerr'].value**2)).sum()/(len(ts_folded)-1)
    print('sin_chi2:',sin_chi2)


    plt.scatter(ts_folded.time.jd, ts_folded['mag'],c=t_ztf['mjd'].value,s=20,label='Source '+str(SourceID)+r', $'+band[-1]+'$ band, OID = '+OID)
    plt.plot(tt, yy, c='r', label='sinusoidal fit, period = {:.1f} days'.format(period))
    plt.legend(fontsize=fs*0.8)
    #colorbar tick size
    plt.colorbar().ax.tick_params(labelsize=fs*0.8)
    plt.xlabel('Time (days)',fontsize=fs*1.2)
    #plt.ylabel(r'$m_r$',fontsize=fs*1.3)
    plt.ylabel(r'$m_'+band[-1]+'$',fontsize=fs*1.3)
    #ticksize
    plt.xticks(fontsize=fs*0.8)
    plt.yticks(fontsize=fs*0.8)
    plt.gca().invert_yaxis()

    if print_param==True:
        print('Period',period)
        print('A:',A,'±',perr[0])
        print('w:',w)
        print('p:',p,'±',perr[1])
        print('c:',c,'±',perr[2])
        print('t0:',t_ztf['mjd'][0])
        print('')
    


def sinfit2(SourceID,force_period=None):
    def sinfunc(t, A, w, p, c):
        return A * np.sin(w*t + p) + c

    plt.figure(figsize=(20,15))
    plt.suptitle('Source ID = '+str(SourceID))
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

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

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

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

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




def windowfunc(ztf_ID,high=2800,low=0.1,nan=False,nterms=1,samples_per_peak=5,return_period=False):
    t=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+str(ztf_ID))
    #remove bad days
    t=t[t['catflags']==0]
    
    t=t[(t['mjd']<58481*u.day)|(t['mjd']>58482*u.day)]
    t=t[(t['mjd']<58472*u.day)|(t['mjd']>58474*u.day)]
    t=t[(t['mjd']<59032*u.day)|(t['mjd']>59033*u.day)]

    # up to DR15, Nov 2022, MJD 59892
    t=t[t['mjd']<59892*u.day]

    con_mag=np.zeros(t['mag'].shape)+t['mag'].mean()
    con_err=np.zeros(t['magerr'].shape)+t['magerr'].mean()

    if nan==False:
        ls = LombScargle(t['mjd'],con_mag, con_err)
    else:
        # Probably need this if you see NaN
        ls = LombScargle(t['mjd'],con_mag, con_err,fit_mean=False, center_data=False)


    freq, con_power = ls.autopower(nyquist_factor=300,minimum_frequency=1/(high*u.day),maximum_frequency=1/(low*u.day),samples_per_peak=samples_per_peak)
    #LNP_SIG=ls.false_alarm_probability(power.max())
    plt.plot(1/freq, con_power,label='window function',alpha=0.5)
    plt.xlabel('period (days)')
    plt.ylabel('power')
    #plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")

    ls = LombScargle(t['mjd'],t['mag'], t['magerr'],nterms=nterms)
    freq, power = ls.autopower(nyquist_factor=300,minimum_frequency=1/(high*u.day),maximum_frequency=1/(low*u.day),samples_per_peak=samples_per_peak)
    if nterms==1:
        LNP_SIG=ls.false_alarm_probability(power.max())
    plt.plot(1/freq, power,label='L-S',alpha=0.5)
    plt.legend()
    if nterms==1:
        print('period=',1/freq[power.argmax()],'    LNP_SIG=',ls.false_alarm_probability(power.max()))
    else:
        print('period=',1/freq[power.argmax()])
    plt.vlines(1/freq[power.argmax()].value,1e-3,10,colors='r',linestyles='dashed')

    # plot power/con_power on the right y axis
    #plt.twinx()
    #plt.plot(1/freq, power/con_power,label='L-S/window',alpha=0.5,color='r')
    #plt.ylabel('L-S/window')
    #plt.legend(loc='lower right')

    if return_period==True:
        return 1/freq[power.argmax()].value
    else:
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

    # up to DR15, Nov 2022, MJD 59892
    tt_zr=tt_zr[tt_zr['mjd']<59892*u.day]
    tt_zg=tt_zg[tt_zg['mjd']<59892*u.day]

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

def color2(SourceID,band='r',ployfit=False,c0=18.5):
    plt.figure(figsize=(15,10))
    plt.suptitle('Source ID = '+str(SourceID), fontsize=20, y=0.925)
    ax2=plt.subplot(4,1,4)
    ax1=plt.subplot(4,1,(1,3),sharex=ax2)
    plt.subplots_adjust(hspace=0)


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

    # up to DR15, Nov 2022, MJD 59892
    tt_zr=tt_zr[tt_zr['mjd']<59892*u.day]
    tt_zg=tt_zg[tt_zg['mjd']<59892*u.day]

    if SourceID==90:
        tt_zr=tt_zr[tt_zr['mjd']>58750*u.day]
        tt_zg=tt_zg[tt_zg['mjd']>58750*u.day]
    

    start=np.floor(min([tt_zr['mjd'].value.min(),tt_zg['mjd'].value.min()]))
    end=np.ceil(max([tt_zr['mjd'].value.max(),tt_zg['mjd'].value.max()]))

    days=np.arange(start,end)
    arr=np.zeros((5,len(days)))
    arr[0,:]=days
    ## arr[days,zr,zg]

    for i,day in enumerate(days):
        zr_cand=tt_zr[np.floor(tt_zr['mjd'].value)==day]
        zg_cand=tt_zg[np.floor(tt_zg['mjd'].value)==day]
        if len(zr_cand)==1 and len(zg_cand)==1:
            arr[1,i]=zr_cand['mag'].value
            arr[2,i]=zg_cand['mag'].value
            arr[3,i]=zr_cand['magerr'].value
            arr[4,i]=zg_cand['magerr'].value
        else:
            arr[1,i]=np.nan
            arr[2,i]=np.nan
            arr[3,i]=np.nan
            arr[4,i]=np.nan

    arr=arr[:,np.isnan(arr[1,:])==False]

    if SourceID==90:
        color_lim=(arr[2]-arr[1])<1.2
        arr=arr[:,color_lim]

    #plt.scatter(arr[1],arr[2]-arr[1])  #G-R
    if band=='r':
        ax1.errorbar(arr[1],arr[2]-arr[1],xerr=arr[3],yerr=np.sqrt(arr[3]**2+arr[4]**2),fmt='o',ecolor='grey',capsize=2)
        ax1.set_xlabel(r'$r$',fontsize=20)
        
        if ployfit==True:
            # fit with a polynomial fit g-r=a*(r-c0)^2+b*(r-c0)+d, considering the error in r and g-r
            popt, pcov =curve_fit(lambda x, a, b, d: a*(x-c0)**2+b*(x-c0)+d,arr[1],arr[2]-arr[1],sigma=np.sqrt(arr[3]**2+arr[4]**2))
            a,b,d=popt
            xx=np.linspace(arr[1].min(),arr[1].max(),100)
            ax1.plot(xx,a*(xx-c0)**2+b*(xx-c0)+d,c='r',label=r'$g-r={:.2f}(r-{:.2f})^2+{:.2f}(r-{:.2f})+{:.2f}$'.format(a,c0,b,c0,d),alpha=0.5,linestyle='--')

        
    if band=='g':
        ax1.errorbar(arr[2],arr[2]-arr[1],xerr=arr[4],yerr=np.sqrt(arr[3]**2+arr[4]**2),fmt='none',ecolor='grey',capsize=2,alpha=0.5)
        ax1.scatter(arr[2],arr[2]-arr[1],c=arr[0])
        #plt.colorbar()
        ax2.set_xlabel(r'$g$',fontsize=20)
        ax1.set_ylabel(r'$g-r$',fontsize=20)
        # x tick size
        ax1.tick_params(labelbottom=False)
        ax1.tick_params(axis='y', labelsize=20)
        ax2.tick_params(axis='both', labelsize=20)
        # colorbar
        cbar = plt.colorbar(ax1.scatter(arr[2],arr[2]-arr[1],c=arr[0]), ax=(ax1, ax2))

        if ployfit==True:
            # fit with a polynomial fit g-r=a*(g-c0)^2+b*(g-c0)+d, considering the error in g and g-r
            popt, pcov =curve_fit(lambda x, a, b, d: a*(x-c0)**2+b*(x-c0)+d,arr[2],arr[2]-arr[1],sigma=np.sqrt(arr[3]**2+arr[4]**2))
            a,b,d=popt
            xx=np.linspace(arr[2].min(),arr[2].max(),100)
            ax1.plot(xx,a*(xx-c0)**2+b*(xx-c0)+d,c='r',label=r'$g-r={:.2f}(g-{:.2f})^2+{:.2f}(g-{:.2f})+{:.2f}$'.format(a,c0,b,c0,d),alpha=0.5,linestyle='--')
            ax1.legend(fontsize=20)
            
            # Calculate Delta chi
            observed_color = arr[2] - arr[1]
            observed_color_err = np.sqrt(arr[3]**2 + arr[4]**2)
            model_color = a * (arr[2] - c0)**2 + b * (arr[2] - c0) + d
            delta_chi = (observed_color - model_color) / observed_color_err

            # Plot Delta chi vs g
            ax2.scatter(arr[2], delta_chi)
            ax2.set_ylabel(r'$\Delta \chi$', fontsize=20)
            ax2.axhline(0, color='r', linestyle='--', linewidth=2)


    
    
    #plt.gca().invert_yaxis()
    #plt.gca().invert_xaxis()
    
    return None


def color3(SourceID,corr=False):
    plt.figure(figsize=(20,15))

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

    # up to DR15, Nov 2022, MJD 59892
    tt_zr=tt_zr[tt_zr['mjd']<59892*u.day]
    tt_zg=tt_zg[tt_zg['mjd']<59892*u.day]

    start=np.floor(min([tt_zr['mjd'].value.min(),tt_zg['mjd'].value.min()]))
    end=np.ceil(max([tt_zr['mjd'].value.max(),tt_zg['mjd'].value.max()]))

    days=np.arange(start,end)
    arr=np.zeros((5,len(days)))
    arr[0,:]=days
    ## arr[days,zr,zg]

    for i,day in enumerate(days):
        zr_cand=tt_zr[np.floor(tt_zr['mjd'].value)==day]
        zg_cand=tt_zg[np.floor(tt_zg['mjd'].value)==day]
        if len(zr_cand)==1 and len(zg_cand)==1:
            arr[1,i]=zr_cand['mag'].value
            arr[2,i]=zg_cand['mag'].value
            arr[3,i]=zr_cand['magerr'].value
            arr[4,i]=zg_cand['magerr'].value
        else:
            arr[1,i]=np.nan
            arr[2,i]=np.nan
            arr[3,i]=np.nan
            arr[4,i]=np.nan

    arr=arr[:,np.isnan(arr[1,:])==False]

    plt.errorbar(arr[1],arr[2],xerr=arr[3],yerr=arr[4],fmt='o',ecolor='grey',capsize=2,label='Source ID = '+str(SourceID))
    
    if corr==True:
        print(pearsonr(arr[1],arr[2]))

    # fit with g=r+c
    popt, pcov =curve_fit(lambda x, c: x+c,arr[1],arr[2])
    c=popt[0]
    plt.plot(arr[1],arr[1]+c,c='r',label=r'$m_g=m_r+{:.2f}$'.format(c),alpha=0.5,linestyle='--')
    plt.xlabel(r'$m_r$',fontsize=30)
    plt.ylabel(r'$m_g$',fontsize=30)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    
    return None


def zrzg_corr(SourceID):
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

    # up to DR15, Nov 2022, MJD 59892
    tt_zr=tt_zr[tt_zr['mjd']<59892*u.day]
    tt_zg=tt_zg[tt_zg['mjd']<59892*u.day]

    start=np.floor(min([tt_zr['mjd'].value.min(),tt_zg['mjd'].value.min()]))
    end=np.ceil(max([tt_zr['mjd'].value.max(),tt_zg['mjd'].value.max()]))

    days=np.arange(start,end)
    arr=np.zeros((5,len(days)))
    arr[0,:]=days
    ## arr[days,zr,zg]

    for i,day in enumerate(days):
        zr_cand=tt_zr[np.floor(tt_zr['mjd'].value)==day]
        zg_cand=tt_zg[np.floor(tt_zg['mjd'].value)==day]
        if len(zr_cand)==1 and len(zg_cand)==1:
            arr[1,i]=zr_cand['mag'].value
            arr[2,i]=zg_cand['mag'].value
            arr[3,i]=zr_cand['magerr'].value
            arr[4,i]=zg_cand['magerr'].value
        else:
            arr[1,i]=np.nan
            arr[2,i]=np.nan
            arr[3,i]=np.nan
            arr[4,i]=np.nan

    arr=arr[:,np.isnan(arr[1,:])==False]

    corr=pearsonr(arr[1],arr[2])
    
    return corr

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
    plt.suptitle('Source ID = '+str(SourceID))


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

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]


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

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

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
    plt.suptitle('Source ID = '+str(SourceID))


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

        # up to DR15, Nov 2022, MJD 59892
        tt_ztf=tt_ztf[tt_ztf['mjd']<59892*u.day]

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

        # up to DR15, Nov 2022, MJD 59892
        tt_ztf=tt_ztf[tt_ztf['mjd']<59892*u.day]

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
    plt.suptitle('Source ID = '+str(SourceID))
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

        # up to DR15, Nov 2022, MJD 59892
        tt_ztf=tt_ztf[tt_ztf['mjd']<59892*u.day]

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

        # up to DR15, Nov 2022, MJD 59892
        tt_ztf=tt_ztf[tt_ztf['mjd']<59892*u.day]

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

    # up to DR15, Nov 2022, MJD 59892
    tt_zr=tt_zr[tt_zr['mjd']<59892*u.day]
    tt_zg=tt_zg[tt_zg['mjd']<59892*u.day]

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

def fit_mag(ztf_ID,plot=True,if_g1=True,if_gu=True,if_g2=True,if_return=True):
    g2_flag=False
    lw=4
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

    # up to DR15, Nov 2022, MJD 59892
    t=t[t['mjd']<59892*u.day]

    mag=np.array(t['mag'])
    magerr=np.array(t['magerr']).mean()

    histo=np.histogram(mag,bins=10,density=1)
    x=(histo[1][:-1]+histo[1][1:])/2
    y=histo[0]
    
    if plot:
        plt.hist(mag,bins=10,density=1,alpha=0.5)
        xx=np.linspace(x.min(),x.max(),50)
    
    # 1 gaussian
    if if_g1:
        try:
            popt_g1,pcov_g1 = curve_fit(gaussian1,x,y,bounds=([0,x.min(),magerr*0.7],[50,x.max(),x.max()-x.min()]))
            
            if plot:
                plt.plot(xx,gaussian1(xx,popt_g1[0],popt_g1[1],popt_g1[2]),label='gaussian',linewidth=lw)
                
            sqerr_gaussian1=((y-gaussian1(x,popt_g1[0],popt_g1[1],popt_g1[2]))**2).sum()/(10-3)
            
        except:
            sqerr_gaussian1=99999
        
    # 2 gaussian
    if if_g2:
        try:
            popt_g2,pcov_g2 = curve_fit(gaussian2,x,y,bounds=([0,x.min(),magerr*0.7,0,x.min(),magerr*0.5],[50,x.max(),x.max()-x.min(),50,x.max(),x.max()-x.min()]))
            if plot:
                plt.plot(xx,gaussian2(xx,popt_g2[0],popt_g2[1],popt_g2[2],popt_g2[3],popt_g2[4],popt_g2[5]),label='2 gaussian',linewidth=lw)
            
            sqerr_gaussian2=((y-gaussian2(x,popt_g2[0],popt_g2[1],popt_g2[2],popt_g2[3],popt_g2[4],popt_g2[5]))**2).sum()/(10-6)
            g2_flag=(popt_g2[2]+popt_g2[5])>np.abs(popt_g2[1]-popt_g2[4])
        except:
            sqerr_gaussian2=99999
        
    # gumbel
    if if_gu:    
        try:    
            popt_gum,pcov_gum = curve_fit(gumbel,x,y,p0=[1,x.mean(),1],bounds=([-50,x.min()-magerr,-100],[50,x.max()+magerr,100]))
            
            if plot:
                plt.plot(xx,gumbel(xx,popt_gum[0],popt_gum[1],popt_gum[2]),label='gumbel',linewidth=lw)
            
            sqerr_gumbel=((y-gumbel(x,popt_gum[0],popt_gum[1],popt_gum[2]))**2).sum()/(10-3)
            
        except:
            sqerr_gumbel=99999

    if plot:
        plt.xlabel('magnitude')
        plt.ylabel('probability density')
        plt.gca().invert_xaxis()
        plt.legend()

    if if_return:
        return sqerr_gaussian1,sqerr_gaussian2,sqerr_gumbel,g2_flag
    else:
        return None


def counterpart(ra,dec,t):
    c1 = SkyCoord(ra, dec,unit="deg")
    c2 = SkyCoord(t['RA'], t['DEC'],unit="deg")
    print(c1.separation(c2).min().to(u.arcsec))
    return t[c1.separation(c2).argmin()]

def slope_sinfit(SourceID,force_period=None,max_light=False, print_param=False, plot_fit_r=True, plot_fit_g=True):
    fs=40
    def sinfunc(t, A, w, p, k, c):
        return A * np.sin(w*t + p) + k*t + c
    plt.figure(figsize=(20,15))
    #plt.suptitle('Source ID = '+str(SourceID),fontsize=fs,y=0.93)
    plt.plot([],[],color='none',label='Source ID = '+str(SourceID))
    rbandcolor=['red','black','maroon']
    gbandcolor=['green','blue','seagreen']


    OIDs_r=t_zr[t_zr['SourceID']==SourceID]
    OIDs_g=t_zg[t_zg['SourceID']==SourceID]
    length=1
    
    if len(OIDs_r)!=0:
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

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]


        if force_period!=None:
            guess_period=force_period
        else:
            guess_period=t_zr[t_zr['OID']==OID]['period']*u.day
            guess_period=float(guess_period.value)

        guess_freq = 1./guess_period
        guess_amp = np.std(t_ztf['mag'].value) * 2.**0.5
        guess_offset = np.mean(t_ztf['mag'].value)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., 0.,guess_offset])


        popt, pcov =curve_fit(sinfunc,t_ztf['mjd'].value,t_ztf['mag'].value,p0=guess)
        A, w, p, k, c = popt
        perr = np.sqrt(np.diag(pcov))
        f = w/(2.*np.pi)
        fit_period=1./f
        print('guess_period:',guess_period)
        print('fit_period:',fit_period,'±',(2. * np.pi / w**2) * perr[1])

        tt=np.linspace(t_ztf['mjd'].value.min(),t_ztf['mjd'].value.max(),500)
        yy=A*np.sin(w*tt+p)+k*tt+c
        y=A*np.sin(w*t_ztf['mjd'].value+p)+k*t_ztf['mjd'].value+c

        sin_chi2=(((t_ztf['mag'].value-y)**2)/(t_ztf['magerr'].value**2)).sum()/(len(t_ztf)-1)
        print('sin_chi2:',sin_chi2)

        plt.errorbar(np.array(t_ztf['mjd']),np.array(t_ztf['mag']),yerr=np.array(t_ztf['magerr']),fmt='xb',ecolor='grey',capsize=4,label='r band',markeredgecolor=rbandcolor[0],markersize=10)
        if plot_fit_r==True:
            plt.plot(tt,yy,c='r',label='r fit')
        

        max_r=yy.max()
        min_r=yy.min()

        if print_param==True:
            print('A:',A,'±',perr[0])
            print('w:',w,'±',perr[1])
            print('p:',p,'±',perr[2])
            print('k:',k,'±',perr[3])
            print('c:',c,'±',perr[4])
            print('')
    
    if len(OIDs_g)!=0:
        if len(OIDs_g)==1:
            OID=OIDs_g['OID']
        else:
            OID=OIDs_g[OIDs_g['numobs'].argmax()]['OID']
    
        OID_str=str(int(OID))
        t_ztf=QTable.read('https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='+OID_str)
        t_ztf=t_ztf[t_ztf['catflags']==0]

        # up to DR15, Nov 2022, MJD 59892
        t_ztf=t_ztf[t_ztf['mjd']<59892*u.day]

        if force_period!=None:
            guess_period=force_period
        else:
            guess_period=t_zg[t_zg['OID']==OID]['period']*u.day
            guess_period=float(guess_period.value)

        guess_freq = 1./guess_period
        guess_amp = np.std(t_ztf['mag'].value) * 2.**0.5
        guess_offset = np.mean(t_ztf['mag'].value)
        guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., 0.,guess_offset])


        popt, pcov =curve_fit(sinfunc,t_ztf['mjd'].value,t_ztf['mag'].value,p0=guess)
        A, w, p, k, c = popt
        perr = np.sqrt(np.diag(pcov))
        f = w/(2.*np.pi)
        fit_period=1./f
        print('guess_period:',guess_period)
        print('fit_period:',fit_period, '±',(2. * np.pi / w**2) * perr[1])

        tt=np.linspace(t_ztf['mjd'].value.min(),t_ztf['mjd'].value.max(),500)
        yy=A*np.sin(w*tt+p)+k*tt+c
        y=A*np.sin(w*t_ztf['mjd'].value+p)+k*t_ztf['mjd'].value+c

        sin_chi2=(((t_ztf['mag'].value-y)**2)/(t_ztf['magerr'].value**2)).sum()/(len(t_ztf)-1)
        print('sin_chi2:',sin_chi2)

        plt.errorbar(np.array(t_ztf['mjd']),np.array(t_ztf['mag']),yerr=np.array(t_ztf['magerr']),fmt='xb',ecolor='grey',capsize=4,label='g band',markeredgecolor=gbandcolor[0],markersize=10)
        if plot_fit_g==True:
            plt.plot(tt,yy,c='r',ls='--',label='g fit')
        plt.legend(fontsize=fs*0.8)
        plt.xlabel('Time (days)',fontsize=fs*1.1)
        plt.ylabel('mag',fontsize=fs*1.1)
        plt.xticks(fontsize=fs*0.8)
        plt.yticks(fontsize=fs*0.8)
        plt.gca().invert_yaxis()

        max_g=yy.max()
        min_g=yy.min()

        if print_param==True:
            print('A:',A,'±',perr[0])
            print('w:',w,'±',perr[1])
            print('p:',p,'±',perr[2])
            print('k:',k,'±',perr[3])
            print('c:',c,'±',perr[4])
            print('')

    if max_light==True:
        if len(OIDs_r)==0:
            max_r=-999
            min_r=-999
        if len(OIDs_g)==0:
            max_g=-999
            min_g=-999
        return max_r,max_g,min_r,min_g
    else:
        return None
    

