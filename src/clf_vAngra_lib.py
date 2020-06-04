import numpy as np
import pandas as pd
import math 

from astropy.stats import mad_std
from sklearn.metrics import f1_score

def rotate(oX, oY, pX, pY, angle):
    from math import sin
    from math import cos
    from numpy import array
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox = oX
    oy = oY
    px = array(pX)
    py = array(pY)

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
    
    return qx.tolist(), qy.tolist()


def getAngle(X,Y):
    from numpy import polyfit
    from numpy import poly1d
    from numpy import arctan
    
    # - - - - Reta 0,0
    xo = [-2048,2048]
    yo = [0.001,0.001]
    
    zo = polyfit(yo,xo, 1)
    fo = poly1d(zo)    
    m1 = fo.c[0] 
    
    z = polyfit(X,Y, 1)
    func = poly1d(z) 
    m2 = func.c[0]
    
    angle = arctan(m1-m2/(1-m1*m2))    
    return angle


def hardCuts(newX_sig, i, df):
    cutLow = np.mean(newX_sig) - i*mad_std(newX_sig)
    cutUpp = np.mean(newX_sig) + i*mad_std(newX_sig)
      
    df.Tag[df.index[(df.AreaRot >= cutLow) & (df.AreaRot <= cutUpp)]] = 1    
    df.Tag[df.index[(df.AreaRot < cutLow)]] = 3
    df.Tag[df.index[(df.AreaRot > cutUpp)]] = 4
    df.Tag[df.index[(df.Amp >= 163)]] = 2
    df.Tag[df.index[(df.Pos_Amp <= 8) | (df.Pos_Amp >= 46)]]  = 3
    df.Tag[df.index[(df.Pos_Amp >= 15) & (df.Pos_Amp <= 35)]] = 4
    df.Tag[df.index[(df.FWHM >= 17)]] = 4
    df.Tag[df.index[(df.FWHM <= 6)]]  = 3
    return df


def NeutrinosClassifier(df, angle, sigma, test=False):      
    if angle == 'None':
        angle = getAngle(df.Area[df.Label==1],df.Amp[df.Label==1]) 
    
    newX_sig,newY_sig = rotate(0,0,df.Area[df.Label==1],df.Amp[df.Label==1],angle+math.radians(90))
    newX_dpc,newY_dpc = rotate(0,0,df.Area[df.Label==4],df.Amp[df.Label==4],angle+math.radians(90))
    newX_cut,newY_cut = rotate(0,0,df.Area[df.Label==3],df.Amp[df.Label==3],angle+math.radians(90))
    newX_sat,newY_sat = rotate(0,0,df.Area[df.Label==2],df.Amp[df.Label==2],angle+math.radians(90))
    
    xRot = np.concatenate((newX_sig,newX_sat,newX_cut,newX_dpc),axis=0)
    yRot = np.concatenate((newY_sig,newY_sat,newY_cut,newY_dpc),axis=0)
    
    df['AreaRot'] = xRot
    df['Tag'] = np.zeros(len(df))
    
    nF1 = np.array([])
    
    df = hardCuts(newX_sig, sigma, df)
    f1 = f1_score(df.Label, df.Tag,average='micro')
    nF1 = np.append(nF1, f1)
    
    if test:
        return nF1, angle, df
    else:
        return nF1, angle