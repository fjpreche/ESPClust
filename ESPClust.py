# -*- coding: utf-8 -*-
"""
EffModClust set of functions for clustering of effect modifiers in groups with different effect sizes

Created on Wed Dec 13 09:20:45 2023

@author: s05fp2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as smapi

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.cluster import KMeans

from sklearn.impute import KNNImputer

from statsmodels.stats.multitest import multipletests

from collections import Counter

import warnings
warnings.filterwarnings('ignore')


def data_cleaning(featuresIn,Yin,otherVariables,annotation,thmissing,k_neighbours,featTransform,plotYN):
    features = featuresIn
    classdata = Yin
    
    nmissing = features.isnull().sum().sum()
    print("Missing metabolites: " + str(nmissing))

    if nmissing > 0:
        missingFeatures = features.isnull().sum()/len(features)
        missingObservations = features.isnull().T.sum()/len(features.T)

        if plotYN == 'y':
            # Visualising the missing values 
            plt.figure()
            plt.imshow(features.isnull().T,cmap='Greys',  interpolation='nearest')
            plt.title("Array of missing exposures")
            plt.xlabel("Individuals")
            plt.ylabel("Exposures")


            plt.figure(figsize = (6.4,2.8))
            plt.plot(range(len(missingFeatures)),100*missingFeatures)
            #plt.plot(range(len(missingFeatures)),np.zeros(len(missingFeatures))+20)
            plt.title("Missing percentage for each exposure")
            plt.xlabel("Exposures")
            plt.ylabel("Missing percentage")
            plt.ylim((0,100))

            plt.figure(figsize = (6.4,2.8))
            plt.plot(range(len(missingObservations)),100*missingObservations)
            #plt.plot(range(len(missingFeatures)),np.zeros(len(missingFeatures))+20)
            plt.title("Missing percentage for each individual")
            plt.xlabel("Individuals")
            plt.ylabel("Missing percentage")
            plt.ylim((0,100))


        # Drop variables for which data is missing for more than 20% of individuals
        features = features.loc[:, (features.isnull().sum()/len(features)<0.2)]

        # Drop observations that are missing more than 20% of variables
        classdata = classdata.loc[features.isnull().T.sum()/len(features.T)<0.2, :]
        otherVariables = otherVariables.loc[features.isnull().T.sum()/len(features.T)<0.2, :]
        features = features.loc[features.isnull().T.sum()/len(features.T)<0.2, :]

        # Nearest neighbours imputation for missing variables
        imputer = KNNImputer(n_neighbors=k_neighbours, weights="uniform")
        imputer.fit(features)
        features[:] = imputer.transform(features)  # Assigning the values back with [:] is important to get a dataframe back

    met_names = features.columns

    if featTransform == "Plus1Log":
        features = np.log(features+1.)  # Add 1 and take the logarithm

    # Scale features after imputation (z-score)
    scaler = StandardScaler().fit(features)
    features[:] = scaler.transform(features)

    features["outcome"] = np.array(classdata["label"])

    annotation = annotation.loc[annotation['met_labels'].isin(features.columns)]

#    confounders = otherVariables #[['StudyNumber','BMI','sex','richnessGenes7M']]
#    confounders['sex'] = otherVariables['sex'].map({'M':0 , 'F':1})

    featuresExtended = features

    ccol = otherVariables.columns

    for i in range(len(ccol)):
        featuresExtended[ccol[i]] = list(otherVariables[ccol[i]])
    
    featuresExtended = featuresExtended.dropna()  # Removing rows (observations) containing Nan
    
    return featuresExtended,met_names


# A function for univariate linear regression of a set of variables
def LinSignificance(dataX,dataY,confounders,alphasig,plot_yn):
    cx = dataX.columns[0:(len(dataX.columns))]
    pvtlr = np.zeros(len(cx))
    beta = np.zeros(len(cx))
    ciL = np.zeros(len(cx))
    ciH = np.zeros(len(cx))
    oddsr = np.zeros(len(cx))
    oddsrciL = np.zeros(len(cx))
    oddsrciH = np.zeros(len(cx))
    
    for i in range(len(cx)):
        ix = cx[i]
        t = [ix]
        xdf = confounders.copy()
        xdf[t] = dataX[t]
        nf = xdf.columns
        x = np.array(xdf)
        if len(nf) == 1:
            x = x.reshape(-1,1)
    
        scaler = StandardScaler().fit(x)
        x_scaled = scaler.transform(x)
        xs = pd.DataFrame(data=x_scaled, columns=xdf.columns)
        y = dataY.values.reshape(-1,1)
    
        log_reg = smapi.OLS(y,xs).fit(disp=0)
    
        p = log_reg.pvalues.values
        pvtlr[i] = p[-1]
        beta[i] = log_reg.params.values[-1]
        ciL[i] = log_reg.params.values[-1]-log_reg.conf_int()[0][-1]
        ciH[i] = log_reg.conf_int()[1][-1]-log_reg.params.values[-1]       
        oddsr[i] = np.exp(log_reg.params.values[-1])
        oddsrciL[i] = np.exp(log_reg.params.values[-1])-np.exp(log_reg.conf_int()[0][-1])
        oddsrciH[i] = np.exp(log_reg.conf_int()[1][-1])-np.exp(log_reg.params.values[-1])
    
    siglr = dataX.columns[range(len(dataX.columns))][pvtlr < alphasig]
    
    BHtest = multipletests(pvtlr, alpha = alphasig, method = 'fdr_bh')
    pvtBenjaminiHlr = BHtest[1]
    
    sigBHlr = dataX.columns[range(len(dataX.columns))][BHtest[0]]
    len(sigBHlr)
    
    if plot_yn == "y":
        #Plot of the p-values - Uncorrected
        plt.figure()
        if len(dataX.columns)<31:
            plt.plot(dataX.columns,pvtlr,'o')
            plt.xticks(rotation = 90)
        else:
            plt.plot(pvtlr)
        plt.plot(np.zeros(len(pvtlr))+alphasig)
        plt.xlabel("metabolites")
        plt.ylabel('p-value (uncorrected)')
        plt.title("$y \sim x_m$")
        
        #Plot for p-values - Benjamini-Hochberg correction
        plt.figure()
        
        if len(dataX.columns)<31:
            plt.plot(dataX.columns,pvtBenjaminiHlr,"o")
            plt.xticks(rotation = 90)
        else:
            plt.plot(pvtBenjaminiHlr)
        plt.plot(np.zeros(len(pvtlr)) + alphasig)
        plt.xlabel("metabolites")
        plt.ylabel('p-value (Benjamini-Hochberg correction')
        plt.title("$y \sim x_m$")
        
        
        #Coefficients
        fig, ax = plt.subplots(figsize=(7, 4))
        #xlabels=range(len(sigBH0)) #['sbp','tobacco', 'ldl','obesity','alcohol', 'age','famhist']
        y = beta
        yerrlow = ciL
        yerrhigh = ciH
        ax.errorbar(range(len(y)),y,yerr=[yerrlow,yerrhigh],fmt='o')
        ax.set_ylabel("Regression coeffs. (All metabolites)")
        ax.set_xlabel("metabolites")
        ax.set_title("$y \sim x_m$")
    
        #Coefficients - Only significant metabolites
        alpha = alphasig
        fig, ax = plt.subplots(figsize=(7, 4))
        y = beta[pvtBenjaminiHlr < alpha]
        yerrlow = ciL[pvtBenjaminiHlr < alpha]
        yerrhigh = ciH[pvtBenjaminiHlr < alpha]
        ax.errorbar(y,sigBHlr,xerr=[yerrlow,yerrhigh],fmt='o')
        ax.axvline(x = 0.0, color = 'black', ls='--', lw=1, label = 'axvline - full height')
        #ax.plot(np.zeros(len(sigBHlr)))
        ax.set_ylabel("metabolites")
        plt.xticks(rotation = 90)
        ax.set_xlabel("Regression coeffs. (Only significant)")
        
    # Correction using statsmodel (https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html)
    #import statsmodels.stats as smstats
    pvalBonferroni = multipletests(pvtlr, alpha=alphasig, method = 'bonferroni')
    pvalBH = multipletests(pvtlr, alpha=alphasig, method = 'fdr_bh')
    
    # Corrected p-values of significant metabolites according to Benjamini - Hochberg
    sigBonferroni = dataX.columns[range(len(dataX.columns))][pvalBonferroni[0]]
    len(sigBonferroni)
    sigBH0 = dataX.columns[range(len(dataX.columns))][pvalBH[0]]
    len(sigBH0)

    return pvtlr,siglr,pvalBH,sigBH0,beta,ciL,ciH


# Defining functions with separation Delta_All between their origin and width L_All
def Homogeneous_Windows(data,modifier_names,L_All,Delta_All):
    z0 = [0 for m in range(len(Delta_All))]
    Lw = [0 for m in range(len(Delta_All))]

    for m in range(len(Delta_All)):
        z = data[modifier_names[m]]
        zmin = np.min(z)
        zmax = np.max(z)

        Delta_t = Delta_All[m]
        L0 = L_All[m]

        ws=Delta_t
        ns = int((zmax-zmin)/ws)
        #print([zmin,zmax,ns])

        wtemp = []
        for i in range(ns+1):
            wtemp.append(zmin+i*ws)

        z0[m] = wtemp
        Lw[m] = [L0 for i in range(len(wtemp))]
        
    return z0,Lw


# A function to find the intersection of a list listarrays of 2 or more arrays
def intersectMultiple(listarrays):
    x = listarrays[0]
    for i in range(1,len(listarrays)):
        x = np.intersect1d(x,listarrays[i])
    return x

## Example
#aux = [np.array([1,2,3,4]),np.array([2,3,4,5]),np.array([3,4,5,6])]
#intersectMultiple(aux)

# Rows from data that satisfy the conditions for a window = [window z1, window z2, ....] (a list of length equal to the number of modifiers)
def selectedIndex(data,i_window,modifier_names,z0,Lw):
    #print(z0)
    #print(w_l)
    listarrays = []
    for m in range(len(modifier_names)):
        i = i_window[m]
        zL = z0[m][i]
        zH = zL+Lw[m][i]
        var = modifier_names[m]
        log = (data[var]>=zL) & (data[var]<zH)
        listarrays.append(np.array([i for i, x in enumerate(log) if x]))
    return intersectMultiple(listarrays)



# Function to calculate window-dependent effect sizes (odds ratios). One effect modifier. Linear regression
def effect_windows1_lin(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin):
    
    Eff_table_1d = np.zeros([len(z0[0]),len(X_name)])

    for ix in range(len(X_name)):
        xvar = X_name[ix]
        print('Variable '+str(ix+1)+' of '+str(len(X_name))+', '+xvar)

        nobs = []
        oddsr_list = []

        w00_list = []
        m0_list = []
        Lw0_list = []
        
        index_1d_list = []
        index_1d = 0
        for m0 in range(len(z0[0])):
            i_window = [m0]

            #print(i_window)

            positions = list(selectedIndex(data,i_window,modifier_names,z0,Lw))
            indsel = data.index[positions]  # using the position list to select indices from data

            frange = data.loc[indsel]

            if len(frange)>nmin:
                index_1d_list.append(index_1d)
                ow0 = z0[0][m0] # 1st coordinate of Origin of window m1,m2
                w00_list.append(ow0)

                #print(len(frange))
                temp = frange[xvar]
                x0 = np.transpose(np.array(temp)).flatten()
                y = list(frange[Y_name])

                xdf = frange[confound_names]
                xdf['x'] = list(x0)

                nf = xdf.columns
                x = np.array(xdf)
                if len(nf) == 1:
                    x = x.reshape(-1,1)

                scaler = StandardScaler().fit(x)
                x_scaled = scaler.transform(x)
                xs = pd.DataFrame(data=x_scaled, columns=xdf.columns)

                log_reg = smapi.OLS(y,xs).fit(disp=0)

                beta = log_reg.params.values[-1]

                Eff_table_1d[index_1d,ix] = beta

                nobs.append(len(frange))
                oddsr_list.append(beta)
                m0_list.append(m0)
                Lw0_list.append(Lw[0][m0])
                
            index_1d = index_1d+1
                
    Eff_table_1d = Eff_table_1d[index_1d_list,:]

    cols = list(X_name)
    cols.insert(0,modifier_names[0]+'_Lw')
    cols.insert(0,modifier_names[0]+'_z0')
    cols.insert(0,"nobs")
    
    esp_df = pd.DataFrame(Eff_table_1d,columns=X_name)
    esp_df['nobs'] = nobs
    esp_df[modifier_names[0]+'_z0'] = w00_list
    esp_df[modifier_names[0]+'_Lw'] = Lw0_list

    esp_df = esp_df[cols]
    
    return esp_df

# Function to calculate window-dependent effect sizes (odds ratios). One effect modifier. Logistic regression
def effect_windows1_LR(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin):
    
    Eff_table_1d = np.zeros([len(z0[0]),len(X_name)])

    for ix in range(len(X_name)):
        xvar = X_name[ix]
        print('Variable '+str(ix+1)+' of '+str(len(X_name))+', '+xvar)

        nobs = []
        oddsr_list = []

        w00_list = []
        m0_list = []
        Lw0_list = []
        
        index_1d_list = []
        index_1d = 0
        for m0 in range(len(z0[0])):
            i_window = [m0]

            #print(i_window)

            positions = list(selectedIndex(data,i_window,modifier_names,z0,Lw))
            indsel = data.index[positions]  # using the position list to select indices from data

            frange = data.loc[indsel]

            if len(frange)>nmin:
                index_1d_list.append(index_1d)
                ow0 = z0[0][m0] # 1st coordinate of Origin of window m1,m2
                w00_list.append(ow0)

                #print(len(frange))
                temp = frange[xvar]
                x0 = np.transpose(np.array(temp)).flatten()
                y = list(frange[Y_name])

                xdf = frange[confound_names]
                xdf['x'] = list(x0)

                nf = xdf.columns
                x = np.array(xdf)
                if len(nf) == 1:
                    x = x.reshape(-1,1)

                scaler = StandardScaler().fit(x)
                x_scaled = scaler.transform(x)
                xs = pd.DataFrame(data=x_scaled, columns=xdf.columns)

                log_reg = smapi.Logit(y,xs).fit(disp=0)

                beta = log_reg.params.values[-1]
                ciL = log_reg.params.values[-1]-log_reg.conf_int()[0][-1]
                ciH = log_reg.conf_int()[1][-1]-log_reg.params.values[-1]
                oddsr = np.exp(log_reg.params.values[-1])

                Eff_table_1d[index_1d,ix] = oddsr

                nobs.append(len(frange))
                oddsr_list.append(beta)
                m0_list.append(m0)
                Lw0_list.append(Lw[0][m0])
                
            index_1d = index_1d+1
                
    Eff_table_1d = Eff_table_1d[index_1d_list,:]

    cols = list(X_name)
    cols.insert(0,modifier_names[0]+'_Lw')
    cols.insert(0,modifier_names[0]+'_z0')
    cols.insert(0,"nobs")
    
    esp_df = pd.DataFrame(Eff_table_1d,columns=X_name)
    esp_df['nobs'] = nobs
    esp_df[modifier_names[0]+'_z0'] = w00_list
    esp_df[modifier_names[0]+'_Lw'] = Lw0_list

    esp_df = esp_df[cols]
    
    
    return esp_df


# Function to calculate window-dependent effect sizes (odds ratios). Linear regression
def effect_windows2_lin(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin):
    
    Eff_table_1d = np.zeros([len(z0[0])*len(z0[1]),len(X_name)])

    for ix in range(len(X_name)):
        xvar = X_name[ix]
        print('Variable '+str(ix+1)+' of '+str(len(X_name))+', '+xvar)

        nobs = []
        oddsr_list = []

        w00_list = []
        w01_list = []
        m0_list = []
        m1_list = []
        Lw0_list = []
        Lw1_list = []
        
        index_1d_list = []
        index_1d = 0
        for m0 in range(len(z0[0])):
        #for m0 in range(3):
            for m1 in range(len(z0[1])):
                i_window = [m0,m1]

                #print(i_window)

                positions = list(selectedIndex(data,i_window,modifier_names,z0,Lw))
                indsel = data.index[positions]  # using the position list to select indices from data

                frange = data.loc[indsel]

                if len(frange)>nmin:
                    index_1d_list.append(index_1d)
                    ow0 = z0[0][m0] # 1st coordinate of Origin of window m1,m2
                    w00_list.append(ow0)
                    ow1 = z0[1][m1] # 2nd coordinate of Origin of window m1,m2
                    w01_list.append(ow1)
                        
                    #print(len(frange))
                    temp = frange[xvar]
                    x0 = np.transpose(np.array(temp)).flatten()
                    y = list(frange[Y_name])

                    xdf = frange[confound_names]
                    xdf['x'] = list(x0)

                    nf = xdf.columns
                    x = np.array(xdf)
                    if len(nf) == 1:
                        x = x.reshape(-1,1)

                    scaler = StandardScaler().fit(x)
                    x_scaled = scaler.transform(x)
                    xs = pd.DataFrame(data=x_scaled, columns=xdf.columns)

                    log_reg = smapi.OLS(y,xs).fit(disp=0)

                    beta = log_reg.params.values[-1]

                    Eff_table_1d[index_1d,ix] = beta

                    nobs.append(len(frange))
                    oddsr_list.append(beta)
                    m0_list.append(m0)
                    m1_list.append(m1)
                    Lw0_list.append(Lw[0][m0])
                    Lw1_list.append(Lw[1][m1])
                    
                index_1d = index_1d+1
                    
    Eff_table_1d = Eff_table_1d[index_1d_list,:]

    cols = list(X_name)
    cols.insert(0,modifier_names[1]+'_Lw')
    cols.insert(0,modifier_names[1]+'_z0')
    cols.insert(0,modifier_names[0]+'_Lw')
    cols.insert(0,modifier_names[0]+'_z0')
    cols.insert(0,"nobs")
    
    esp_df = pd.DataFrame(Eff_table_1d,columns=X_name)
    esp_df['nobs'] = nobs
    esp_df[modifier_names[0]+'_z0'] = w00_list
    esp_df[modifier_names[0]+'_Lw'] = Lw0_list
    esp_df[modifier_names[1]+'_z0'] = w01_list
    esp_df[modifier_names[1]+'_Lw'] = Lw1_list
    esp_df = esp_df[cols]
        
    return esp_df


# Function to calculate window-dependent effect sizes (odds ratios). Logistic regression
def effect_windows2_LR(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin):
    
    Eff_table_1d = np.zeros([len(z0[0])*len(z0[1]),len(X_name)])

    for ix in range(len(X_name)):
        xvar = X_name[ix]
        print('Variable '+str(ix+1)+' of '+str(len(X_name))+', '+xvar)

        nobs = []
        oddsr_list = []
        oddsrciL_list = []
        oddsrciH_list = []

        w00_list = []
        w01_list = []
        m0_list = []
        m1_list = []
        Lw0_list = []
        Lw1_list = []
        
        index_1d_list = []
        index_1d = 0
        for m0 in range(len(z0[0])):
        #for m0 in range(3):
            for m1 in range(len(z0[1])):
                i_window = [m0,m1]

                #print(i_window)

                positions = list(selectedIndex(data,i_window,modifier_names,z0,Lw))
                indsel = data.index[positions]  # using the position list to select indices from data

                frange = data.loc[indsel]

                if len(frange)>nmin:
                    index_1d_list.append(index_1d)
                    ow0 = z0[0][m0] # 1st coordinate of Origin of window m1,m2
                    w00_list.append(ow0)
                    ow1 = z0[1][m1] # 2nd coordinate of Origin of window m1,m2
                    w01_list.append(ow1)
                        
                    #print(len(frange))
                    temp = frange[xvar]
                    x0 = np.transpose(np.array(temp)).flatten()
                    y = list(frange[Y_name])

                    xdf = frange[confound_names]
                    xdf['x'] = list(x0)

                    nf = xdf.columns
                    x = np.array(xdf)
                    if len(nf) == 1:
                        x = x.reshape(-1,1)

                    scaler = StandardScaler().fit(x)
                    x_scaled = scaler.transform(x)
                    xs = pd.DataFrame(data=x_scaled, columns=xdf.columns)

                    log_reg = smapi.Logit(y,xs).fit(disp=0)

                    beta = log_reg.params.values[-1]
                    oddsr = np.exp(log_reg.params.values[-1])

                    Eff_table_1d[index_1d,ix] = oddsr

                    nobs.append(len(frange))
                    oddsr_list.append(beta)
                    m0_list.append(m0)
                    m1_list.append(m1)
                    Lw0_list.append(Lw[0][m0])
                    Lw1_list.append(Lw[1][m1])
                    
                index_1d = index_1d+1
                    
    Eff_table_1d = Eff_table_1d[index_1d_list,:]

    cols = list(X_name)
    cols.insert(0,modifier_names[1]+'_Lw')
    cols.insert(0,modifier_names[1]+'_z0')
    cols.insert(0,modifier_names[0]+'_Lw')
    cols.insert(0,modifier_names[0]+'_z0')
    cols.insert(0,"nobs")
    
    esp_df = pd.DataFrame(Eff_table_1d,columns=X_name)
    esp_df['nobs'] = nobs
    esp_df[modifier_names[0]+'_z0'] = w00_list
    esp_df[modifier_names[0]+'_Lw'] = Lw0_list
    esp_df[modifier_names[1]+'_z0'] = w01_list
    esp_df[modifier_names[1]+'_Lw'] = Lw1_list
    esp_df = esp_df[cols]
        
    return esp_df


# Function to calculate window-dependent effect sizes 
# 3 effect modifiers - Linear regression
def effect_windows3_lin(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin):
    Eff_table_1d = np.zeros([len(z0[0])*len(z0[1])*len(z0[2]),len(X_name)])
    
    for ix in range(len(X_name)):
        xvar = X_name[ix]
        print('Variable '+str(ix+1)+' of '+str(len(X_name))+', '+xvar)

        nobs = []
        oddsr_list = []
        oddsrciL_list = []
        oddsrciH_list = []

        w00_list = []
        w01_list = []
        w02_list = []
        m0_list = []
        m1_list = []
        m2_list = []
        Lw0_list = []
        Lw1_list = []
        Lw2_list = []
        
        index_1d_list = []
        index_1d = 0
        for m0 in range(len(z0[0])):
        #for m0 in range(3):
            for m1 in range(len(z0[1])):
                for m2 in range(len(z0[2])):
                    i_window = [m0,m1,m2]

                    #print(i_window)

                    positions = list(selectedIndex(data,i_window,modifier_names,z0,Lw))
                    indsel = data.index[positions]  # using the position list to select indices from data

                    frange = data.loc[indsel]

                    if len(frange)>nmin:
                        index_1d_list.append(index_1d)
                        ow0 = z0[0][m0] # 1st coordinate of Origin of window m1,m2
                        w00_list.append(ow0)
                        ow1 = z0[1][m1] # 2nd coordinate of Origin of window m1,m2
                        w01_list.append(ow1)
                        ow2 = z0[2][m2] # 2nd coordinate of Origin of window m1,m2
                        w02_list.append(ow2)
        
                        #print(len(frange))
                        temp = frange[xvar]
                        x0 = np.transpose(np.array(temp)).flatten()
                        y = list(frange[Y_name])

                        xdf = frange[confound_names]
                        xdf['x'] = list(x0)

                        nf = xdf.columns
                        x = np.array(xdf)
                        if len(nf) == 1:
                            x = x.reshape(-1,1)

                        scaler = StandardScaler().fit(x)
                        x_scaled = scaler.transform(x)
                        xs = pd.DataFrame(data=x_scaled, columns=xdf.columns)

                        log_reg = smapi.OLS(y,xs).fit(disp=0)

                        beta = log_reg.params.values[-1]
                        oddsr = np.exp(log_reg.params.values[-1])

                        Eff_table_1d[index_1d,ix] = beta
                        
                        #print(oddsr)

                        nobs.append(len(frange))
                        oddsr_list.append(oddsr)
                        m0_list.append(m0)
                        m1_list.append(m1)
                        m2_list.append(m2)
                        Lw0_list.append(Lw[0][m0])
                        Lw1_list.append(Lw[1][m1])
                        Lw2_list.append(Lw[2][m2])
    
                    index_1d = index_1d+1
    
    Eff_table_1d = Eff_table_1d[index_1d_list,:]
    
    cols = list(X_name)
    cols.insert(0,modifier_names[2]+'_Lw')
    cols.insert(0,modifier_names[2]+'_z0')
    cols.insert(0,modifier_names[1]+'_Lw')
    cols.insert(0,modifier_names[1]+'_z0')
    cols.insert(0,modifier_names[0]+'_Lw')
    cols.insert(0,modifier_names[0]+'_z0')
    cols.insert(0,"nobs")
    
    esp_df = pd.DataFrame(Eff_table_1d,columns=X_name)
    esp_df['nobs'] = nobs
    esp_df[modifier_names[0]+'_z0'] = w00_list
    esp_df[modifier_names[0]+'_Lw'] = Lw0_list
    esp_df[modifier_names[1]+'_z0'] = w01_list
    esp_df[modifier_names[1]+'_Lw'] = Lw1_list
    esp_df[modifier_names[2]+'_z0'] = w02_list
    esp_df[modifier_names[2]+'_Lw'] = Lw2_list    
    esp_df = esp_df[cols]
    
    
    return esp_df

# Function to calculate window-dependent effect sizes 
# 3 effect modifiers - Logistic regression
def effect_windows3_LR(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin):
    Eff_table_1d = np.zeros([len(z0[0])*len(z0[1])*len(z0[2]),len(X_name)])
    
    for ix in range(len(X_name)):
        xvar = X_name[ix]
        print('Variable '+str(ix+1)+' of '+str(len(X_name))+', '+xvar)

        nobs = []
        oddsr_list = []

        w00_list = []
        w01_list = []
        w02_list = []
        m0_list = []
        m1_list = []
        m2_list = []
        Lw0_list = []
        Lw1_list = []
        Lw2_list = []
        
        index_1d_list = []
        index_1d = 0
        for m0 in range(len(z0[0])):
        #for m0 in range(3):
            for m1 in range(len(z0[1])):
                for m2 in range(len(z0[2])):
                    i_window = [m0,m1,m2]

                    #print(i_window)

                    positions = list(selectedIndex(data,i_window,modifier_names,z0,Lw))
                    indsel = data.index[positions]  # using the position list to select indices from data

                    frange = data.loc[indsel]

                    if len(frange)>nmin:
                        index_1d_list.append(index_1d)
                        ow0 = z0[0][m0] # 1st coordinate of Origin of window m1,m2
                        w00_list.append(ow0)
                        ow1 = z0[1][m1] # 2nd coordinate of Origin of window m1,m2
                        w01_list.append(ow1)
                        ow2 = z0[2][m2] # 2nd coordinate of Origin of window m1,m2
                        w02_list.append(ow2)
        
                        #print(len(frange))
                        temp = frange[xvar]
                        x0 = np.transpose(np.array(temp)).flatten()
                        y = list(frange[Y_name])

                        xdf = frange[confound_names]
                        xdf['x'] = list(x0)

                        nf = xdf.columns
                        x = np.array(xdf)
                        if len(nf) == 1:
                            x = x.reshape(-1,1)

                        scaler = StandardScaler().fit(x)
                        x_scaled = scaler.transform(x)
                        xs = pd.DataFrame(data=x_scaled, columns=xdf.columns)

                        log_reg = smapi.Logit(y,xs).fit(disp=0)

                        beta = log_reg.params.values[-1]
                        oddsr = np.exp(log_reg.params.values[-1])

                        Eff_table_1d[index_1d,ix] = beta
                        
                        #print(oddsr)

                        nobs.append(len(frange))
                        oddsr_list.append(oddsr)
                        m0_list.append(m0)
                        m1_list.append(m1)
                        m2_list.append(m2)
                        Lw0_list.append(Lw[0][m0])
                        Lw1_list.append(Lw[1][m1])
                        Lw2_list.append(Lw[2][m2])
    
                    index_1d = index_1d+1
    
    Eff_table_1d = Eff_table_1d[index_1d_list,:]
    
    cols = list(X_name)
    cols.insert(0,modifier_names[2]+'_Lw')
    cols.insert(0,modifier_names[2]+'_z0')
    cols.insert(0,modifier_names[1]+'_Lw')
    cols.insert(0,modifier_names[1]+'_z0')
    cols.insert(0,modifier_names[0]+'_Lw')
    cols.insert(0,modifier_names[0]+'_z0')
    cols.insert(0,"nobs")
    
    esp_df = pd.DataFrame(Eff_table_1d,columns=X_name)
    esp_df['nobs'] = nobs
    esp_df[modifier_names[0]+'_z0'] = w00_list
    esp_df[modifier_names[0]+'_Lw'] = Lw0_list
    esp_df[modifier_names[1]+'_z0'] = w01_list
    esp_df[modifier_names[1]+'_Lw'] = Lw1_list
    esp_df[modifier_names[2]+'_z0'] = w02_list
    esp_df[modifier_names[2]+'_Lw'] = Lw2_list    
    esp_df = esp_df[cols]
    
    
    return esp_df


# Plot of the effect size as a function of windows for an effect modifier mod_name
# esp_df: Result from the effect windows analysis
# variable: Name of the variable whose effect size will be plotted
# modifier: Name of the effect modifier to be plotted
# errorbar (y/n): Whether to plot the interval for the effect modifier or not. 
#   If not plotted (useful for discrete modifiers), a solid circle indicates the effect size for each window.
#   If plotted, a symbol is used to indicate the midpoint of the interval.
def Eff_size_Windows_plot(esp_df,variable,modifier,no_effect_value,errorbar):
    xerrlow =  esp_df[modifier+'_z0'] #np.array(w00_list)
    xerrhigh = esp_df[modifier+'_z0']+esp_df[modifier+'_Lw'] #np.array(w00_list)+np.array(Lw[0])[m0_list]
    x = (xerrlow+xerrhigh)/2 #5/2
    y = esp_df[variable] #Eff_table_1d[:,20]
    
    fig = plt.figure(figsize=(6, 2))
    
    if (errorbar == 'n') | (errorbar == 'N'):
        plt.plot(xerrlow,y,'o')
        plt.axhline(y = no_effect_value, color = 'black', ls='--', lw=1, label = 'axvline - full height')    
    elif (errorbar == 'y') | (errorbar == 'Y'):
        #plt.errorbar(x,y,xerr=[np.array(Lw[0])[m0_list]/2,np.array(Lw[0])[m0_list]/2],fmt='.')
        plt.errorbar(x,y,xerr=[esp_df[modifier+'_Lw']/2,esp_df[modifier+'_Lw']/2],fmt='.')
        plt.axhline(y = no_effect_value, color = 'black', ls='--', lw=1, label = 'axvline - full height')
    else:
        print('Specify if the error bars are to be plotted')

    
    plt.xlabel(modifier)
    plt.ylabel('Effect size')
    plt.title(variable)


# kmax: This is the largest number of neighbours to be explored. All clustering indices
# are explored for k >= 2 except inertia which is explored for k >= 1.
def Clustering_indices(features_df,kmax,cluster_method,plotYN):
    scaler = StandardScaler().fit(features_df) # Define a standardisation scaler
    x_scaled = scaler.transform(features_df) # Transform the data.

    import random
    random.seed(2)
    
    CH = [] #Calinski-Harabasz
    DB = [] #Davies-Bouldin
    Sil = [] #Silhouette
    K_range = range(2, kmax)

    for k in K_range:
        if cluster_method == "Kmeans":
            inicentroids = [] #np.zeros(n_clusters)
            delta = int(len(x_scaled)/k)
            for i in range(k):
                inicentroids.append(list(x_scaled[i*delta]))
            kmeans = KMeans(n_clusters = k, init=np.array(inicentroids))

        if cluster_method == "Agglomerate":
            kmeans = AgglomerativeClustering(n_clusters = k)
        
        kmeans.fit(x_scaled)
        labels = kmeans.labels_
        
        CH.append(metrics.calinski_harabasz_score(x_scaled, labels))
        DB.append(metrics.davies_bouldin_score(x_scaled, labels))
        Sil.append(metrics.silhouette_score(x_scaled, labels))

    KoptimalCH = K_range[np.argmax(CH)]
    KoptimalDB = K_range[np.argmax(DB)]
    KoptimalSil = K_range[np.argmax(Sil)]
    
    
    inertia = [] #Scree plot
    K_range2 = range(1, kmax)

    for k in K_range2:
        if cluster_method == "Kmeans":
            inicentroids = [] #np.zeros(n_clusters)
            delta = int(len(x_scaled)/k)
            for i in range(k):
                inicentroids.append(list(x_scaled[i*delta]))
            kmeans = KMeans(n_clusters = k, init=np.array(inicentroids))

        if cluster_method == "Agglomerate":
            kmeans = AgglomerativeClustering(n_clusters = k)
        
        kmeans.fit(x_scaled)
        labels = kmeans.labels_

        centres = np.zeros((k,np.shape(x_scaled)[1]))
        for i in range(np.shape(centres)[0]):
            c = np.mean(x_scaled[labels==i],axis=0)
            for j in range(np.shape(centres)[1]):
                centres[i,j] = c[j]
        inertia0=0
        for c in range(k):
            error = 0.0
            for i in range(len(x_scaled[labels==c])):
                error = error + np.sum((x_scaled[labels==c][i]-centres[c])**2)
            inertia0 = inertia0+error

        inertia.append(inertia0)

    slope = [inertia[k]-inertia[k-1] for k in range(1,len(inertia))]
    k_values = np.array([k for k in range(2,len(slope)+1)])
    slope_change = np.array([slope[k]/slope[k+1]-1 for k in range(len(slope)-1)])
    KoptimalElbow = k_values[slope_change == np.max(slope_change)][0]
    
    if (plotYN == 'y') | (plotYN == 'Y'):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 12))
        ax1.plot(K_range,CH,'o-')
        ax1.set_xlabel('$K$')
        ax1.set_ylabel('Calinski-Harabasz')
        ax1.axvline(x = K_range[np.argmax(CH)], color = 'black', ls='--', lw=1, label = 'axvline - full height')
        ax1.set_xlim([min(K_range2)-1, max(K_range2)+1])

        ax2.plot(K_range,DB,'o-')
        ax2.set_xlabel('$K$')
        ax2.set_ylabel('Davies-Bouldin')
        ax2.axvline(x = K_range[np.argmax(DB)], color = 'black', ls='--', lw=1, label = 'axvline - full height')
        ax2.set_xlim([min(K_range2)-1, max(K_range2)+1])
        
        ax3.plot(K_range,Sil,'o-')
        ax3.set_xlabel('$K$')
        ax3.set_ylabel('Silhouette')
        ax3.axvline(x = K_range[np.argmax(Sil)], color = 'black', ls='--', lw=1, label = 'axvline - full height')
        ax3.set_xlim([min(K_range2)-1, max(K_range2)+1])
        
        ax4.plot(K_range2,inertia,'o-')
        ax4.set_xlabel('$K$')
        ax4.set_ylabel('Inertia')
        ax4.axvline(x = KoptimalElbow, color = 'black', ls='--', lw=1, label = 'axvline - full height')
        ax4.set_xlim([min(K_range2)-1, max(K_range2)+1])
        
        plt.tight_layout()
        
    CH.insert(0,None)
    DB.insert(0,None)
    Sil.insert(0,None)
    
    index_vs_K_df = pd.DataFrame({'k': K_range2, 'CH': CH, 'DB': DB, 'Sil': Sil, 'inertia': inertia})
    
    arr = [KoptimalCH,KoptimalDB,KoptimalSil,KoptimalElbow]
    
    a = Counter(arr).most_common(2)
    
    koptimal_overall = a[0][0]
    if a[0][1] == 1:
        koptimal_overall = min(arr)
    if a[1][1] == 2:
        koptimal_overall = min(arr)

    return index_vs_K_df,KoptimalCH,KoptimalDB,KoptimalSil,KoptimalElbow,koptimal_overall


def Window_clusters_labels(distance,n_clusters,cluster_method,clusterOrder):
    scaler = StandardScaler().fit(distance) # Define a standardisation scaler
    x_scaled = scaler.transform(distance) # Transform the data.

    sortlables = "y" #"n" # "y"

    inicentroids = [] #np.zeros(n_clusters)
    delta = int(len(x_scaled)/n_clusters)

    import random
    random.seed(2)

    #centroids 1
    for i in range(n_clusters):
        inicentroids.append(list(x_scaled[i*delta]))

    #centroids 2
    #deltaHalph = int(delta/2)
    #for i in range(n_clusters):
    #    inicentroids.append(list(x_scaled[i*delta+deltaHalph]))

    # Define the model

    if cluster_method=="Kmeans":
        kmeans = KMeans(n_clusters = n_clusters, init=np.array(inicentroids))
    if cluster_method=="Agglomerate":
        kmeans = AgglomerativeClustering(n_clusters = n_clusters)

    # Fit it to the data, analogous to FindClusters in Mathematica
    kmeans.fit(x_scaled)

    labels = kmeans.labels_

    if sortlables == "y":
        # Change label of clusters - Growing order
        uniqueval = []
        for il in range(len(labels)):
            if labels[il] not in uniqueval:
                uniqueval.append(labels[il])
                #lprev=labels[il]
        uniqueval

        #uniqueval = list(set(labels))

        test_keys = uniqueval
        test_values = list(range(n_clusters))

        dictionary = {}
        for key in test_keys:
            for value in test_values:
                dictionary[key] = value
                test_values.remove(value)
                break
        dictionary

        labels_df = pd.DataFrame(labels, columns=['value'])
        labels_df2 = labels_df.replace({'value': dictionary})
        labels = np.array(labels_df2['value'])

        labels_temp = labels.copy()
        for i in range(n_clusters):
            l0 = labels == i
            labels_temp[l0] = clusterOrder[i]
        labels = labels_temp.copy()
        #print(ltemp)
        #for i in range(n_clusters):
        #    labels[ltemp[i]] = clusterOrder[i]
    
    return labels



# Calculate the centres of the window clusters
# Input: 
#    - distance: Distance matrix between effect modifier windows
#    - n_clusters: Number of clusters
#    - cluster_method: "Agglomerate" or "Kmeans"
#    - A plot is provided if plot is set to 'lines', 'points', 'points-lines' or 'errorbar'
#    - clusterOrder: A list with a specific order for clusters (e.g. [1,0] if there are two clusters)
def Window_cluster_centroids(distance,n_clusters,cluster_method,clusterOrder,plot,scale_ErrorSD,diff_sorted):
    if diff_sorted == "N":
        centres2,centres_SD2,inertia2 = Window_cluster_centres_aux(distance,n_clusters,cluster_method,scale_ErrorSD,plot,clusterOrder)
    if diff_sorted == "Y":
        centres,centres_SD,inertia = Window_cluster_centres_aux(distance,n_clusters,cluster_method,scale_ErrorSD,'none',clusterOrder)
        sorted_total_overlap, index_sorted_total_overlap = total_overlap(centres,centres_SD,scale_ErrorSD)

        distance2 = distance[distance.columns[index_sorted_total_overlap]]
        centres2,centres_SD2,inertia2 = Window_cluster_centres_aux(distance2,n_clusters,cluster_method,scale_ErrorSD,plot,clusterOrder)
        
    return centres2,centres_SD2,inertia2

def Window_cluster_centres_aux(distance,n_clusters,cluster_method,scale_ErrorSD,plot,clusterOrder):
    scaler = StandardScaler().fit(distance) # Define a standardisation scaler
    x_scaled = scaler.transform(distance) # Transform the data.

    sortlables = "y" #"n" # "y"

    inicentroids = [] #np.zeros(n_clusters)
    delta = int(len(x_scaled)/n_clusters)

    import random
    random.seed(2)

    #centroids 1
    for i in range(n_clusters):
        inicentroids.append(list(x_scaled[i*delta]))

    if cluster_method=="Kmeans":
        kmeans = KMeans(n_clusters = n_clusters, init=np.array(inicentroids))
    if cluster_method=="Agglomerate":
        kmeans = AgglomerativeClustering(n_clusters = n_clusters)

    # Fit it to the data, analogous to FindClusters in Mathematica
    kmeans.fit(x_scaled)

    labels = kmeans.labels_

    if sortlables == "y":
        # Change label of clusters - Growing order
        uniqueval = []
        for il in range(len(labels)):
            if labels[il] not in uniqueval:
                uniqueval.append(labels[il])
                #lprev=labels[il]
        uniqueval

        #uniqueval = list(set(labels))

        test_keys = uniqueval
        test_values = list(range(n_clusters))

        dictionary = {}
        for key in test_keys:
            for value in test_values:
                dictionary[key] = value
                test_values.remove(value)
                break
        dictionary

        labels_df = pd.DataFrame(labels, columns=['value'])
        labels_df2 = labels_df.replace({'value': dictionary})
        labels = np.array(labels_df2['value'])

    labels_temp = labels.copy()
    for i in range(n_clusters):
        l0 = labels == i
        labels_temp[l0] = clusterOrder[i]
    labels = labels_temp.copy()
    
    uniqueval = []
    for il in range(len(labels)):
        if labels[il] not in uniqueval:
            uniqueval.append(labels[il])
            #lprev=labels[il]
    uniqueval
    
    n_clusters=len(uniqueval)
    
    centres = np.zeros((n_clusters,np.shape(x_scaled)[1]))
    centres_SD = np.zeros((n_clusters,np.shape(x_scaled)[1]))
    for i in range(np.shape(centres)[0]):
        c = np.mean(x_scaled[labels==i],axis=0)
        cSD = np.std(x_scaled[labels==i],axis=0)
        for j in range(np.shape(centres)[1]):
            centres[i,j] = c[j]
            centres_SD[i,j] = cSD[j]
    
    # Calculation of the quadratic error (inertia)
    inertia=0
    for c in range(n_clusters):
        error = 0.0
        for i in range(len(x_scaled[labels==c])):
            error = error + np.sum((x_scaled[labels==c][i]-centres[c])**2)
        inertia = inertia+error
    
    X_name = distance.columns
    
    df_centres = pd.DataFrame(data = centres, columns=list(X_name))
    df_centres_SD = pd.DataFrame(data = centres_SD, columns=list(X_name))
    
    if plot == 'lines':
        font = {'size'   : 11}
        plt.rc('font', **font)

        verticalD = max([4.,len(X_name)/4.])
        fig, ax = plt.subplots(figsize=(7, verticalD))

        #centres = kmeans.cluster_centers_

        #ax.hlines(metsgroup,centres[0],np.zeros(len(centres[0])),colors='y')
        #ax.hlines(metsgroup,centres[1],np.zeros(len(centres[0])),colors='b')
        for i in range(len(centres)):
            ax.plot(centres[i],X_name,'-',label='Cluster '+str(i),markersize=10)

        ax.axvline(x = 0, color = 'black', ls='-', lw=1)
        ax.set_xlabel('Normalised centroids')
        ax.set_ylim([-1,len(X_name)+1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -2.5/len(X_name)),fancybox=True, shadow=True, ncol=5)
    
    if plot == 'points':
        font = {'size'   : 11}
        plt.rc('font', **font)

        verticalD = max([4.,len(X_name)/4.])
        fig, ax = plt.subplots(figsize=(7, verticalD))

        #centres = kmeans.cluster_centers_

        #ax.hlines(metsgroup,centres[0],np.zeros(len(centres[0])),colors='y')
        #ax.hlines(metsgroup,centres[1],np.zeros(len(centres[0])),colors='b')
        for i in range(len(centres)):
            ax.plot(centres[i],X_name,'o',label='Cluster '+str(i),markersize=10)

        ax.axvline(x = 0, color = 'black', ls='-', lw=1)
        ax.set_xlabel('Normalised centroids')
        ax.set_ylim([-1,len(X_name)+1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -2.5/len(X_name)),fancybox=True, shadow=True, ncol=5)
 
    if plot == 'points-lines':
        font = {'size'   : 11}
        plt.rc('font', **font)

        verticalD = max([4.,len(X_name)/4.])
        fig, ax = plt.subplots(figsize=(7, verticalD))

        #centres = kmeans.cluster_centers_

        #ax.hlines(metsgroup,centres[0],np.zeros(len(centres[0])),colors='y')
        #ax.hlines(metsgroup,centres[1],np.zeros(len(centres[0])),colors='b')
        for i in range(len(centres)):
            ax.plot(centres[i],X_name,'o-',label='Cluster '+str(i),markersize=10)

        ax.axvline(x = 0, color = 'black', ls='-', lw=1)
        ax.set_xlabel('Normalised centroids')
        ax.set_ylim([-1,len(X_name)+1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -2.5/len(X_name)),fancybox=True, shadow=True, ncol=5)
 
    
    if plot == 'errorbar':
        font = {'size'   : 11}
        plt.rc('font', **font)

        verticalD = max([4.,len(X_name)/4.])
        fig, ax = plt.subplots(figsize=(7, verticalD))

        #centres = kmeans.cluster_centers_

        #ax.hlines(metsgroup,centres[0],np.zeros(len(centres[0])),colors='y')
        #ax.hlines(metsgroup,centres[1],np.zeros(len(centres[0])),colors='b')
        for i in range(len(centres)):
            #ax.plot(centres[i],X_name,'-o',label='Cluster '+str(i),markersize=10)
            ax.errorbar(centres[i],X_name,xerr=scale_ErrorSD*centres_SD[i],fmt='.',label='Cluster '+str(i))

        ax.axvline(x = 0, color = 'black', ls='-', lw=1)
        ax.set_xlabel('Normalised centroids') # !!!!CHANGE THIS!!!!!
        ax.set_ylim([-1,len(X_name)+1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -2.5/len(X_name)),fancybox=True, shadow=True, ncol=5)
            
    return df_centres,df_centres_SD,inertia



# Plot of centres for arbitrary variables
def Window_cluster_centres_plot(centres,centres_SD,scale_ErrorSD,plotType):
    X_name = centres.columns
    
    if plotType == 'lines':
        font = {'size'   : 11}
        plt.rc('font', **font)

        verticalD = max([4.,len(X_name)/4.])
        fig, ax = plt.subplots(figsize=(7, verticalD))

        #centres = kmeans.cluster_centers_

        #ax.hlines(metsgroup,centres[0],np.zeros(len(centres[0])),colors='y')
        #ax.hlines(metsgroup,centres[1],np.zeros(len(centres[0])),colors='b')
        for i in range(centres.shape[0]):
            ax.plot(centres.loc[i],X_name,'-',label='Cluster '+str(i),markersize=10)

        ax.axvline(x = 0, color = 'black', ls='-', lw=1)
        ax.set_xlabel('Normalised centroids')
        ax.set_ylim([-1,len(X_name)+1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01),fancybox=True, shadow=True, ncol=5)
    
    if plotType == 'points':
        font = {'size'   : 11}
        plt.rc('font', **font)

        verticalD = max([4.,len(X_name)/4.])
        fig, ax = plt.subplots(figsize=(7, verticalD))

        #centres = kmeans.cluster_centers_

        #ax.hlines(metsgroup,centres[0],np.zeros(len(centres[0])),colors='y')
        #ax.hlines(metsgroup,centres[1],np.zeros(len(centres[0])),colors='b')
        for i in range(centres.shape[0]):
            ax.plot(centres.loc[i],X_name,'o',label='Cluster '+str(i),markersize=10)

        ax.axvline(x = 0, color = 'black', ls='-', lw=1)
        ax.set_xlabel('Normalised centroids')
        ax.set_ylim([-1,len(X_name)+1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01),fancybox=True, shadow=True, ncol=5)
 
    if plotType == 'points-lines':
        font = {'size'   : 11}
        plt.rc('font', **font)

        verticalD = max([4.,len(X_name)/4.])
        fig, ax = plt.subplots(figsize=(7, verticalD))

        #centres = kmeans.cluster_centers_

        #ax.hlines(metsgroup,centres[0],np.zeros(len(centres[0])),colors='y')
        #ax.hlines(metsgroup,centres[1],np.zeros(len(centres[0])),colors='b')
        for i in range(centres.shape[0]):
            ax.plot(centres.loc[i],X_name,'o-',label='Cluster '+str(i),markersize=10)

        ax.axvline(x = 0, color = 'black', ls='-', lw=1)
        ax.set_xlabel('Normalised centroids')
        ax.set_ylim([-1,len(X_name)+1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01),fancybox=True, shadow=True, ncol=5)
 
    
    if plotType == 'errorbar':
        font = {'size'   : 11}
        plt.rc('font', **font)

        verticalD = max([4.,len(X_name)/4.])
        fig, ax = plt.subplots(figsize=(7, verticalD))

        #centres = kmeans.cluster_centers_

        #ax.hlines(metsgroup,centres[0],np.zeros(len(centres[0])),colors='y')
        #ax.hlines(metsgroup,centres[1],np.zeros(len(centres[0])),colors='b')
        for i in range(centres.shape[0]):
            #ax.plot(centres[i],X_name,'-o',label='Cluster '+str(i),markersize=10)
            ax.errorbar(centres.loc[i],X_name,xerr=scale_ErrorSD*centres_SD.loc[i],fmt='.',label='Cluster '+str(i))

        ax.axvline(x = 0, color = 'black', ls='-', lw=1)
        ax.set_xlabel('Normalised centroids')
        ax.set_ylim([-1,len(X_name)+1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -2.5/centres.shape[1]),fancybox=True, shadow=True, ncol=5)


# Overlap between the centre interval for two groups
def pair_overlap(c1,c2,centres,centres_SD,scale_ErrorSD):
    l1 = np.array(centres.iloc[c1])-np.array(scale_ErrorSD*centres_SD.iloc[c1])
    l2 = np.array(centres.iloc[c2])-np.array(scale_ErrorSD*centres_SD.iloc[c2])
    h1 = np.array(centres.iloc[c1])+np.array(scale_ErrorSD*centres_SD.iloc[c1])
    h2 = np.array(centres.iloc[c2])+np.array(scale_ErrorSD*centres_SD.iloc[c2])
    
    max_l = np.array([max(l1[i],l2[i]) for i in range(len(l1))])
    min_h = np.array([min(h1[i],h2[i]) for i in range(len(h1))])
    
    overlap = min_h-max_l
    return overlap

# Total overlap between the centre interval of several pairs groups
def total_overlap(centres,centres_SD,scale_ErrorSD):
    sum_overlap = np.zeros(len(centres.columns))
    n_clusters = len(centres)
    for c1 in range(n_clusters-1):
        for c2 in range(1,n_clusters):
            ov = pair_overlap(c1,c2,centres,centres_SD,scale_ErrorSD)
            sum_overlap = sum_overlap+ov
    
    sorted_total_overlap = list(np.sort(sum_overlap)[::-1])
    index_sorted_total_overlap = list(np.argsort(sum_overlap)[::-1])
    return sorted_total_overlap,index_sorted_total_overlap

# - Proportion of a continuous effect modifier in each cluster
# Inputs:
# - x: Value of the effect modifier
# - w_origin: 
def prop_cluster0(x,w_origin,Lwin,clabels):
    dft = pd.DataFrame({"w0": w_origin, "Lw0": Lwin, "Cluster": clabels})
    
    n_clusters = len(set(clabels))
    
    clustlist = []
    for i in range(len(dft)):
        xmin = dft["w0"][i]
        xmax = dft["w0"][i]+dft["Lw0"][i]
        if (x>=xmin) & (x<xmax):
            clustlist.append(dft["Cluster"][i])
    #print(clustlist)
    if len(clustlist)>0:
        p_in_c = []
        for c in range(n_clusters):
            p_in_c.append(clustlist.count(c)/len(clustlist))

    return p_in_c


## - Proportion of a continuous effect modifier in each cluster
#def prop_cluster(x,w_origin,Lwin,clabels):
#    dft = pd.DataFrame({"w0": w_origin, "Lw0": Lwin, "Cluster": clabels})
#    
#    n_clusters = len(set(clabels))
#    
#    clustlist = []
#    for i in range(len(dft)):
#        xmin = dft["w0"][i]
#        xmax = dft["w0"][i]+dft["Lw0"][i]
#        if (x>=xmin) & (x<xmax):
#            clustlist.append(dft["Cluster"][i])
#    #print(clustlist)
#    if len(clustlist)>0:
#        p_in_c = []
#        for c in range(n_clusters):
#            p_in_c.append(clustlist.count(c)/len(clustlist))
#
#    return p_in_c

# - Plot of the cluster proportion for continuous effect modifiers
def plot_prop_clusters(x,x_label,w_origin,Lwin,clabels):
    w_origin=list(w_origin)
    n_clusters = len(set(clabels))
    p_in_c_list = []
    x_list = []
    for xin in x:
#        if xin<w_origin[0]:
#            xin = w_origin[0]
#        if xin>w_origin[-1]:
#            xin = w_origin[-1]
        if (xin>=w_origin[0]) &  (xin<=w_origin[-1]):
            p_in_c = prop_cluster0(xin,w_origin,Lwin,clabels)
            p_in_c_list.append(p_in_c)
            x_list.append(xin)

    c_names = ["Cluster "+str(c) for c in range(n_clusters)]
    df_clustProp = pd.DataFrame(p_in_c_list, columns = c_names)
    df_clustProp['x'] = x_list


    font = {'size'   : 14}
    plt.rc('font', **font)

    colorlist = ['#1f77b4', '#FF7F0E', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    df_clustProp.plot(x='x', kind='bar', width= 1.0, stacked=True, color=colorlist[0:n_clusters])#, title='...')#, xticks=[20,40,60])
    
    range_x = np.max(df_clustProp['x'])-np.min(df_clustProp['x'])
    
    xtick_loc = [5*i*int(range_x)/20 for i in range(5)] #[0,10,20,60,70,80]
    #ytick_loc = [0.12, 0.80, 0.76]
    plt.xticks(xtick_loc)
    plt.xlabel(x_label)
    plt.ylabel('Proportion in each cluster')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),fancybox=True, shadow=True, ncol=5)
    plt.axhline(y = 0.5, color = 'black', ls='--', lw=1)
    #plt.tight_layout()



# - Plot of the cluster proportion for continuous effect modifiers
#prop_cluster0(modifier_value,modifier_name,esp_df,clabels)
def prop_cluster_xlist(modifier_name,modifier_values,esp_df,clabels,plotYN):
    w_origin = list(esp_df[modifier_name+'_z0'])
    Lwin = list(esp_df[modifier_name+'_Lw'])
    x = modifier_values
    #print(x)
    
    n_clusters = len(set(clabels))
    p_in_c_list = []
    x_list = []
    for xin in x:
#        if xin<w_origin[0]:
#            xin = w_origin[0]
#        if xin>w_origin[-1]:
#            xin = w_origin[-1]
        if (xin>=w_origin[0]) &  (xin<=w_origin[-1]):
            p_in_c = prop_cluster0(xin,w_origin,Lwin,clabels)#prop_cluster0(modifier_name,xin,esp_df,cs) #prop_cluster0(xin,w_origin,Lwin,clabels)
            p_in_c_list.append(p_in_c)
            x_list.append(xin)
            
    c_names = ["Cluster "+str(c) for c in range(n_clusters)]
    df_clustProp = pd.DataFrame(p_in_c_list, columns = c_names)
    df_clustProp['x'] = x_list
    
    if plotYN == 'Y':
        font = {'size'   : 14}
        plt.rc('font', **font)

        colorlist = ['#1f77b4', '#FF7F0E', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        df_clustProp.plot(x='x', kind='bar', width= 1.0, stacked=True, color=colorlist[0:n_clusters])#, title='...')#, xticks=[20,40,60])

        range_x = np.max(df_clustProp['x'])-np.min(df_clustProp['x'])
        xtick_loc = [5*i*int(range_x)/20 for i in range(5)] #[0,10,20,60,70,80]
        #ytick_loc = [0.12, 0.80, 0.76]
        plt.xticks(xtick_loc)
        plt.xlabel(modifier_name)
        plt.ylabel('Proportion in each cluster')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),fancybox=True, shadow=True, ncol=5)
        plt.axhline(y = 0.5, color = 'black', ls='--', lw=1)
    
    return df_clustProp


def EffSize_Clusters(data,modifier,confound_names,modifier_names,X_name,df_propClust,clabels,plotYN):
    n_clusters = len(set(clabels))


    EffSize_clust_df_list = []
    clust_index = []
    for c in range(n_clusters):
        if (df_propClust['Cluster '+str(c)]>0.5).sum()>0:
            clust_index.append(c)
            xt = df_propClust[df_propClust['Cluster '+str(c)]>0.5]['x']
            xmin = np.min(xt)
            xmax = np.max(xt)

            data_c = data.loc[((data[modifier_names] >= xmin) & (data[modifier_names] <= xmax))[modifier]]

            alphasig = 1 #0.05
            dataX = data_c[X_name]
            dataY = data_c["class"]
            plot_yn = "n"

            confounderIn=data_c[confound_names]
            pvtlr,siglr, pvalBHlr, sigBH0lr, beta, ciL, ciH = LinSignificance(dataX,dataY,confounderIn,alphasig,'n')

            dforsorted = pd.DataFrame({"met_name": siglr, "beta": beta, "ciL": ciL, "ciH": ciH})

            EffSize_clust_df_list.append(dforsorted)
            
        if (plotYN == 'Y') | (plotYN == 'y'):
            font = {'size'   : 11}
            plt.rc('font', **font)
    
            verticalD = max([4.,len(X_name)/4.])
            fig, ax = plt.subplots(figsize=(7, verticalD))
    
            #centres = kmeans.cluster_centers_
    
            #ax.hlines(metsgroup,centres[0],np.zeros(len(centres[0])),colors='y')
            #ax.hlines(metsgroup,centres[1],np.zeros(len(centres[0])),colors='b')
            for i in range(len(EffSize_clust_df_list)):
                    ax.plot(EffSize_clust_df_list[i]["beta"],X_name,'-s',label='Cluster '+str(clust_index[i]),markersize=10)
    
            ax.axvline(x = 0, color = 'black', ls='-', lw=1)
            ax.set_xlabel('Effect size')
            ax.set_ylim([-1,len(X_name)+1])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -2.5/len(X_name)),fancybox=True, shadow=True, ncol=5)

            
    return EffSize_clust_df_list,clust_index


def plot_clusters_CovSpace(esp_df,X_name,modifier_names,n_clusters,cluster_method,clusterOrder):
    distance = esp_df[X_name]
    labels = Window_clusters_labels(distance,n_clusters,cluster_method,clusterOrder)
    if len(modifier_names)==1:
        labelsDF = pd.DataFrame({modifier_names[0]: esp_df[modifier_names[0]+'_z0'], '2D': [0 for i in range(len(esp_df[modifier_names[0]+'_z0']))], 'Cluster': labels})
        plt.figure(figsize=(12,4))
        sns.lmplot(x=modifier_names[0], y='2D', data=labelsDF, fit_reg=False, hue='Cluster', legend=False)
        plt.yticks([])
        plt.ylabel('')
        # Move the legend to an empty part of the plot
        #plt.legend(loc='upper left')
        plt.legend() #w0_2D[0], '2D': [0 for i in range(len(w0_2D[0]))], 'Cluster': labels})
        plt.show()

    if len(modifier_names)==2:
        #labelsDF = pd.DataFrame({modifier_names[0]: w0_2D[0], modifier_names[1]: w0_2D[1], 'Cluster': labels})
        plt.figure(figsize=(8,4))
        labelsDF = pd.DataFrame({modifier_names[0]: esp_df[modifier_names[0]+'_z0'], modifier_names[1]: esp_df[modifier_names[1]+'_z0'], 'Cluster': labels})
        sns.lmplot(x=modifier_names[0], y=modifier_names[1], data=labelsDF, fit_reg=False, hue='Cluster', legend=False)

        plt.legend()
        #plt.legend(loc='upper left')
        plt.legend(loc = 'center right')
        #ax.set_yticks([0,1],["M","F"])
        plt.show()

    if len(modifier_names)==3:
        labelsDF = pd.DataFrame({modifier_names[0]: esp_df[modifier_names[0]+'_z0'], modifier_names[1]: esp_df[modifier_names[1]+'_z0'], modifier_names[2]: esp_df[modifier_names[2]+'_z0'],'Cluster': labels})
        sns.lmplot(x=modifier_names[0], y=modifier_names[1], data=labelsDF, fit_reg=False, hue='Cluster', legend=False)
        #plt.legend(loc='upper left')
        plt.legend()

        sns.lmplot(x=modifier_names[0], y=modifier_names[2], data=labelsDF, fit_reg=False, hue='Cluster', legend=False)
        #plt.legend(loc='upper left')
        plt.legend()

        plt.show()

        from mpl_toolkits import mplot3d
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        ax.view_init(elev=20, azim=10)

        for ic in range(n_clusters):
            aux = labelsDF[labelsDF['Cluster']==ic]
            #ax.scatter(aux[modifier_names[0]],aux[modifier_names[1]],aux[modifier_names[2]], s=40);
            ax.plot(aux[modifier_names[0]],aux[modifier_names[1]],aux[modifier_names[2]],'o',label='Cluster '+str(ic));

        ax.set_xlabel(modifier_names[0])
        ax.set_ylabel(modifier_names[1])
        ax.set_zlabel(modifier_names[2])
        #ax.set_yticks([0,1],["M","F"])
        #ax.zaxis.labelpad=1
        ax.xaxis._axinfo['label']['space_factor'] = 2.8
        ax.zaxis._axinfo['label']['space_factor'] = 1.0
        #ax.legend(loc = 'center right')
        
    return labelsDF

def ESP_pca(features_df,cluster_method,plot_yn,pcomp_1,pcomp_2,n_clusters,clusterOrder):
    from sklearn.decomposition import PCA
    distance = features_df
    
    pca = PCA(n_components=len(distance.columns))
    
    # Scaled
    scaler = StandardScaler().fit(distance) # Define a standardisation scaler
    x_scaled = scaler.transform(distance) # Transform the data.
    X = x_scaled

    ## Not scaled
    #X = distance_measure

    x_pca = pca.fit(X).transform(X)

    cumVar = np.cumsum(pca.explained_variance_ratio_)
    
    if plot_yn == 'y':
        pc1 = pcomp_1-1
        pc2 = pcomp_2-1

        labels = Window_clusters_labels(distance,n_clusters,cluster_method,clusterOrder)

        fig = plt.figure(figsize=(5,5))
        font = {'size'   : 14}
        plt.rc('font', **font)

        if pc1==0:
            percent1 = 100*np.round(cumVar[pc1],2)
        else:
            percent1 = 100*np.round(cumVar[pc1],2)-100*np.round(cumVar[pc1-1],2)

        if pc2==0:
            percent2 = 100*np.round(cumVar[pc2],2)
        else:
            percent2 = 100*np.round(cumVar[pc2],2)-100*np.round(cumVar[pc2-1],2)

        for ic in range(n_clusters):
            plt.plot(x_pca[labels==ic,pc1],x_pca[labels==ic,pc2],'o',label = 'Cluster '+str(ic))

        plt.xlabel('$PC $'+str(pc1+1)+' ('+str(int(percent1))+'%)')
        plt.ylabel('$PC $'+str(pc2+1)+' ('+str(int(percent2))+'%)')
        #plt.xlim([-45,45])

        plt.legend()
    
    return x_pca,cumVar



def Eff_size_Windows_plot_local(dfout,variable,modifier,no_effect_value,errorbar,xmin,xmax,xtitle,ytitle):
    xerrlow =  dfout[modifier+'_w0'] #np.array(w00_list)
    xerrhigh = dfout[modifier+'_w0']+dfout[modifier+'_Lw'] #np.array(w00_list)+np.array(Lw[0])[m0_list]
    x = (xerrlow+xerrhigh)/2 #5/2
    y = dfout[variable] #Eff_table_1d[:,20]

    fig = plt.figure(figsize=(6, 2))
    
    if (errorbar == 'n') | (errorbar == 'N'):
        plt.plot(xerrlow,y,'o')
        plt.axhline(y = no_effect_value, color = 'black', ls='--', lw=1, label = 'axvline - full height')    
    elif (errorbar == 'y') | (errorbar == 'Y'):
        #plt.errorbar(x,y,xerr=[np.array(Lw[0])[m0_list]/2,np.array(Lw[0])[m0_list]/2],fmt='.')
        plt.errorbar(x,y,xerr=[dfout[modifier+'_Lw']/2,dfout[modifier+'_Lw']/2],fmt='.')
        plt.axhline(y = no_effect_value, color = 'black', ls='--', lw=1, label = 'axvline - full height')
    else:
        print('Specify if the error bars are to be plotted')

    
    plt.xlabel(xtitle,fontsize=14)
    plt.ylabel(ytitle,fontsize=14)
    #plt.title(variable)
    plt.ylim([xmin, xmax])