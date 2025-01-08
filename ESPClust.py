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

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from statsmodels.stats.multitest import multipletests

from collections import Counter

from scipy.spatial import distance

import warnings
warnings.filterwarnings('ignore')

from functools import reduce

from scipy.spatial.distance import pdist

def data_cleaning(featuresIn,Yin,otherVariables,annotation,thmissing,k_neighbours,featTransform,plotYN):
    features = featuresIn
    classdata = Yin
    
    nmissing = features.isnull().sum().sum()
    print("Missing metabolites: " + str(nmissing))

    if nmissing > 0:
        missingFeatures = features.isnull().sum()/len(features)
        missingObservations = features.isnull().T.sum()/len(features.T)

        if plotYN == 'Y':
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


def window_parameters(data1,nmin=10,CL=0.95):

    if not (0 <= CL <= 1):
        raise ValueError(f"The value of 'CL' should be between 0 and 1.")

    Kmin = nmin-1
    #print("Minimal number of neighbours:", Kmin)

    explore_names=data1.columns

    scaler = StandardScaler().fit(data1)
    scaled_data1 = scaler.transform(data1)
    
    Lwindow=np.zeros([len(data1),len(explore_names)])
    dmin=np.zeros(len(explore_names))+1.0e7
    dminlist=np.zeros([len(data1),len(explore_names)])
    i = 0
    for ir in list(data1.index):#range(len(data1)):
        reference_point = scaled_data1[i]
    
        # Calculate the Euclidean distances between the reference point and all other points
        distances = np.linalg.norm(scaled_data1 - reference_point, axis=1)
        
        # Get the indices of the 10 smallest distances (ignoring the first point itself which has a distance of 0)
        closest_indices = np.argsort(distances)[1:(Kmin+1)]
        
        #print(closest_indices)
        
        for j in range(len(explore_names)):
            ic=list(data1.index)[closest_indices[0]]

            # Debugging check
            #print("Index i:", i, "Index ir:", ir, "Closest index ic:", ic, "Feature:", explore_names[j])
 
            #print(data1[explore_names[j]])
            #print([data1[explore_names[j]][ic],data1[explore_names[j]][ir]])
            diff=np.abs(data1[explore_names[j]][ic]-data1[explore_names[j]][ir])
           
            dminlist[i,j]=diff
            if diff<dmin[j]:
                dmin[j]=diff
            
        dtemp = data1.iloc[closest_indices]
         
        a=np.array([dtemp.min(),dtemp.max()])
        #Lwindow[i,0]=a[1,0]-a[0,0]
        #Lwindow[i,1]=a[1,1]-a[0,1]
        
        for j in range(len(explore_names)):
            Lwindow[i,j]=a[1,j]-a[0,j]

        i=i+1

    L=[]
    for i in range(len(explore_names)):
        L.append(np.quantile(Lwindow[:,i],CL))

    Delta=[]
    for i in range(len(explore_names)):
        Delta.append(np.min(Lwindow[:,i]))

    dmin1d=[]  # minimal distance between points in each dimension
    for i in range(len(explore_names)):
        array = np.array(data1[explore_names[i]])
        # Calculate all pairwise distances
        distances = pdist(array.reshape(-1, 1))
        # Find the smallest non-zero distance
        min_nonzero_distance = np.min(distances[distances > 0])
        dmin1d.append(min_nonzero_distance)

        
    return L,Delta


# Defining gliding windows with separation Delta_All between their origin and width L_All
def Homogeneous_Windows(data,modifier_names,L_All,Delta_All,var_type):
    z0 = [0 for m in range(len(Delta_All))]
    Lw = [0 for m in range(len(Delta_All))]

    for m in range(len(Delta_All)):
        z = data[modifier_names[m]]
        zmin = np.min(z)
        if var_type[m] == 'c':
            zmin = np.min(z)-L_All[m]/2
        zmax = np.max(z)
        if var_type[m] == 'c':
            zmax = np.max(z)-L_All[m]/2

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
        
    win_dim = [len(sublist) for sublist in Lw]
    n_windows = reduce(lambda x, y: x * y, win_dim)
     
    return z0,Lw,win_dim,n_windows


# A function to find the intersection of a list listarrays of 2 or more arrays
def intersectMultiple(listarrays):
    x = listarrays[0]
    for i in range(1,len(listarrays)):
        x = np.intersect1d(x,listarrays[i])
    return x

## Example
#aux = [np.array([1,2,3,4]),np.array([2,3,4,5]),np.array([3,4,5,6])]
#intersectMultiple(aux)



def selectedIndex(data, i_window, modifier_names, z0, Lw):
    positions = np.ones(len(data), dtype=bool)  # Start with all True (include all rows)

    for i, mod in enumerate(modifier_names):
        lower_bound = z0[i][i_window[i]]
        upper_bound = lower_bound + Lw[i][i_window[i]]

        # Update positions to select rows where the modifier is within the window
        positions &= (data[mod] >= lower_bound) & (data[mod] < upper_bound)
    
    return positions

# Define the main function for effect windows
def effect_windows(data, X_name, Y_name, confound_names, modifier_names, z0, Lw, nmin, effsize_method):
    n_modifiers = len(modifier_names)
    Eff_table_1d = np.zeros([np.prod([len(z) for z in z0]), len(X_name)])

    for ix in range(len(X_name)):
        xvar = X_name[ix]
        print(f'Variable {ix + 1} of {len(X_name)}, {xvar}')

        nobs = []
        effsize_list = []

        # Dynamic lists for each modifier
        modifier_z0_list = [[] for _ in range(n_modifiers)]
        modifier_Lw_list = [[] for _ in range(n_modifiers)]

        index_1d_list = []
        index_1d = 0

        # Handle any number of modifiers using a recursive loop
        for i_window in np.ndindex(*[len(z) for z in z0]):
            positions = list(selectedIndex(data, i_window, modifier_names, z0, Lw))
            indsel = data.index[positions]
            frange = data.loc[indsel]

            if len(frange) > nmin:
                index_1d_list.append(index_1d)

                for i_mod in range(n_modifiers):
                    ow = z0[i_mod][i_window[i_mod]]
                    modifier_z0_list[i_mod].append(ow)
                    modifier_Lw_list[i_mod].append(Lw[i_mod][i_window[i_mod]])

                # Prepare the data for effsize calculation
                temp = frange[xvar]
                x0 = np.transpose(np.array(temp)).flatten()
                y = list(frange[Y_name])

                xdf = frange[confound_names]
                xdf['x'] = list(x0)

                # Calculate effsize with the given method
                effsize = effsize_method(y, xdf)
                Eff_table_1d[index_1d, ix] = effsize

                nobs.append(len(frange))

            index_1d += 1

    Eff_table_1d_temp = Eff_table_1d[index_1d_list, :]

    # - scaling the effect sizes
    scaler = StandardScaler().fit(Eff_table_1d_temp)
    Eff_table_1d = scaler.transform(Eff_table_1d_temp)
    
    # Dynamically create columns for each modifier's z0 and Lw
    cols = list(X_name)
    for i_mod in range(n_modifiers):
        cols.insert(0, modifier_names[i_mod] + '_Lw')
        cols.insert(0, modifier_names[i_mod] + '_z0')
    cols.insert(0, "nobs")

    esp_df = pd.DataFrame(Eff_table_1d, columns=X_name)
    esp_df['nobs'] = nobs

    # Add z0 and Lw values for each modifier
    for i_mod in range(n_modifiers):
        esp_df[modifier_names[i_mod] + '_z0'] = modifier_z0_list[i_mod]
        esp_df[modifier_names[i_mod] + '_Lw'] = modifier_Lw_list[i_mod]

    esp_df = esp_df[cols]
    return esp_df

# Define a linear regression-based effsize method
def effsize_lin(y, xdf):
    scaler = StandardScaler().fit(xdf)
    x_scaled = scaler.transform(xdf)
    model = LinearRegression().fit(x_scaled, y)
    effsize = model.coef_[-1]  # Use the last coefficient as effsize for X
    return effsize

def effsize_logit_odds(y, xdf):
    # Scale the features
    scaler = StandardScaler().fit(xdf)
    x_scaled = scaler.transform(xdf)
    
    # Fit logistic regression model
    model = LogisticRegression().fit(x_scaled, y)
    # Get the coefficient for the predictor variable of interest (last column in this case)
    odds_ratio = np.exp(model.coef_[0][-1])  # Exponentiate to get odds ratio
    
    return odds_ratio


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
    
    # Count occurrences of each k in the array
    a = Counter(arr).most_common(2)
    
    # Set a default value for koptimal_overall
    koptimal_overall = a[0][0]  # Default to the most common k
    
    # Handle cases where there are fewer than two distinct values in `a`
    if len(a) > 1 and a[1][1] == 2:  # Check if the second most common value exists and is tied
        koptimal_overall = min(arr)
    elif a[0][1] == 1:  # Check if the most common value occurs only once
        koptimal_overall = min(arr)
    
    return index_vs_K_df,KoptimalCH,KoptimalDB,KoptimalSil,KoptimalElbow,koptimal_overall

def shuffle_dataframe_elements(df, random_state=None):
    """
    Randomly shuffle all elements in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to shuffle.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with shuffled elements.
    """
    # Flatten the DataFrame into a 1D array
    flattened = df.values.flatten()
    
    # Shuffle the array
    rng = np.random.default_rng(seed=random_state)
    rng.shuffle(flattened)
    
    # Reshape back into the DataFrame's original shape
    shuffled_df = pd.DataFrame(flattened.reshape(df.shape), columns=df.columns, index=df.index)
    
    return shuffled_df

def shuffle_columns(df, random_state=None):
    """
    Randomly shuffle the elements within each column of a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to shuffle.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with columns shuffled independently.
    """
    rng = np.random.default_rng(seed=random_state)
    shuffled_df = df.apply(lambda col: rng.permutation(col).tolist(), axis=0)
    
    return shuffled_df

def shuffle_rows(df, random_state=None):
    """
    Randomly shuffle the elements within each row of a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to shuffle.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with rows shuffled independently.
    """
    rng = np.random.default_rng(seed=random_state)
    # Apply shuffle to each row and stack back into a DataFrame
    shuffled_array = np.apply_along_axis(rng.permutation, axis=1, arr=df.values)
    return pd.DataFrame(shuffled_array, columns=df.columns, index=df.index)

def data_random(data):
    n_samples, n_features = data.shape
    feature_min = np.min(data, axis=0)
    feature_max = np.max(data, axis=0)
    reference_data = np.random.uniform(feature_min, feature_max, size=(n_samples, n_features))
    return reference_data

def Inertia_k(features_df,kmax,cluster_method):
    scaler = StandardScaler().fit(features_df) # Define a standardisation scaler
    x_scaled = scaler.transform(features_df) # Transform the data.

    import random
    random.seed(2)
    
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

    return inertia,slope,slope_change,k_values


def Elbow_significance(data, k_test, nr, cluster_method, alpha):
    
    # Compute inertia, slopes, and slope changes for the original data
    kmax = k_test+3
    inertia0, slope0, slope_change0, k_values0 = Inertia_k(data, kmax, cluster_method)

    # Index of the specific k to test (adjusting for Python 0-based indexing)
    k_index = k_test - 2  # Since slope_change corresponds to k values from 2 to kmax
    
    # Initialize variables for randomisation
    slope_change_distribution = np.zeros(nr)
    
    for r in range(nr):
        # Shuffle data for randomization
        features_df = data_random(data)  # Replace with appropriate shuffle function
        
        # Compute inertia, slopes, and slope changes for the randomized data
        kmax = k_test+3
        inertia, slope, slope_change, k_values = Inertia_k(features_df, kmax, cluster_method)
        
        # Store the slope_change value for this random sample at the specific k
        slope_change_distribution[r] = slope_change[k_index]
    
    # Calculate the mean and standard deviation for the slope change
    mean_slope_change = np.mean(slope_change_distribution)
    sd_slope_change = np.std(slope_change_distribution, ddof=1)
    
    # Calculate the percentile for significance
    percentile = np.percentile(slope_change_distribution, 100 * (1 - alpha))
    
    # Calculate the p-value for the specific slope change
    p_value = np.mean(slope_change_distribution >= slope_change0[k_index])
    
    # Determine whether to reject the null hypothesis
    #reject_null = slope_change0[k_index] > percentile
    reject_null = p_value < alpha
    
    # Ensure the index is within bounds
    if 2 <= k_test <= len(slope_change0) + 1:
        if reject_null:
            koptimal_overall_2 = k_test
        else:
            koptimal_overall_2 = 1
    else:
        # If koptimal_overall_1 is out of range, set a default value
        koptimal_overall_2 = 1
        

    return slope_change0[k_index], percentile, p_value, koptimal_overall_2, reject_null


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
        ax.set_xlabel('Normalized centroids')
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
        ax.set_xlabel('Normalized centroids')
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
        ax.set_xlabel('Normalized centroids')
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
            ax.errorbar(centres[i],X_name,xerr=scale_ErrorSD*centres_SD[i],fmt='.',label='Cluster '+str(i),capsize=5)

        ax.axvline(x = 0, color = 'black', ls='-', lw=1)
        ax.set_xlabel('Normalized centroids') # !!!!CHANGE THIS!!!!!
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
        ax.set_xlabel('Normalized centroids')
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
        ax.set_xlabel('Normalized centroids')
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
        ax.set_xlabel('Normalized centroids')
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
        ax.set_xlabel('Normalized centroids')
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



def plot_clusters_CovSpace(esp_df,X_name,modifier_names,n_clusters,cluster_method,clusterOrder,L):
    distance = esp_df[X_name]
    labels = Window_clusters_labels(distance,n_clusters,cluster_method,clusterOrder)
    if len(modifier_names)==1:
        labelsDF = pd.DataFrame({modifier_names[0]: esp_df[modifier_names[0]+'_z0']+L[0]/2, '2D': [0 for i in range(len(esp_df[modifier_names[0]+'_z0']))], 'Cluster': labels})
        plt.figure(figsize=(12,4))
        sns.lmplot(x=modifier_names[0], y='2D', data=labelsDF, fit_reg=False, hue='Cluster', legend=False)
        plt.yticks([])
        plt.ylabel('')
        # Move the legend to an empty part of the plot
        #plt.legend(loc='upper left')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) #w0_2D[0], '2D': [0 for i in range(len(w0_2D[0]))], 'Cluster': labels})
        plt.show()

    if len(modifier_names)==2:
        #labelsDF = pd.DataFrame({modifier_names[0]: w0_2D[0], modifier_names[1]: w0_2D[1], 'Cluster': labels})
        plt.figure(figsize=(8,4))
        labelsDF = pd.DataFrame({modifier_names[0]: esp_df[modifier_names[0]+'_z0']+L[0]/2, modifier_names[1]: esp_df[modifier_names[1]+'_z0']+L[1]/2, 'Cluster': labels})
        sns.lmplot(x=modifier_names[0], y=modifier_names[1], data=labelsDF, fit_reg=False, hue='Cluster', legend=False)

        plt.legend()
        #plt.legend(loc='upper left')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        #ax.set_yticks([0,1],["M","F"])
        plt.show()

    if len(modifier_names)==3:
        labelsDF = pd.DataFrame({modifier_names[0]: esp_df[modifier_names[0]+'_z0']+L[0]/2, modifier_names[1]: esp_df[modifier_names[1]+'_z0']+L[1]/2, modifier_names[2]: esp_df[modifier_names[2]+'_z0']+L[2]/2,'Cluster': labels})
        sns.lmplot(x=modifier_names[0], y=modifier_names[1], data=labelsDF, fit_reg=False, hue='Cluster', legend=False)
        #plt.legend(loc='upper left')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

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
    
    pca = PCA(n_components=min(len(distance.columns),len(distance)))
    
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

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
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