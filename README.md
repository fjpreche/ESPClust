# ESPClust
Unsupervised identification of modifiers for the effect size spectrum in omics association studies

## Installation

In the terminal or anaconda prompt, type:
```
pip install -i https://test.pypi.org/simple/ ssSIG==1.0.2  (Change this!!!)
```

## Tutorial 

A tutorial is provided in the Tutorial_ssSIG.ipynb notebook. This illustrates the functioning of the ssSIG package applied to synthetic and real data (available from the `Data` directory).

## List of functions

### Data cleaning
```
featuresExtended, exposure_names = data_cleaning(featuresIn,Yin,otherVariables,annotation,thmissing,k_neighbours,featTransform,plotYN)
```
#### Inputs
- `featuresIn`: exposures dataframe.
- `Yin`: outcome.
- `thmissing`: A threshold value in [0,1]. Metabolites and individuals whose percentage of missing values is above this threshold Threshold for discarding metabolites or individuals with missing values.
- `otherVariables`: A dataframe with with covariates.
- `k_neighbours`: Number of neighbours for nearest neighbour imputation.
- `featTransform` (`None`, `Plus1Log`): If set to `None`, do not transform the data. If set to `Plus1Log`, transformation of the data by adding 1 and taking the natural logarithm.
- `plotYN`: If set to 'Y', plots are provided for the array of missing exposures and the missing percentage for exposures and individuals.

#### Outputs
- `featuresExtended`: a dataframe with cleaned exposures together with columns from the `otherVariables` dataset and the `outcome` variable.
- `exposure_names`: Names of the exposures that were kept after cleaning.
- 
### Creating a cover for the covariate space with windows of homogeneous size
```
z0, Lw = Homogeneous_Windows(data,modifier_names,L_in,Delta_in)
```
#### Inputs
- `data` (similar to `featuresExtended`): A dataframe with cleaned exposures together with columns from the `otherVariables` dataset and the `outcome` variable.
- `modifier_names`: List of names of covariates to be explored as potential effect modifiers.
- `L_in`: A list of window lengths for each effect size modifier considered.
- `Delta_in`: A list of the steps used to glide the window in the direction of each effect size modifier.

#### Outputs
- `z0`: List of coordinates of the origin of each window. 
- `Lw`: List of dimensions of each window.

### Estimating the effect size profile (ESP)

#### Continuous outcome using linear regression

ESPClust provides three functions to deal with 1, 2 or 3 potential ESP modifiers:

```
esp_df = effect_windows1_lin(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin)
```

```
esp_df = effect_windows2_lin(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin)
```

```
esp_df = effect_windows3_lin(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin)
```

#### Binary outcome (two classes) using logistic regression

ESPClust provides three functions to deal with 1, 2 or 3 potential ESP modifiers:

```
esp_df = effect_windows1_LR(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin)
```

```
esp_df = effect_windows2_LR(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin)
```

```
esp_df = effect_windows3_LR(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin)
```

##### Inputs
- `data` (similar to `featuresExtended`): A dataframe with cleaned exposures together with columns from the `otherVariables` dataset and the `outcome` variable.
- `X_name`: List with the names of the columns in `data` corresponding to the exposures.
- `Y_name`:  Names of the column in `data` corresponding to the outcome.
- `confound_names`: List of names of covariates considered as confounders. The function will adjust for confounding effects for these variables within each window.
- `modifier_names`: List of names of covariates to be explored as potential effect modifiers.
- `z0`: List of coordinates of the origin of each window. 
- `Lw`: List of dimensions of each window.
- `nmin`: Minimum number of observations within a window for effect sizes to be estimated.

##### Outputs
`esp_df`:  A dataframe with a row for each window of the cover used to sample the covariate space. 

For a given window, the datagrame gives the following columns:
* `nobs`: Number of observations within the window.
* `BMI_z0`: Origin of the window.
* `BMI_Lw`: Length of the window.
* M columns giving the effect sizes $\{e_m\}_{m=1}^M$ within the window for each metabolite.


### Plotting the effect size for the cover windows of an exposure variable and a covariate

```
Eff_size_Windows_plot(esp_df,variable,modifier,no_effect_value,errorbar)
```

#### Inputs
* `esp_df`:  A dataframe with a row for each window of the cover used to sample the covariate space (see full description in the section "Estimating the effect size profile (ESP)"). 
* `variable`: Exposure variable whose window effect size will be plotted vs. a covariate which is considered as a potential effect size modifier.
* `modifier`: Continuous covariate explored as potential effect size modifier.
* `no_effect_value`: Value where effect size is absent. Set to 0 if the effect size corresponds to the slope of a linear regression model. Set to 1 if the effect size corresponds to the odds ratio of a logistic regression model. 
- `errorbar`: If set to "Y", it indicates the length of windows with error bars. If set to any other value (e.g. "N"), only the midpoints of the window segments are plotted.
  
#### Outputs
A plot of the effect size of the association between the `variable` and the outcome for different windows of the `modifier` covariate. 

### Clustering indices analysis

```
index_vs_K_df,KoptimalCH,KoptimalDB,KoptimalSil,KoptimalElbow,koptimal_overall = Clustering_indices(features_df,kmax,cluster_method,plotYN)
```

### Inputs
* `features_df`: Dataframe with the features (effect size profile) used to describe each element (window) to be clustered.
* `kmax`: Maximum number of clusters to be explored.
*  `cluster_method` ("Agglomerate" or "Kmeans"): Clustering method.
* `plotYN` (Y/N): If set to "Y", a plot for each of the clustering measures as a function of the number of clusters is displayed.

#### Outputs
* `index_vs_K_df`: A dataframe with the value of the clustering measures as a function of the number of clusters.
* `KoptimalCH`: Number of clusters at the maximum of the Calinski-Harabasz measure.
* `KoptimalDB`: Number of clusters at the maximum of the Davies-Bouldin measure.
* `KoptimalSil`: Number of clusters at the maximum of the silhouette measure.
* `KoptimalElbow`: Elbow of the inertia vs. the number of clusters.
* `koptimal_overall`: Most frequently occurring value among the four clustering indices. If there is no repeated number across the indices or in case of a tie, we will use the smallest number of clusters. Agglomerative clustering will be used throughout the article.
  
### Visualisation of the clusters in the effect size space using two principal components

```
x_pca,cumVar = ESP_pca(features_df,cluster_method,plotYN,pcomp_1,pcomp_2,n_clusters,clusterOrder)
```

#### Inputs
* `features_df`: Dataframe with the features (effect size profile) used to describe each element (window) to be clustered.
*  `cluster_method` ("Agglomerate" or "Kmeans"): Clustering method.
* `plotYN` (Y/N): If set to "Y", a plot for each of the clustering measures as a function of the number of clusters is displayed.
* `pcomp_1`: Integer giving the first principal component.
* `pcomp_2`: Integer giving the second principal component.
* `n_clusters`: Number of clusters. Windows within different clusters are represented with different colours.
* `clusterOrder`:  The order of clusters is arbitrary. This variable accepts a list of integers giving the label we wish for each cluster.

#### Outputs
* `x_pca`: Coordinates of the projection of each feature on the principal component directions. 
* `cumVar`: array giving the cumulative explained variance for the principal components.

### Obtaining a list of cluster labels for the window

An array with the cluster label for each window can be obtained with the following function:

```
labels = ESPClust.Window_clusters_labels(features_df,n_clusters,cluster_method,clusterOrder)
```

#### Inputs
* `features_df`: Dataframe with the features (effect size profile) used to describe each element (window) to be clustered.
* `n_clusters`: Number of clusters. Windows within different clusters are represented with different colours.
*  `cluster_method` ("Agglomerate" or "Kmeans"): Clustering method.
* `clusterOrder`:  The order of clusters is arbitrary. This variable accepts a list of integers giving the label we wish for each cluster.

#### Outputs
`lables`: An array of cluster labels for each window.


### Plot of clusters in the covariate space

The effect size profile clusters can be visualised in the covariate space using this function (windows are represented by their midpoint):

```
plot_clusters_CovSpace(esp_df,X_name,modifier_names,n_clusters,cluster_method,clusterOrder)
```

#### Inputs
* `esp_df`: A dataframe with a row for each window of the cover used to sample the covariate space (see full description in the section "Estimating the effect size profile (ESP)"). 
* `X_name`: List with the names of the columns in `data` corresponding to the exposures.
* `modifier_names`: List of names of covariates to be explored as potential effect modifiers.
* `n_clusters`: Number of clusters. Windows within different clusters are represented with different colours.
* `cluster_method` ("Agglomerate" or "Kmeans"): Clustering method.
* `clusterOrder`:  The order of clusters is arbitrary. This variable accepts a list of integers giving the label we wish for each cluster.
  
#### Outputs
Plots are provided which depend on the number of effect modifiers considered:
* One effect modifier: A 2D scatterplot with the value of the modifier in the horizontal axis.
* Two effect modifiers: A 2D scatterplot with each modifier represented along each of the axes.
* Three effect modifiers: Two 2D scatterplots representing clusters in the space spanned by pairs of covariates. One 3D plot with axes corresponding to each of the covariates.

### Cluster centroids and clustering inertia

The coordinates of the centroids of the clusters, their dispersion and inertia of the clustering are provided by the following function:

```
centroids,centroids_SD,inertia = ESPClust.Window_cluster_centroids(features_df,n_clusters,cluster_method,plot,scale_ErrorSD,clusterOrder,diff_sorted)
```

#### Inputs:
* `features_df`: Dataframe with the features (effect size profile) used to describe each element (window) to be clustered.
* `n_clusters`: Number of clusters. Windows within different clusters are represented with different colours.
* `cluster_method` ("Agglomerate" or "Kmeans"): Clustering method.
* `clusterOrder`:  The order of clusters is arbitrary. This variable accepts a list of integers giving the label we wish for each cluster.
* `plot` ('none', 'errorbar',  'lines', 'points', 'points-lines'): A plot is not provided if this variable is set to 'none'. For any other option, a plot is provided. `errorbar` plots the coordinates and their standard deviation (multiplied by a factor `scale_ErrorSD`). If the variation of the coordinates is not required, `lines`, `points`, `points-lines` can be used to plot the coordinates of the cluster centroids in three different ways.
* `scale_ErrorSD`: If `plot = errorbar`, the error bar is the standard deviation scaled by a factor `scale_ErrorSD`. For `scale_ErrorSD = 1`, the error bars correspond to one standard deviation.
* `diff_sorted` (Y/N): Whether the coordinates (i.e. exposures) should be sorted to show those that differ the most between clusters first.

#### Outputs:
* `centroids`: Dataframe giving the coordinates of the cluster centroids. It has one column for each exposure and a row for each cluster.
* `centroids_SD`: Dataframe giving the standard deviation of the coordinates of the cluster centroids. It has one column for each exposure and a row for each cluster.
* `inertia`: Inertia of the clustering. This is the sum of the quadratic distance from each exposure to the cluster centroid.
