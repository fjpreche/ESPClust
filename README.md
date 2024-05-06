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
- `featTransform` (`Plus1Log`, ????): Transformation of the data.
- `plotYN`: If set to 'Y', plots are provided for the array of missing exposures and the missing percentage for exposures and individuals.

#### Outputs
- `featuresExtended`: a dataframe with cleaned exposures together with columns from the `otherVariables` dataset and the `outcome` variable.
- `exposure_names`: Names of the exposures that were kept after cleaning.
- 
### Creating a cover for the covariate space with windows of homogeneous size
```
z0, Lw = Homogeneous_Windows(data,modifier_names,Delta_All,L_All)
```
#### Inputs
- `data`
- `modifier_names`
- `Delta_All`
- `L_All`

#### Outputs
- `z0`: List of coordinates of the origin of each window. 
- `Lw`: List of dimensions of each window.

### Estimating the effect size profile (ESP) for a continuous outcome using linear regression

Three functions are defined to deal with 1, 2 or 3 potential ESP modifiers:

```
esp_df = effect_windows1_lin(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin)
```

```
esp_df = effect_windows2_lin(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin)
```

```
esp_df = effect_windows3_lin(data,X_name,Y_name,confound_names,modifier_names,z0,Lw,nmin)
```

#### Inputs
- `data` (similar to `featuresExtended`): A dataframe with cleaned exposures together with columns from the `otherVariables` dataset and the `outcome` variable.
- `X_name`: List with the names of the columns in `data` corresponding to the exposures.
- `Y_name`:  Names of the column in `data` corresponding to the outcome.
- `confound_names`: List of names of covariates considered as confounders. The function will adjust for confounding effects for these variables within each window.
- `modifier_names`: List of names of covariates to be explored as potential effect modifiers.
- `z0`: List of coordinates of the origin of each window. 
- `Lw`: List of dimensions of each window.
- `nmin`: Minimum number of observations within a window for effect sizes to be estimated.

#### Outputs
`esp_df`:  A dataframe with a row for each window of the cover used to sample the covariate space. 

For a given window, the datagrame gives the following columns:
* `nobs`: Number of observations within the window.
* `BMI_z0`: Origin of the window.
* `BMI_Lw`: Length of the window.
* M columns giving the effect sizes $\{e_m\}_{m=1}^M$ within the window for each metabolite.


### Plotting the effect size for the cover windows 
```
Eff_size_Windows_plot(esp_df,variable,modifier,no_effect_value,errorbar)
```

### Clustering indices analysis

```
index_vs_K_df,KoptimalCH,KoptimalDB,KoptimalSil,KoptimalElbow,koptimal_overall = Clustering_indices(distance,kmax,cluster_method,plotYN)
```

### Principal component visualisation of the clusters in the effect size space

```
x_pca,cumVar = ESP_pca(esp_df,X_name,cluster_method,plot_yn,pcomp_1,pcomp_2,n_clusters,clusterOrder)
```

### Cluster labels

```
labels = Window_clusters_labels(distance,n_clusters,cluster_method,clusterOrder)
```

### Plot of clusters in the covariate space
Using windows midpoints...
```
plot_clusters_CovSpace(esp_df,X_name,modifier_names,n_clusters,cluster_method,clusterOrder)
```
