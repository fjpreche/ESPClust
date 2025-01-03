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

### 1. Data cleaning

#### Function: `data_cleaning`

##### Description
The `data_cleaning` function preprocesses and cleans a dataset of exposures, outcome variables, and covariates by addressing missing values, scaling features, and optionally transforming data. It ensures that exposures and observations with excessive missingness are removed and imputes remaining missing values using nearest neighbors.

---

##### Inputs
- **`featuresIn`**:  
  Dataframe of exposures (metabolites or other features) to be cleaned.

- **`Yin`**:  
  Dataframe containing the outcome variable.

- **`thmissing`**:  
  A threshold value between `[0, 1]` that determines the maximum allowable percentage of missing values for metabolites and individuals.  
  - Metabolites and individuals with missing values exceeding this threshold are discarded.

- **`otherVariables`**:  
  Dataframe of covariates (additional variables related to the dataset).

- **`k_neighbours`**:  
  Number of neighbors to use for imputation of missing values using the nearest neighbors algorithm.

- **`featTransform`** (`None` or `Plus1Log`):  
  Transformation method for exposures:  
  - If set to `None`, no transformation is applied.  
  - If set to `Plus1Log`, exposures are transformed by adding 1 and applying the natural logarithm.

- **`plotYN`** (`'Y'` or `'N'`):  
  Flag to control whether to generate plots:  
  - `'Y'`: Generate visualizations of missing data patterns.  
  - `'N'`: Skip plotting.

---

##### Outputs
- **`featuresExtended`**:  
  A dataframe containing the cleaned exposures, covariates (`otherVariables`), and the outcome variable. Missing values are imputed, features are scaled, and columns from `otherVariables` are appended.

- **`exposure_names`**:  
  A list of names of exposures retained after cleaning.

---

##### Steps Performed
1. **Missing Value Analysis**:  
   - Calculates the total number of missing values and their proportions for both exposures and individuals.
   - Removes exposures and individuals exceeding the missingness threshold (`thmissing`).

2. **Visualization (Optional)**:  
   - If `plotYN` is set to `'Y'`, generates plots:  
     - Array of missing exposures.  
     - Percentage of missing values for exposures and individuals.

3. **Imputation**:  
   - Performs nearest neighbors imputation on remaining missing values using the specified `k_neighbours`.

4. **Feature Transformation**:  
   - If `featTransform` is `Plus1Log`, applies the transformation:  
     \[
     \text{transformed\_feature} = \log(\text{feature} + 1)
     \]

5. **Scaling**:  
   - Standardizes all features (z-score normalization).

6. **Combining Data**:  
   - Appends the outcome variable (`Yin`) and covariates (`otherVariables`) to the cleaned features.

7. **Final Cleanup**:  
   - Drops rows containing any remaining missing values.

---

##### Example Usage
```python
# Example inputs
featuresIn = pd.DataFrame({
    "Metabolite1": [1, np.nan, 3],
    "Metabolite2": [np.nan, 5, 6],
    "Metabolite3": [7, 8, 9]
})
Yin = pd.DataFrame({"label": [0, 1, 0]})
otherVariables = pd.DataFrame({
    "Covariate1": [10, 11, 12],
    "Covariate2": [20, 21, 22]
})
annotation = pd.DataFrame({"met_labels": ["Metabolite1", "Metabolite2", "Metabolite3"]})
thmissing = 0.2
k_neighbours = 2
featTransform = "Plus1Log"
plotYN = "N"

# Call the function
featuresExtended, exposure_names = data_cleaning(featuresIn, Yin, otherVariables, annotation, thmissing, k_neighbours, featTransform, plotYN)

print("Cleaned Dataframe:")
print(featuresExtended)
print("Exposure Names:", exposure_names)
```

### 2. Building a cover of the covariate space with gliding windows
This is achieved through the functions `window_parameters` and `Homogeneous_Windows`

#### 2.1. Function: `window_parameters`

##### Description
This function calculates the dimensions (`L`) and gliding steps (`Delta`) of the gliding windows based on the data distribution. These parameters or other set by the analyst are used as input for the `Homogeneous_Windows` function to systematically explore the covariate space.  

##### Inputs
- **`data1`**: A DataFrame containing the covariates for which window dimensions and gliding steps need to be calculated.  
- **`nmin`** *(optional)*: The minimum number of observations required in each window (default = 10).  
- **`CL`** *(optional)*: The confidence level for estimating window dimensions, used to compute the quantile of the empirical window size distribution (default = 0.95). Must be a value between 0 and 1.

##### Outputs
- **`L`**: A list of window lengths for each covariate in `data1`.  
- **`Delta`**: A list of gliding step sizes for each covariate in `data1`.  

##### Example Usage
```python
# Example input data
data1 = pd.DataFrame({
    "Z1": np.random.normal(size=100),
    "Z2": np.random.uniform(size=100)
})

# Calculate window parameters
L, Delta = window_parameters(data1, nmin=20, CL=0.9)

print("Window Dimensions (L):", L)
print("Gliding Steps (Delta):", Delta)
```

#### 2.2. Function: `Homogeneous_Windows`

##### Description
The `Homogeneous_Windows` function systematically divides a J-dimensional covariate space into gliding hyperrectangles (windows) of specified dimensions (`L_All`) and step sizes (`Delta_All`). This creates a grid-like structure that allows for the exploration of local heterogeneities in effect sizes.  

##### Inputs
- **`data`**: A DataFrame containing the dataset with covariates of interest.  
- **`modifier_names`**: A list of column names in the `data` DataFrame, representing the covariates to be used for window generation.  
- **`L_All`**: A list of window dimensions (lengths) for each covariate.  
- **`Delta_All`**: A list of gliding step sizes for each covariate.  
- **`var_type`**: A list specifying the type of each covariate:  
  - `'c'` for continuous covariates.  
  - Other values can be used for categorical covariates (e.g., `'d'` for discrete).  

##### Outputs
- **`z0`**: A list of origin coordinates for each window in the covariate space. These represent the starting points of the windows for each dimension.  
- **`Lw`**: A list of dimensions (lengths) for each window along each covariate.  
- **`win_dim`**: A list of the number of windows generated along each covariate dimension.  
- **`n_windows`**: The total number of windows generated across all dimensions, calculated as the product of `win_dim`.

##### Example Usage
```python
# Example input data
data = pd.DataFrame({
    "Z1": np.random.normal(size=100),
    "Z2": np.random.uniform(size=100)
})

modifier_names = ["Z1", "Z2"]
L_All = [0.5, 0.3]  # Window dimensions for Z1 and Z2
Delta_All = [0.1, 0.1]  # Gliding steps for Z1 and Z2
var_type = ['c', 'c']  # Both covariates are continuous

z0, Lw, win_dim, n_windows = Homogeneous_Windows(data, modifier_names, L_All, Delta_All, var_type)

print("Window Origins (z0):", z0)
print("Window Dimensions (Lw):", Lw)
print("Number of Windows per Dimension (win_dim):", win_dim)
print("Total Number of Windows:", n_windows)
```

### 3. Estimating the effect size profile (ESP)

#### Function: `effect_windows`

##### Description
The `effect_windows` function calculates effect sizes for exposures within windows defined across a covariate space. This is a flexible function that supports user-defined methods (`effsize_method`) for effect size estimation, allowing for various types of outcomes and modeling strategies. The windowing process is guided by modifier variables that define the dimensions of the covariate space.

---

##### Inputs
- **`data`**:  
  A dataframe containing cleaned exposures, confounders, effect modifiers, and the outcome variable.

- **`X_name`**:  
  A list of column names in `data` corresponding to the exposures (predictors) for which effect sizes will be estimated.

- **`Y_name`**:  
  The name of the column in `data` corresponding to the outcome variable.

- **`confound_names`**:  
  A list of names of covariates to adjust for as confounders. These variables are accounted for in the effect size estimation.

- **`modifier_names`**:  
  A list of names of covariates defining the dimensions of the windows (effect modifiers). These modifiers determine how the covariate space is partitioned.

- **`z0`**:  
  A list of coordinates for the origin of each window in each modifier's dimension.

- **`Lw`**:  
  A list of lengths (dimensions) of the windows along each modifier dimension.

- **`nmin`**:  
  The minimum number of observations required within a window for effect sizes to be estimated. This ensures statistical reliability.

- **`effsize_method`**:  
  A user-specified function to calculate the effect size. The function must accept two arguments:  
  1. `y`: The outcome variable.  
  2. `xdf`: A dataframe containing confounders and the exposure of interest.  
  It should return a single numeric value representing the effect size. Two examples are provided below:  
  - **`effsize_lin`**: For linear regression-based effect sizes.  
  - **`effsize_logit_odds`**: For logistic regression-based odds ratios.

---

##### Outputs
- **`esp_df`**:  
  A dataframe summarizing the results for each window, with the following columns:
  - `nobs`: The number of observations within the window.
  - Modifier-specific columns:  
    - `<modifier_name>_z0`: The origin coordinate of the window along the modifier.  
    - `<modifier_name>_Lw`: The length of the window along the modifier.  
  - Effect sizes: One column for each exposure variable in `X_name`, representing the scaled effect sizes within the window.

---

##### Example Usage
```python
# Example data
data = pd.DataFrame({
    "Exposure1": np.random.normal(size=100),
    "Exposure2": np.random.uniform(size=100),
    "Outcome": np.random.randint(0, 2, size=100),
    "Confounder1": np.random.normal(size=100),
    "Modifier1": np.random.normal(size=100),
    "Modifier2": np.random.uniform(size=100)
})

# Inputs
X_name = ["Exposure1", "Exposure2"]
Y_name = "Outcome"
confound_names = ["Confounder1"]
modifier_names = ["Modifier1", "Modifier2"]
z0 = [[0, 0.5, 1.0], [0, 0.5, 1.0]]  # Example window origins
Lw = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]  # Example window lengths
nmin = 10

# Example using effsize_lin
esp_df_lin = effect_windows(data, X_name, Y_name, confound_names, modifier_names, z0, Lw, nmin, effsize_lin)

# Example using effsize_logit_odds
esp_df_logit = effect_windows(data, X_name, Y_name, confound_names, modifier_names, z0, Lw, nmin, effsize_logit_odds)
```

#### Function: `effsize_lin`

##### Description
The `effsize_lin` function calculates the effect size of an exposure on an outcome using linear regression. The function assumes that the last column in the input dataframe (`xdf`) is the exposure variable of interest, while the other columns represent confounders.

---

##### Inputs
- **`y`**:  
  The outcome variable (continuous or discrete). This can be a list, numpy array, or pandas Series.

- **`xdf`**:  
  A dataframe containing confounders and the exposure variable.  
  - Each row corresponds to an observation.  
  - The last column is treated as the exposure variable.

---

##### Outputs
- **`effsize`**:  
  The regression coefficient of the exposure variable, representing its effect size.

---

##### Example Usage
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Example input
y = [10, 12, 14, 13, 15]
xdf = pd.DataFrame({
    "Confounder1": [1, 2, 3, 4, 5],
    "Confounder2": [10, 11, 12, 13, 14],
    "Exposure": [5, 6, 7, 8, 9]
})

# Calculate effect size
effsize = effsize_lin(y, xdf)
print(f"Effect size (linear regression): {effsize}")
```

#### Function: `effsize_logit_odds`

##### Description
The `effsize_logit_odds` function calculates the odds ratio of an exposure on a binary outcome using logistic regression. The function assumes that the last column in the input dataframe (`xdf`) is the exposure variable of interest, while the other columns represent confounders.

---

##### Inputs
- **`y`**:  
  The binary outcome variable (values should be 0 or 1). This can be a list, numpy array, or pandas Series.

- **`xdf`**:  
  A dataframe containing confounders and the exposure variable.  
  - Each row corresponds to an observation.  
  - The last column is treated as the exposure variable.

---

##### Outputs
- **`odds_ratio`**:  
  The odds ratio of the exposure variable, representing its effect size on the binary outcome.

---

##### Example Usage
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Example input
y = [0, 1, 0, 1, 1]
xdf = pd.DataFrame({
    "Confounder1": [1, 2, 3, 4, 5],
    "Confounder2": [10, 11, 12, 13, 14],
    "Exposure": [5, 6, 7, 8, 9]
})

# Calculate odds ratio
odds_ratio = effsize_logit_odds(y, xdf)
print(f"Odds ratio (logistic regression): {odds_ratio}")
```

### 4. Plotting the effect size for the cover windows of an exposure variable and a covariate

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
