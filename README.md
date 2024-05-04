# ESPClust
Unsupervised identification of modifiers for the effect size spectrum in omics association studies

`data_cleaning` which takes the following inputs:
- `featuresIn`: exposures dataframe.
- `Yin`: outcome.
- `thmissing`: A threshold value in [0,1]. Metabolites and individuals whose percentage of missing values is above this threshold Threshold for discarding metabolites or individuals with missing values.
- `otherVariables`: A dataframe with with covariates.
- `k_neighbours`: Number of neighbours for nearest neighbour imputation.
- `featTransform` (`Plus1Log`, ????): Transformation of the data.
- `plotYN`: If set to 'Y', plots are provided for the array of missing exposures and the missing percentage for exposures and individuals.
