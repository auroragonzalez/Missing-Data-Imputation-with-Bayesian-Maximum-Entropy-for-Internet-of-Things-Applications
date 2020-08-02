# Missing-Data-Imputation-with-Bayesian-Maximum-Entropy-for-Internet-of-Things-Applications

The folder "data" contains the 2 public datasets used for the papers' experiments: IBRL and BEACH. The DOCKLANDS dataset remains private.
Each folder contains the following:
  - `Rgt.csv`: The "ground truth" data. This is the real data with no modifications. It is used as input for the hard experiments and in both hard and soft for the comparison between the estimation and the real data.
  
  - `R.csv`: The modified data (with delta as explained in the paper). This is used as input for the soft experiment.
  
  - `hard.csv`: The rows of the previous datasets (`Rgt` and `R`) that 
correspond to hard sensors.

  - `soft.csv`: The rows of the previous datasets (`Rgt` and `R`) that correspond to hard soft.
  
  - `locations.txt`: The coordinates of each sensor
  
  