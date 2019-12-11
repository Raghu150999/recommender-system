# recommender-system
Implementing different approaches for recommendation systems

## Usage
### Collaborative Filtering
run: `python cf.py`
Returns the RMSE and MAE loss metrics on test data using three different approaches of collaborative filtering namely:
1. user-user filtering
2. item-item filtering
3. baseline approach

### SVD
run: `python svd.py`

Performs Singular Value Decomposition on the given utility matrix and report the reconstruction error (RMSE and MAE loss) at the specified energy.

### CUR
run: `python cur.py`

Similar to SVD, perform decomposition and reports reconstruction error for the specified `r` value.
`r` is the parameter which specifies the number of columns and rows in C and R matrix respectively in CUR.

### Latent Factor model
run: `python main.py`

Predicts user-movie rating using the latent factor model. Implemented using Stochastic Gradient Descent learns the latent (hidden) factors for each user and movie and along with baseline approximation computes the prediction.

## Hyperparameters:

`alpha (learning rate) = 0.01`

`beta (regularisation coefficient) = 0.05`

`epochs = 50`

## Results

#### Tuning Latent Factors
Latent Factors |    RMSE (test)       |     MAE (test)
---------------|----------------------|--------------------
10             | 0.836                | 0.654
20             | 0.840                | 0.657
50             | 0.829                | 0.649
100            | 0.833                | 0.653


#### Collaborative Filtering 

CF Approach         |     RMSE (test)       |   MAE (test)
--------------------|-----------------------|---------------------
Baseline            | 0.904                 | 0.724
user-user filtering | 1.147                 | 0.831
item-item filtering | 0.921                 | 0.730

#### SVD 

Energy      |   RMSE (test)         |   MAE (test)
------------|-----------------------|------------------------
100         | 0                     | 0
90          | 0.243                 | 0.132

#### CUR

r      |   RMSE (test)      |   MAE (test)
-------|--------------------|--------------------
3000   | 0.615              | 0.205
2000   | 2.109              | 0.276

### Loss curve for Latent Factor model

##### Using 50 latent factors:

![Figure 1-1](plots/loss.png?raw=true)
