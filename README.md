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

## Results

#### Tuning Latent Factors
Latent Factors |    RMSE (test)       |     MAE (test)
---------------|----------------------|--------------------
10             | 0.8362632368281437   | 0.6541915961250121
20             | 0.8409508999352776   | 0.657184615805829
50             | 0.8298577652916366   | 0.6492783690947389
100            | 0.8333043309739515   | 0.6534675560938471


#### Collaborative Filtering 

CF Approach         |     RMSE (test)       |   MAE (test)
--------------------|-----------------------|---------------------
Baseline            | 0.9039461334116203    | 0.7246502356309865
user-user filtering | 1.1469032289592804    | 0.8313772348609321
item-item filtering | 0.9205974541607472    | 0.730551377319183

#### SVD 

Energy      |   RMSE (test)         |   MAE (test)
------------|-----------------------|------------------------
100         | 2.946690428449501e-15 | 1.8063024181862973e-15
90          | 0.24275765324185486   | 0.13180619643134991

#### CUR

r      |   RMSE (test)      |   MAE (test)
-------|--------------------|--------------------
3000   | 0.6156695183567535 | 0.20507647108696353
2000   | 2.109496316432644  | 0.27622562044814736

### Loss curve for Latent Factor model

##### Using 50 latent factors:

![Figure 1-1](plots/loss.png?raw=true)
