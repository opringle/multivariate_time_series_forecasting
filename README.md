# Aim

- Implement [this](https://arxiv.org/pdf/1703.07015.pdf) state of the art time series forecasting model in MXNet.
- [data](https://github.com/laiguokun/multivariate-time-series-data)

## Proof of implementation

- Model in the paper predicts with h = 3 on electricity dataset
- L2 model achieves *RSE = 0.0967, RAE = 0.0581 and CORR = 0.8941* on test data

## To do

1. Which symbol in the unroll list is t=0? First or last?
2. Confirm AR component is correct? Needs regularization parameter lambda???!?!?...

## Hyperparameters...

- q = {2^0, 2^1, ... , 2^9}
- Convolutional kernel size  = {50, 100, 200}?? Or number kernels?
- Recurrent state size = {50, 100, 200}
- Skip recurrent state size = {20, 50, 100}
- p = 24h for electricity dataset
- AR lambda = {0.1,1,10}
- Adam optimizer
- Dropout after every layer =  {0.1, 0.2}
- 


