# Aim

- Implement [this](https://arxiv.org/pdf/1703.07015.pdf) state of the art time series forecasting model in MXNet.
- [data](https://github.com/laiguokun/multivariate-time-series-data)

## To do

1. What to do with states from RNN cells (learning to do here)
2. Does my skip-connection between rnn cells do what it should?
2. Which side of the input gets padded?????? I am currently padding future side (bottom of array)\
4. Confirm AR component is correct
5. If learning rate too high I get overflow error (<= 0.000001).  Need smaller learning rate for AR component.  Consider multiplying gradient here to overcome this.  Or other optimizers such as Adam could help this.
6. Model in paper predicts all ts next values, mine just predicts a single series.  Consider which makes more sense...
7. Do I need batch normalization???
