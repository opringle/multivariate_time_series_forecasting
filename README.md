# Aim

- Implement [this](https://arxiv.org/pdf/1703.07015.pdf) state of the art time series forecasting model in MXNet.
- Apply to thousands of stocks in parallel and maximize recall
- Once functioning correctly source data at a higher frequency

## To do

- Which side of the input gets padded?????? I am currently padding future
- Does my connection between cells do the same thing?
- What dataset did they use that i can benchmark against?
    - https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption


- Pass in data that is easier to model and see if network performs
- Then start mining shit loads of stock data at high frequency
- Remove pooling layer, figure out what this means for data shapes as they pass through the network
- Refine RNN layer to match paper (detail regarding time dependancy etc):
    - Forget layer between each unrolled symbol, which is learned.  this can connect t and t-10 for example where p=10.  Use a learning gate to output through or not...
    - hidden state is passed forwards. you didnt even know this was a thing...
- Add autoregressive layer:
    - linear component = AR
    - nl component = RNN + CNN
