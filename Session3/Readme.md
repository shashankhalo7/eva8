# Session 3
## Part1 . Neural Networks in Excel

# Steps : 
![calculations](images/neural_network.png)


* h1 = w1\*i1 + w2\*i2

* h2 = w3\*i1 + w4\*i2

* a_h1 = sigmoid(h1)

* a_h2 = sigmoid(h2)

* o1 = w5\*a_h1 + w6\*a_h2 

* o2 = w7\*a_h1 + w8\*a_h2

* a_o1 = sigmoid(o1)

* a_o2 = sigmoid(o2)

* E1 = 1/2\*(t1-a_o1)^2

* E2 = 1/2\*(t2-a_o2)^2

* E_total = E1 + E2

* sigmoid(x)  = 1/(1+exp(-x))

## Gradient calculations(step by step in excel file)

### Gradient of E_total wrt w5,w6,w7 and w8(Layer 2 Weights)
* ðE_t/ðw5 = (a_o1-t1) \* (a_o1 \*(1-a_01)) \* a_h1

* ðE_t/ðw6 = (a_o1-t1) \* (a_o1 \*(1-a_01)) \* a_h2

* ðE_t/ðw7 = (a_o2-t2) \* (a_o2 \*(1-a_02)) \* a_h1

* ðE_t/ðw7 = (a_o2-t2) \* (a_o2 \*(1-a_02)) \* a_h2


### Gradient of E_total wrt a_h1 and a_h2(Layer 1 activations)
* ðE_t/ða_h1 = ((a_o1-t1)\*(a_o1\*(1-a_o1))\*w5) + ((a_o2-t2)\*(a_o2\*(1-a_o2))\*w7)

* ðE_t/ða_h2 = ((a_o1-t1)\*(a_o1\*(1-a_o1))\*w6) + ((a_o2-t2)\*(a_o2\*(1-a_o2))\*w8)


### Gradient of E_total wrt w1,w2,w3 and w4(Layer 1 Weights)
* ðE_t/ðw1 = (((a_o1-t1)\*(a_o1\*(1-a_o1))\*w5) + ((a_o2-t2)\*(a_o2\*(1-a_o2))\*w7)) \* a_h1\*(1-a_h1) \* i1

* ðE_t/ðw2 = (((a_o1-t1)\*(a_o1\*(1-a_o1))\*w5) + ((a_o2-t2)\*(a_o2\*(1-a_o2))\*w7)) \* a_h1\*(1-a_h1) \* i2

* ðE_t/ðw3 = (((a_o1-t1)\*(a_o1\*(1-a_o1))\*w6) + ((a_o2-t2)\*(a_o2\*(1-a_o2))\*w8)) \* a_h2\*(1-a_h2) \* i1 

* ðE_t/ðw4 = (((a_o1-t1)\*(a_o1\*(1-a_o1))\*w6) + ((a_o2-t2)\*(a_o2\*(1-a_o2))\*w8)) \* a_h2\*(1-a_h2) \* i2 



## Results 
### 1. lr = 0.1
**Weights and Gradients**
![Weights and Gradients lr = 0.1](images/backprop_lr_0.1.png)
**Loss Plot**
![Loss Plot lr = 0.1](images/loss_plot_lr_0.1.png)
### 2. lr = 0.2
**Weights and Gradients**
![Weights and Gradients lr = 0.2](images/backprop_lr_0.2.png)
**Loss Plot**
![Loss Plot lr = 0.2](images/loss_plot_lr_0.2.png)
### 3. lr = 0.5
**Weights and Gradients**
![Weights and Gradients lr = 0.5](images/backprop_lr_0.5.png)
**Loss Plot**
![Loss Plot lr = 0.5](images/loss_plot_lr_0.5.png)
### 4. lr = 0.8
**Weights and Gradients**
![Weights and Gradients lr = 0.8](images/backprop_lr_0.8.png)
**Loss Plot**
![Loss Plot lr = 0.8](images/loss_plot_lr_0.8.png)
### 5. lr = 1.0
**Weights and Gradients**
![Weights and Gradients lr = 1](images/backprop_lr_1.png)
**Loss Plot**
![Loss Plot lr = 1](images/loss_plot_lr_1.png)
### 6. lr = 2.0
**Weights and Gradients**
![Weights and Gradients lr = 2](images/backprop_lr_2.png)
**Loss Plot**
![Loss Plot lr = 2](images/loss_plot_lr_2.png)

## Part2

The task was to create a train a CNN network that can classify MNIST data with the following constraints:
* 99.4% validation accuracy
* Less than 20k Parameters
* You can use anything from above you want. 
* Less than 20 Epochs
* No fully connected layer
 

## Final Model 
![](images/parameters.png)

## Parameters
**19,290**

## Best Test Accuracy
**99.47%**
![](images/metrics.png)

## Approach
In the base network there were around 6.3 million parameters and after training it for 19 epochs it got around 99.25%

In order to train a model with less than 20k parameters and satify the above constraints I did the following changes

* **Batch Normalization** : Adding batch normaliztion layer after every convolution layer.
* **Drop Out** : Dropout layer adds a regularizing effect. It would have been beeter if Cutout was used as dropout is mostly for 1d data. Used it only twice.After the maxpool layers. There was no logic in that, it was more of a performance based decision(Got more accuracy after maxpool layers).
* **1x1 Convolutions** : Used 1x1 Counvolution layer twice after Maxpooling layers to reduce channels, combine and mix channel inforamtion and reduce load over subsequent 3x3 convolutions
* **GAP** : Used global average pooling after channel size reached 7x7 to flatten the 2d channels. Didn't use FC layers. 
* **Remove ReLU before the last layer** : ReLU, Maxpooling and Dropout should not be present before the last layer(log_softmax)
* **Kernels** : Started from 16 went till 32 then used 1X1 to reduce channels. After that kept the kernel count to 16 to reduce the parameter count. As the problem was relatively simpler, as inter class variation was significant(variantion between the number (0,1,2,..9)) and intra class variation was low (inside a class i.e variations of 1's or 2's) this problem can be solved with relatively lesser number of kernels.

