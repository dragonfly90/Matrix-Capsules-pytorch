# Matrix-Capsules-pytorch
This is a pytorch implementation of [Matrix Capsules with EM routing](https://openreview.net/pdf?id=HJWLfGWRb)

In ```Capsules.py```, there are two implemented classes: ```PrimaryCaps``` and ```ConvCaps```.
The ClassCapsules in the paper is actually a special case of ```ConvCaps``` with whole receptive field, transformation matrix sharing and Coordinate Addition.

In ```train.py```, I define a CapsNet in the paper using classes in ```Capsules.py```, and could be used to train a model for MNIST dataset.

## Train a small CapsNet on MNIST
```python train.py -batch_size=64 -lr=2e-2 -num_epochs=5 -r=1 -print_freq=5```.

Note:
more args can be found in ```utils.py```, and if you want to change A,B,C,D, go to ```line 62``` of ```train.py```

## Results
The test accuracy is around 97.6% after 1 epoch and 98.7% after 2 epochs of training with a small Capsule of A,B,C,D,r = 64,8,16,16,1. After 30 epochs of training, the best acc is around 99.3%. More results on different configurations are welcomed.

## Time

Matrix-Capsules-EM-Tensorflow: 

https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow

Original: 

4 epochs

batch:935, loss:0.0025, acc:64/64

Epoch4 Train acc:0.99175

4*4

Testing...

Epoch4 Test acc:0.9872

time duration:  1927.7292823791504



Speed up:
Epoch4 Test acc:0.9877
time duration:  1948.444207906723

3*3 matrix capsule

Epoch4 Test acc:0.9866

time duration:  1801.0306959152222

5*5 matrix capsule

Epoch4 Test acc:0.9869

time duration:  2110.740709543228

6*6 matrix capsule

Epoch4 Test acc:0.9825

time duration:  2382.0352709293365
 
40mins

iterations: 5

Epoch4 Train acc:0.09871666666666666
Testing...
Epoch4 Test acc:0.098
time duration:  5617.123788833618


# third_tensor = torch.cat((first_tensor, second_tensor), 0)

https://github.com/www0wwwjs1

1.5k iteration, time: 23:37:08-23:49:51
accuracy: 0.94

## TODO
* using more matrix operation rather than ```for``` iteration in E-step of ```Capsules.py```.
* make capsules work when height_in != width_in
* find better lambda/m schedule to speed up the convergence.



