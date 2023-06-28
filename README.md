# A PyTorch implement for DenseNet

- The structure of a DenseNet model:

![](F:\jupyterfile\DenseNet\images\DensNet.png)

- to train a DenseNet without Bottleneck layer and Compression:

```
python train.py --layer 40 --growth 12 --no-bottleneck --reduce 1.0 --name DenseNet-40-12
```



- to train a DenseNet with Bottleneck layer and Compression:

```
! python train.py --layers 40 --growth 12 --reduce 0.5 --name DenseNet-BC-40-12 --tensorboard
```

