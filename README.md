# Revisiting Exploding Gradient: A Ghost That Never Leaves

This is the official implementation for our  paper
> [Revisiting Exploding Gradient: A Ghost That Never Leaves](https://www.andrew.cmu.edu/user/kaihu/Revisiting_Exploding_Gradient.pdf)
> 
> Kai Hu

## Abstract 
The exploding gradient problem is one of the main barrier to training deep neural networks. 
It is widely believed that this problem can be greatly solved by techniques such as careful weight initialization and normalization layers.  
However, we find that exploding gradients still exist in deep neural networks, and normalization layers are only able to conceal this problem. 
Our theory shows that the source of such exploding gradients does not come from the linear layer weights but non-linear activations. 
Specifically, plain networks' gradient increases exponentially with the number of nonlinear layers. 
Based on our theory, we are able to mitigate this gradient problem and train deep plain networks without any skip connection or shortcuts.
Our 50-layer plain network, SeqNet50 achieves a 77.1% top-1 validation accuracy on ImageNet, matching the performance of ResNet50.
We hope our work can provide new insights about deep neural networks.

## Get started 
Require torch>=1.7.0.
Put the ImageNet data in a "data" folder with subfolders "train" and "val".

Run the experiment by:

```
bash run.sh ( ... train script args...)
```
