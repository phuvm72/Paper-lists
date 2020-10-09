# CBAM: Convolution Block Attention Module

This is a brief summarization of the orignal [CBAM paper](/attention/CBAM_Convolutional_Block_Attention_Module_ECCV’18.pdf)

![CBAM](/attention/figures/CBAM.JPG)

CBAM is an effective attention module which combines channel and spatial attention. It's like the extention of Squeeze and Excitation Block.

Given a feature map ![$F\in \mathbb{R}^{C\times H\times W}$](/equation/attention/cbam/1.gif), the CBAM module could perform 1D channel attention map ![$M_C \in \mathbb{R}^{C\times 1 \times 1}$](/equation/attention/cbam/2.gif) and 2D spatial attention map ![$M_S \in \mathbb{R}^{1 \times H \times W}$](/equation/attention/cbam/3.gif). We can add channel and spatial attention map in sequential or parallel, but according to the paper, applying channel attention map first then spatial attention map gives the best result.

We can summarize as follow:

![$$ F'=M_C(F)\otimes F\\ F''=M_S(F')\otimes F' $$](/equation/attention/cbam/4.gif)

where ![$\otimes$](/equation/attention/cbam/5.gif) indicates element-wise multiplication.

## Channel attention module

![channel attention](/attention/figures/channel_attention.JPG)

Channel attention focuses on 'what' is meaningful in the feature. This module will squeeze the spatial dimension of the input. It uses both average and max pooling. The author states that in Squeeze and Excitation block(SE block) only Average Pooling is used, while Max pooling also contributes the meaningful feature.

The channel attention module is computed as:

![$$ M_C(F) = \sigma(MLP(AvgPool(F)+MLP(MaxPool(F))))\\= \sigma(W_1\delta(W_0(F^c_{avg}))+W_1\delta(W_0(F^c_{max}))) $$](/equation/attention/cbam/6.gif)

where ![$\sigma$](/equation/attention/cbam/7.gif) denotes the sigmoid function, ![$\delta$](/equation/attention/cbam/11.gif) is the ReLU activation function ![$W_0\in \mathbb{R}^{\frac{C}{r}\times C}$](/equation/attention/cbam/8.gif) and  ![$W_1\in \mathbb{R}^{ C \times\frac{C}{r}}$](/equation/attention/cbam/9.gif) and r is the reduction ratio.

## Spatial attention module 

![spatial attention](/attention/figures/spatial_attention.JPG)

Spatial attention focuses on 'where' is the particular import part. They also use the same technique as Channel attention module but along the channel axis, then concatenate both AvgPool and MaxPool to generate the feature description. Finally they apply a 7x7 convolutional kernel the feature and the sigmoid non-linearity to produce the spatial attention output.

The spatial attention module is computed as:

![$$ M_S(F) = \sigma (f^{7 \times 7}([AvgPool(F);MaxPool(F)]))\\ = \sigma (f^{7 \times 7}(F^s_{avg};F^s_{max}))$$](/equation/attention/cbam/10.gif)

![resblock](/attention/figures/Resblock.JPG)

The CBAM block can be apply anywhere in the architecture just like SE block, with a little extra computation. Even though this module is just add one extra Spatial attention block and slightly adjust the Pooling method, it can achieve a remarkable result compared to SE block. They also showed that the importantance of Maxpooling features.

### Comparison of differnet channel attention methods

![different channel attention methods](/attention/figures/different_channel_methods.JPG)

### Comparison of different spatial attention methods

![different spatial attention methods](/attention/figures/different_channel_methods.JPG)

### Classification results on ImageNet

![Classification on ImageNet](/attention/figures/cbam_imagenet.JPG)

### Grad-Cam visualization 

![grad cam cbam](/attention/figures/grad_cam_cbam.JPG)