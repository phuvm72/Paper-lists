# Non-local Neural Networks

This is a summarization of the [Non-local neural networks](/attention/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)

This paper address the problem of capturing the long-range dependencies data such as sequential data or video by building a non-local block. Specifically, their method outperform the current winner of Kinetics and Charades datasets (video classification datasets), and improve objection detection/segmentation and pose estimation on COCO dataset.

Non-local operation can capture long-range dependencies between two positions regardless of their positional distance. Additionally, non-local operations are efficient even with a few number of layers. Finally, this operations remain the variability of the input sizes and can be cooperated wih other operations.

In order to get a better understanding in this paper, I found that it's better to understand the 3D-convolution first. As we know, 2D-convolution is simple sliding the kernel through the image/feature. Given that intuition, we can understand 3D-convolution as sliding multiply kernel along the temporal data. In order words, we use a cubic of kernel, for example 3x3x3 cube to slide along the input frames. This will create the dependencies between frames in the data. From that we can get the important information from frames to frames.

Non-local block uses a different approach to this problem by creating feed the whole stack of frame into the networks and building the non-linearity among them by using *dot product*, *embedded Gaussian*, *concatenation* and *linear embedding*. We will go through each module one by one.

## Formulation

The authors define the non-local operation in deep neural networks as:

![$$y_i = \frac{1}{C(X)}\sum_{\forall j}f(x_i,x_j)g(x_j)$$](/equation/attention/nonlocal/1.gif) 

where:

        i is the index of an output position (in space, time or spacetime)

        j is the index that enumerates all posible positions

        x is the input signals (*image, sequence, video*); often their features

        y is the output signals

        f computes a scalar between i and all j

        g computes a representation of the input signal at position j 

        C(X) is the normalized factor

## Instantiations

In this section, the authors present various versions of ![$f$](/equation/attention/nonlocal/2.gif) and ![$g$](/equation/attention/nonlocal/3.gif). It is noticeable that the choices of these function are not affected much the overall result, but the main impact is the structure of non-local operation.

They only consider g as the linear embedding: ![$g(x_j)=W_gx_j$](/equation/attention/nonlocal/4.gif) where ![$W_g$](/equation/attention/nonlocal/5.gif) is a weight matrix to be learned

### Gaussian

![$$f(x_i,x_j)=e^{{x_i^T}x_j}$$](/equation/attention/nonlocal/6.gif)
 
Here the ![${{x_i^T}x_j}$](/equation/attention/nonlocal/7.gif) is the dot-product similarity and the normalization factor is set to ![$C(x)=\sum_{\forall j}f(x_i,x_j)$](/equation/attention/nonlocal/8.gif) . 

### Embedded Gaussian 

![$$f(x_i,x_j)=e^{\theta({x_i})^T\phi (x_j)}$$](/equation/attention/nonlocal/9.gif)

Here ![$\theta (x_i)=W_\theta x_i$](/equation/attention/nonlocal/10.gif) and ![$\phi (x_j)=W_{\phi}x_j$](/equation/attention/nonlocal/11.gif) , the normalization factor is the same as Gaussian case

This behavior as a softmax computation along the dimension j
### Dot product

![$$f(x_i,x_j)={\theta({x_i})^T\phi (x_j)}$$](/equation/attention/nonlocal/12.gif)

Normalization factor is set as ![$C(x)=N$](/equation/attention/nonlocal/13.gif) where ![$N$](/equation/attention/nonlocal/14.gif) is the number of positions in x. The normalization term is crucial because we may have different input size

### Concatenation 

![$$ f(x_i,x_j)=ReLU(w_f^T[\theta(x_i),\phi(x_j)]) $$](/equation/attention/nonlocal/15.gif)

the normalization is the same as Dot product case

## Non-local Block

The non-local block is defined as:

![$$z_i=W_zy_i+x_i$$](/equation/attention/nonlocal/16.gif)

![nonlocal block](/attention/figures/nonlocal.JPG)

Typically, they set ![$T=4, H=W=14\ or\ 7$](/equation/attention/nonlocal/17.gif)

## Experiments with various methods for different datasets

![result1](/attention/figures/result1_nonlocal.JPG)

## Compare with SOTA on Kinetics dataset

![result2](/attention/figures/result2_nonlocal.JPG)

## Adding Non-local block to MaskRCNN for COCO dataset

![result3](/attention/figures/result3_nonlocal.JPG)