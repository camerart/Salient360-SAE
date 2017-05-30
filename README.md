# Grant Challenge: Salient360!


## Brief Introduction

The code are for the grand challenge Salient360! at ICME 2017. Two computational
models have been implemented in the code, which are:

1. Head motion based saliency model (Model type 1); and
2. Head and eye-motion based saliency model (Model type 2).

The corresponding functions for model types 1 and 2 are `HeadSalMap` and
`HeadEyeSalMap`, respectively.

## Source code files

To use the code, one need to do the following steps:

- Decompress the file `salient360_XDQS.tar.gz` to a folder `<saliency_source>`. 
- Create two sub-folders, "`images`" and "`saliency`" under `<saliency_source>`.
- Move images to be processed to "`<saliency_source>/images`".
- Execute the matlab script "`process.m`" with command line "`matlab < process.m`".
- Enter the folder "`<saliency_source>/saliency`" to check for results.

After the execution of the script, results will be stored in
"`<saliency_source>/saliency`". The suffix for model types 1 and 2 are "`_SH`"
and "`_SHE`", respectively. For instance, two files, "`P10_SH.bin`" and
"`P10_SHE.bin`" will be generated after processing the image "`P10.jpg`".

The following three files are the major entries to the functions:

- `processing.m`: This matlab script for processing all images under the folder
  "images".
  
- `HeadSalMap.m`: This file implements the function `HeadSalMap`, which
                  estimates the saliency map of model type 1. Its input and
                  output arguments are:
     
  + `imgIn`: the input equirectangular image organized in an RGB, with
             `size(imgIn)` being `[Height,Width,3]`.
  + `matOut`: the output "double" matrix having the saliency values. Its size is
              `[Height,Width]`
	
- `HeadEyeSalMap.m`: This file implements the function `HeadEyeSalMap`, which
                     estimates the saliency of model type 2. Its input and
                     output arguments are:
     
  + `imgIn`: the input equirectangular image organized in an RGB, with
             `size(imgIn)` being `[Height,Width,3]`.
  + `matOut`: the output "double" matrix having the saliency values. Its size is
              `[Height,Width]`

## Team

Our team can be referenced as **XDQS**, with team members listed as below:

- Fei Qi (齐飞)
- Chunhuan Lin (林春焕)
- Zhaohui Xia (夏朝辉)
- Chen Xia (夏辰)
- Hao Li (李昊)
- Guangming Shi (石光明)


## Method

The approach is based on our previous publication [1], which employs a stacked
auto-encoder-based reconstruction framework.

[1] Chen Xia, Fei Qi, Guangming Shi, "Bottom-up Visual Saliency Estimation with
    Deep Autoencoder-based Sparse Reconstruction," IEEE Transactions on Neural
    Networks and Learning Systems, 27(6): 1227–1240, June 2016.
	doi: 10.1109/TNNLS.2015.2512898
