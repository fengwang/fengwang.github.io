---
marp: true
title: Solving inverse problems with deep convolutional neural networks
description: Feng Wang · Empa
theme: gaia
class: lead
paginate: true
_paginate: false
---


# <!--fit--> Solving inverse problems with deep convolutional neural networks
Feng Wang · Empa
12/27/2020

http://fengwang.github.io/siv/

![bg](#BDC3C7)
![](#000000)

---

## Deep learning for inverse problems

**Inverse problems**
- are some of the most important mathematical problems, but are
- usually solved by different application-specific applications.

**Deep learning**
- has shown great potential in problems across different domains,
- but is difficult to train due to their inner nonlinearity.


![bg](#BDC3C7)
![](#000000)

---

## Applications classified by dateset translation



+ 1D
  + sound waves, texts ...
+ 2D
  + images
+ 3D
  + movies
  + voxels


![bg](#BDC3C7)
![](#000000)

---

## Example application: 1D => 1D



![](assets/machine.translation.png)

Machine translation of one paragraph in 庄子-知北游.


![bg](#BDC3C7)
![](#000000)

----

## Example application: 1D => 2D

![](assets/text2image.png)

Image generation from text.

![bg](#BDC3C7)
![](#000000)

----


## Example application: 2D => 3D



![](assets/img2video.gif)

![bg](#BDC3C7)
![](#000000)

----

## Eample application: 3D => 3D


![](assets/video2video.gif)

So you think you can dance?


![bg](#BDC3C7)
![](#000000)

----


![](assets/faceswap.gif)

Well .... Yes!

![bg](#BDC3C7)
![](#000000)

----

##  <!--fit-->  MCNN: Multi-resolution convolutional neural networks

![width:800px](assets/MDCNN_6.jpg)

http://fengwang.github.io/mdcnn

![bg](#BDC3C7)
![](#000000)



---

## <!--fit--> Noise2Atom: Unsupervised Denoising

![width:960px](assets/n2a.png)


http://fengwang.github.io/noise2atom/

![bg](#BDC3C7)
![](#000000)

---

![width:960px](assets/noise2atom.png)

![bg](#BDC3C7)
![](#000000)

http://fengwang.github.io/noise2atom/

---

## Take-home message

- An application-neutral framework for solving inverse problems that involve image to image mappings
  - without being limited to specific applications,
  - but relying on massive datasets
- Application can be supervised/un-supervised.
- Paradigm shift: data-intensive scientific discovery.

![bg](#BDC3C7)
![](#000000)

---



## Thanks and Questions?

![bg](#BDC3C7)
![](#000000)



