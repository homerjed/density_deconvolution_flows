<h1 align='center'>Density Deconvolution with Normalising Flows</h1>

#### Usage 

```
pip install -e . 
```

<!-- #### Samples

I haven't optimised anything here (the authors mention varying the variance of noise used to dequantise the images), nor have I trained for very long. You can see slight artifacts due to the dequantisation noise.

<p align="center">
  <picture>
    <img src="assets/mnist_warp.gif" alt="Your image description">
  </picture>
</p>

<p align="center">
  <picture>
    <img src="assets/cifar10_warp.gif" alt="Your image description">
  </picture>
</p> -->

#### Citations 

```bibtex
@misc{dockhorn2020densitydeconvolutionnormalizingflows,
      title={Density Deconvolution with Normalizing Flows}, 
      author={Tim Dockhorn and James A. Ritchie and Yaoliang Yu and Iain Murray},
      year={2020},
      eprint={2006.09396},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2006.09396}, 
}
```

```bibtex
@misc{zhai2024normalizingflowscapablegenerative,
      title={Normalizing Flows are Capable Generative Models}, 
      author={Shuangfei Zhai and Ruixiang Zhang and Preetum Nakkiran and David Berthelot and Jiatao Gu and Huangjie Zheng and Tianrong Chen and Miguel Angel Bautista and Navdeep Jaitly and Josh Susskind},
      year={2024},
      eprint={2412.06329},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.06329}, 
}
```