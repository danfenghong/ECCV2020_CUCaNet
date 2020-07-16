# Cross-Attention in Coupled Unmixing Nets for Unsupervised Hyperspectral Super-Resolution

[Jing Yao](https://scholar.google.com/citations?user=1SHd5ygAAAAJ&hl=en), [Danfeng Hong](https://sites.google.com/view/danfeng-hong), [Jocelyn Chanussot](https://scholar.google.com/citations?user=6owK2OQAAAAJ&hl=en), [Deyu Meng](https://scholar.google.com/citations?user=an6w-64AAAAJ&hl=en), [Xiaoxiang Zhu](https://scholar.google.com/citations?user=CNakdIgAAAAJ&hl=en), and [Zongben Xu](http://en.xjtu.edu.cn/info/1017/1632.htm)

___________

Code for the paper: [Cross-Attention in Coupled Unmixing Nets for Unsupervised Hyperspectral Super-Resolution](https://arxiv.org/pdf/2007.05230.pdf).

<img src="Imgs/workflow_CUCa.png" width="666px"/>

**Fig.1.** An illustration of the proposed unsupervised hyperspectral super-resolution networks, called Coupled Unmixing Nets with Cross-Attention **(CUCaNet)**, inspired by spectral unmixing techniques, which mainly consists of two important modules: cross-attention and spatial-spectral consistency.

## Training
#TODO with demo on two HSIs (*fake and real food* and *chart and staffed toy*) of the [CAVE dataset](http://www.cs.columbia.edu/CAVE/databases/multispectral) (Using [PyTorch](https://pytorch.org/) implemented on `Windows`).

- Please see the five evaluation metrics (PSNR, SAM, ERGAS, SSIM, and UIQI) logged in `\checkpoints\CAVE_*name*\precision.txt`
 
## Testing
#TODO

## References
If you find this code helpful, please kindly cite:

[1] Yao, Jing, et al. "Cross-attention in coupled unmixing nets for unsupervised hyperspectral super-resolution." In *Proceedings of the European Conference on Computer Vision (ECCV)* (2020).

[2] Zheng, Ke, et al. "Coupled convolutional neural network with adaptive response function learning for unsupervised hyperspectral super-resolution." *IEEE Transactions on Geoscience and Remote Sensing* (2020), DOI: 10.1109/TGRS.2020.3006534.

## BibTex

```
coming soon
```

----------
For further detailed questions, please feel free to contact jasonyao92@gmail.com.
