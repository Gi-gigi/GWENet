# GWENet: Global-Guided Weighted Enhancement for Salient Object Detection
**Official for "Global-Guided Weighted Enhancement for Salient Object Detection"**

## Abstract
Salient Object Detection (SOD) benefits from the guidance of global context to further enhance performance. However, most works focus on using the top-layer features through simple compression and nonlinear processing as the global context, which inevitably lacks the integrity of the object. Moreover, directly integrating multi-level features with global context is ineffective for solving semantic dilution. Although the global context is considered to enhance the relationship among salient regions to reduce feature redundancy, equating high-level features with global context often results in suboptimal performance. To address these issues, we redefine the role of global context within the network and propose a new method called Global-Guided Weighted Enhancement Network (GWENet). We first design a Deep Semantic Feature Extractor (DSFE) to enlarge the receptive field of network, laying the foundation for global context extraction. Secondly, we construct a Global Perception Module (GPM) for global context modeling through pixel-level correspondence, which employs global sliding weighted technology to provide the network with rich semantics and acts on each layer to enhance SOD performance by  Global Guidance Flows (GGFs). Lastly, to effectively merge multi-level features with the global context, we introduce a Comprehensive Feature Enhancement Module (CFEM) that integrates all features within the module through 3D convolution, producing more robust feature maps. Extensive experiments on five challenging benchmark datasets demonstrate that GWENet achieves state-of-the-art results.

## Visualization Results
![Image text](https://github.com/Gi-gigi/GWENet/blob/main/Visual/Figure3.jpg)
****
![Image text](https://github.com/Gi-gigi/GWENet/blob/main/Visual/Figure7Git.jpg)
