# Underwater images Enhancement Model

This repository contains my personal implementation of the third stage of my Master’s thesis on the topic of **Combination of Deep Learning Techniques and
Classical Methods for Underwater Images Enhancement**, proposed and supervised by Professor: **José Luis Lisani Roca**.

It includes three **UNet-based architectures** for underwater image enhancement, each implemented as a Jupyter Notebook. These notebooks are designed to be easy to follow and work in most environments and operating systems.

In recent years, deep learning-based techniques have been widely explored for underwater image enhancement. However, their effectiveness is often limited due to the lack of sufficient training data and conventional architectural design used in most research.

In contrast, classical image enhancement methods frequently outperform neural networks-based models for specific underwater image types, While each method excels under certain
aquatic conditions, no single classical approach works universally well across all cases.

This work aims to leverage the strengths of both classical and deep learning approaches by exploring a hybrid method for underwater image enhancement that is both effective and data-efficient.

the three stages of this work are as follow:

**Stage 1:** an analysis of the Underwater Image Enhancement Benchmark (UIEB) dataset through reverse engineering, this first stage aim to identify which classical method used to generate each image reference provided in the UIEB dataset. To do this for each raw image, we compare the reference images in the UIEB dataset with outputs from various classical enhancement techniques reported in the UIEB paper. This helps infer which methods were likely used or best approximate the references, using both similarity metrics (SSIM, MSE, L1, and PSNR) and visual inspection.

**stage 2:** after identifying the classical enhancement method used to generate each reference image in the UIEB dataset, we train a classification network to predict the most suitable classical algorithm for any given raw underwater image, due to the subjective nature of UIEB dataset annotations, the classifier may predict outputs with better visual quality than the references. While this work focuses exclusively on the UIEB dataset and limited to four classes: Dive+, Fusion, Histogram Prior, and Two-Step method, the classifier can be applied to expand the datasets by automatically selecting optimal enhancement methods for images outside UIEB dataset, This strategy can contributes to solve the primary challenge of limited training data in deep learning based models for underwater image enhancement.

**stage 3:** the classifier's predicted outputs serve as reference images for the three U-Net models implemented in this repository, these U-NETs are trained to enhance raw underwater images, The goal is to develop deep learning models capable of producing superior underwater image quality.
the classifier used in this task achieved 60% accuracy, even when misclassifications occur, the resulting outputs are not necessarily of lower quality. In some cases, the classifier's predicted method produces visually superior results compared to the original UIEB reference, this assessment remains subjective opinion. The high inter class visual similarity among classical enhancement methods makes some misclassifications perceptually insignificant, and the impact on the three models is relative for several reasons that will be discussed in each notebook


The U-Net architecture has proven especially effective for this task thanks to its symmetric encoder-decoder structure and skip connections, which preserve spatial information while facilitating deep contextual learning. 
This unique architecture enables the reconstruction of fine-grained details while maintaining a large receptive field, making U-Net ideally suited for underwater image enhancement.

The UIEB dataset serves as a valuable benchmark for these models, containing original low quality underwater images alongside their enhanced versions produced
by classical algorithms. This dataset was generated through a consensus of both expert and non-expert evaluators, who performed pairwise comparisons to select the best enhancement results.

These implementations are partially based on assignments I completed for a deep learning subject provided by Professor: **Miguel Ángel Calafat Torrens**. 
All of them are also available in my [Deep Learning repository](#), which contains a collection of powerful implementations and carefully designed challenges.
This repository serves as a strong foundation for anyone looking to build expertise and excel in the field of deep learning.

---
## Conventional Boundaries

Despite notable progress in deep learning for underwater image enhancement, researchers still encounter several challenges that constrain important progress. A key limitation is the lack of comprehensive datasets;  most publicly available ones, such as UIEB or EUVP, contain a relatively small number of images and often lack diversity in underwater conditions. 

Additionally, the field remains stuck in a repetitive loop of approaches: researchers frequently adopt similar CNN or GAN-based architectures with minor modifications, repackaged as novel contributions. This dogmatic adherence to established frameworks limits innovation.

Most works still rely on conventional preprocessing steps like resizing, cropping, or padding images to fixed dimensions, dictated largely by the constraints of frameworks like PyTorch, which require uniform input sizes within a batch. These technical limitations have created a methodological box, where almost all researchers operate within the same rigid boundaries, preventing the exploration of more flexible or adaptive models that could better handle the natural variability in underwater imagery.

The majority of deep learning-based approaches published in the field of underwater image enhancement rely on resizing, cropping, or padding input images to fixed dimensions prior to processing. However, resizing introduces irreversible detail loss, distortion, and interpolation artifacts, which are especially detrimental in underwater imagery. This convention stems from architectural constraints of frameworks like PyTorch and TensorFlow that do not natively support batch processing of images with varying resolutions.

This workaround highlights a broader constraint in standard deep learning ecosystems: variable-resolution batching is not supported by most mainstream frameworks, including PyTorch. Although PyTorch is the most widely adopted framework, it imposes a significant restriction by disallowing resolution diversity within batch-level training

## Custom Approach

To overcome the limitations of fixed-size image preprocessing, our architecture introduces a dynamic downsampling strategy that processes images at their native resolution, preserving spatial fidelity and overcoming the constraints of fixed-size preprocessing.

Since images vary in size and aspect ratio, the challenge is how to process all of them while preserving their native resolution and avoiding detail loss, distortion, or interpolation artifacts.

Each image has its own unique resolution (height and width), and The goal of this dynamic downsampling strategy is to process images at their native resolution without resizing, cropping, 
or padding prior to feeding them into the U-Net. A U-Net consists of successive encoder-decoder blocks that downsample and then upsample the image by factors of 2. 

For instance, a compact 234×276 image might be processed through four encoder-decoder blocks, whereas a larger 1007×1347 image could pass through six. In the case of non-square images like 289×1123, the model can adapt by applying downsampling four blocks along its width and height dimension, but continue downsampling to six blocks along only its height dimension effectively supporting symetric and asymetric downsampling for different aspect ratio.

This results in significantly better preservation of fine details compared to fixed-resolution pipelines. 
This allows the model to preserve fine-grained details of the input image, with the loss function comparing outputs directly to their corresponding native-resolution references


While dynamic resolution handling introduces challenges primarily because PyTorch and most deep learning frameworks do not natively support batch processing of images with varying resolutions, this limitation is addressed by setting the batch size to 1 with parametrable gradient accumulation.


This is a custom approach that challenges the conventions of fixed-size image preprocessing by adapting U-Net depth dynamically to each image’s native resolution. 
While it works within the constraints of existing deep learning frameworks, it avoid their batch-processing limitations and goes beyond standard preprocessing pipelines used in most research.

---
