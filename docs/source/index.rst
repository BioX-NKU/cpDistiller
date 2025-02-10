.. docs documentation master file, created by
   sphinx-quickstart on Thu Apr 25 13:51:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Triple-effect correction for Cell Painting data with contrastive and domain-adversarial learning.
=======================================================================================================================

Cell Painting, as a high-throughput imaging technology, generates extensive cell-stained imaging data, providing unique morphological insights for biological research. However, Cell Painting data typically contains three distinct types of technical effects, referred to as triple effects, including batch effects, as well as gradient-influenced row and column effects. The interaction of various technical effects can obscure true biological signals and complicate the characterization of Cell Painting data, making correction essential for reliable analysis. Here, we propose cpDistiller, a triple-effect correction method specially designed for Cell Painting data, which leverages a pre-trained segmentation model coupled with a semi-supervised Gaussian mixture variational autoencoder utilizing contrastive and domain-adversarial learning. Through extensive qualitative and quantitative experiments across various Cell Painting profiles, we demonstrate that cpDistiller effectively corrects well position effects (both row and column effects), a challenge that no current methods address, while preserving cellular heterogeneity. Moreover, in addition to its capabilities for comprehensive triple-effect correction and incremental learning, cpDistiller reliably infers gene functions and relationships combined with scRNA-seq data and excels at identifying gene and compound targets, which is a critical step in drug discovery and broader biological research.

.. toctree::
   :maxdepth: 4
   :hidden:

   API/index
   Tutorial/index
   Release/index
  




