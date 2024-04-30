# CS348K Final Project Proposal (Kyle Sargent, Eric Ryan Chan)

## Proposal Document 

* __Project Title:__  Alternative Architectures for Diffusion Image Transformers
  
* __Names and SuNET ID's__ ksarge, erchan

* __Summary:__ For years, UNets were the predominant network architecture for denoising diffusion models. Recently, transformer-focused network designs, including UViT and DiT, have been demonstrated to outperform UNets for image and video generation. While transformers are a powerful architecture, their computational complexity (O(N2) in sequence length) make them difficult to scale without vast computational resources. We would like to investigate whether sub-quadratic transformer alternatives, such as Mamba, are a viable alternative to standard attention for image-generation diffusion models. To our knowledge, sub-quadratic attention has not been successfully used in diffusion models before, and a thorough evaluation, whether it produces a positive or a negative result, will be useful for the generative modeling community.

* __Inputs and outputs:__ The input to the task is a training dataset, used to train our generative model, and the output of the task is a trained generative model, which can be used to synthesize images. The central constraints are 1. model performance, measured in Frechet Inception Distance (FID, lower is better), and compute required to train the model, measured in GPU hours. Our goal is to design a diffusion model architecture that produces good images (achieving a low FID score) while requiring minimal resources to train.

* __Task list:__
    * Download the Diffusion Transformers github, get an allocation for compute resources on the SVL lab cluster.
    * Download and preprocess the FFHQ dataset.
    * Implement an evaluation protocol. We plan to use training dataset Frechet Inception Distance as our primary metric, with Kernel Inception Distance, validation loss, and validation dataset Frechet Inception Distance as useful secondary measures.
    * Set a baseline using a transformer-based architecture on the FFHQ dataset.
    * Implement an alternative architecture using Mamba.
    * Train, evaluate, and analyze the results of the Mamba-based architecture.
  
* __Expected deliverables and/or evaluation.__ Our intended goal is to train a DiT variant with a Mamba-based architecture to generate images on the FFHq dataset, and compare the performance using the metrics above with the reference DiT implementation. In addition to visual quality, which can be benchmarked via the generative metrics like FID, KID, etc., we intend to perform analysis of the inference throughput of the Mamba-based variant with DiT. 

* __What are the biggest risks?__ Beyond implementation and training, the biggest risk of the project is that we implement novel diffusion architectures and, despite our best efforts, they underperform current SOTA models based on standard transformers. Although less satisfying, a negative result, i.e. a conclusion that sub-quadratic attention cannot outperform traditional attention for image synthesis, would still be an important discovery that would inform future research.

* __What you need help with?__. Although we believe we are comfortable executing the project plan as written, it would be helpful to get some initial feedback about whether the project is targeting an appropriate problem and scoped appropriately. Additionally, if there are any further systems-type analyses that are desirable, please let us know and we can attempt them on our Mamba-based generative model of images.
