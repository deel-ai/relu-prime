# Which derivative for ReLU at 0? The role of numerical precision.

This repository contains all the code and results for the NeurIPS 2021 submission _**Which derivative for ReLU at 0? The role of numerical precision**_.


All the data generated from the experiments are located in [paper_results](paper_results).  
All the figures from the paper are generated with this [notebook](paper_results/mkPlots.ipynb), this [notebook](introduction_experiment/expeSurprise.ipynb) and this [script](paper_results/section_3/mkPlot.R).

Code for all the experiments: 

* Introduction experiment: 
    * [code](introduction_experiment/expeSurprise.ipynb) 
* Section 3 experiments: 
    * [notebook1](notebooks/MNIST_volume_estimation.ipynb)
    * [notebook2](notebooks/volume_estimation_by_architecture.ipynb)
    * [results](paper_results/section_3)

* Section 4.3 experiments: 
    - To run the SGD experiment: 
    ```console
    pyton train.py --network vgg11 --optimizer sgd --epochs 200 --batch_norm [BATCH_NORM]
    ```
    with ```[BATCH_NORM]``` = True or False
    - To run the Adam experiment: 
    ```console
    pyton train.py --network vgg11 --optimizer adam --epochs 200 --batch_norm [BATCH_NORM]
    ```
    with ```[BATCH_NORM]``` = True or False
    * results: 
        - [SGD](paper_results/section_4/cifar10/vgg11/vgg11_cifar10_sgd.csv)
        - [Adam](paper_results/section_4/cifar10/vgg11/vgg11_cifar10_adam.csv)

    * Section 4.4 experiments: 
        * Notebooks:
            - [vgg11](notebooks/CIFAR10_VGG11_volume_estimation.ipynb)
            - [resnet18](notebooks/CIFAR10_VGG11_volume_estimation.ipynb)
        * results:
            - [vgg11(cifar10)](paper_results/section_4/cifar10/vgg11/volume_estimation_sample_vgg11.csv)
            - [vgg11(svhn)](paper_results/section_4/svhn/volume_estimation_sample_vgg11_svnh.csv)
            - [resnet18](paper_results/section_4/cifar10/resnet18/volume_estimation_sample_resnet18.csv)

* Additional experiments:  
    To run the additional experiments:
    ```console
    python train.py --network [NETWORK] --dataset[DATASET] --batch_norm [BATCH_NORM] --epochs 200
    ```
    with ```[NETWORK]``` = vgg11 or resnet18 , ```[DATASET]``` = cifar10 or svhn and ```[BATCH_NORM]``` = True or False

The code used to generate the figures is available [here](paper_results/mkPlots.ipynb)
        


