# cpDistiller

## Overview

cpDistiller (Triple-effect correction for Cell Painting data with contrastive and domain-adversarial learning) is specifically designed for Cell Painting data to map inputs into the low-dimensional embedding spaces that correct multiple technical effects, such as batch, row, and column effects, while capturing true biological signals.


![](./images/cpDistiller.jpg)

## Installation
The `cpDistiller` package can be installed via conda using the following steps:

### Step1

Download the package from Github and install it locally.

```bash
git clone https://github.com/Cell-Painting/cpDistiller
cd cpDistiller
```

### Step2

Create a conda environment, and then activate it as follows in terminal. All software dependencies can be found in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate cpDistiller
```

### Step3

Download the environment packages required for the pre-trained model.

```bash
cd deepcell_release
python setup.py install
pip install deepcell_toolbox==0.12.1
```
### Step4

You can install cpDistiller by the following command:

```bash
cd ..
pip install -e .
```

Our experimental environment includes two 24GB Nvidia 4090 graphics cards and 96 Intel(R) Xeon(R) Gold 783 5318N CPUs @ 2.10GHz. Completing all the preceding steps will approximately take 30 to 40 minutes, depending on the user's computer specifications and internet connection speed.

## Quick Start
We will demonstrate the capability of cpDistiller in correcting technical effects by using Cell Painting profiles from cpg0016 as an example. More details could be found in [cpDistiller documents](https://cpdistiller.readthedocs.io/).



### 1. Preparation of the example dataset
You should download raw Cell Painting images, CellProfiler-based features of Cell Painting images and metadata via AWS. More details could be found in [#preparation](https://cpdistiller.readthedocs.io/en/latest/Tutorial/index.html#preparation-of-dataset). 

```bash
aws s3 cp --no-sign-reques s3://cellpainting-gallery/cpg0016-jump/source_4/images/ /data/pub/cell/cpg0016_source4/images/--recursive --exclude "*.sqlite"
aws s3 cp --no-sign-request s3://cellpainting-gallery/cpg0016-jump/source_4/workspace/backend/ /data/pub/cell/cpg0016_source4/backend --recursive --exclude "*.sqlite"
```

### 2. Feature extraction with cpDistiller
You should process the Cell Painting images and save them in numpy (npz) format using `cpDistiller.prepare_union.tiff2npz`.  You can then use the extractor module of cpDistiller to obtain representations from Cell Painting images (cpDistiller-extractor-based features) with `cpDistiller.prepare_union.npz2embedding`. It takes approximately 11 hours to obtain representations from Cell Painting images for each plate.

You can utilize the following parameters to customize the processing pathway for extracting representations from Cell Painting images.

| Parameters   | Description                              |                                                   
| ------------ | -----------------------------------------|
| npz_path     | the path for saving npz files            |
| illumn_path  | the path for metadata                    |
| download_path| the path where the images are downloaded |
| output_path  | the path for saving representations      |

More details could be found in [#feature extraction](https://cpdistiller.readthedocs.io/en/latest/Tutorial/index.html#feature-extraction-with-cpdistiller). 

### 3. Data preprocessing
* Before data preprocessing, you should load **feature matrix** of Cell Painting images, utilizing both CellProfiler software and the extractor module of cpDistiller. 

    ```python
    data = cpDistiller.utils.merge_csv2h5ad(data_source,adata)
    ```
    | Parameters   | Description                                                                                                   |                                                   
    | ------------ | --------------------------------------------------------------------------------------------------------------|                                                                   
    | data_source  | the path for saving representations using the extractor module of cpDistiller                                 |
    | adata        | AnnData object. Rows correspond to cells (wells or perturbations) and columns to CellProfiler-based features  |

    The function returns an Anndata object that integrates cpDistiller-extractor-based features and CellProfiler-based features.

    ```python
    data = cpDistiller.utils.merge_csv2h5ad(data_source,adata)
    ```
* For data preprocessing, you could use `scanpy.pp.scale` for single batch or `cpDistiller.utils.scale_batch` for multiple batches.
    ```python
    data = scanpy.pp.scale(data)   
    ```
    ```python
    data = cpDistiller.utils.scale_batch(data)   
    ```

    The [**scanpy**](https://scanpy.readthedocs.io/en/stable/), a widely-used Python library for single-cell data analysis, offers exceptional support for preprocessing Anndata objects.
    `cpDistiller.utils.scale_batch` is used to apply z-score standardization within each batch to mitigate batch effects.


### 4. Well position effect correction and triple effect correction with cpDistiller

* You could train the cpDistiller model as following:

    ```python
    dat = DataSet(data,batch_size,mod)
    labeled(data,Mnn,Knn,technic_name_list)
    cpDistiller =  cpDistiller.main.cpDistiller_Model(dat)
    cpDistiller.train()  
    ```

    | Parameters       | Description                                                                                                      |                                                   
    | -----------------|------------------------------------------------------------------------------------------------------------------|                                                           
    | batch_size       | the batch_size for training cpDistiller                                                                          |
    | mod              | 0 for correcting well position effects and 1 for correcting triple effects, default 0                            |
    | Mnn              | the number of nearest neighbors considered when calculating MNN, default 5                                       |
    | Knn              | the number of nearest neighbors considered when calculating KNN, default 10                                      |
    | technic_name_list| the list of technic effect labels to be considered when calculating triplets, default ['row','col']              |

    It takes approximately 1 hour to correct triple effects for about 10w perturbations, depending on the user's computer specifications.

* You could obtain representations free of technical effects:

    ```python
    cpDistiller.ema.apply()
    result,category = cpDistiller.eval(dat.data)
    data.obsm['cpDistiller_embeddings']=result
    cpDistiller.ema.restore()   
    ```

    The representations will be stored in `data.obsm['cpDistiller_embeddings']`, which could be used for downstream analyses.

More training details could be found in [#well position effects](https://cpdistiller.readthedocs.io/en/latest/Tutorial/index.html#well-position-effect-correction-with-cpdistiller) and [#triple effects](https://cpdistiller.readthedocs.io/en/latest/Tutorial/index.html#triple-effect-correction-with-cpdistiller). 


## License
For the extractor module of cpDistiller, we use the pre-trained Mesmer for transfer learning. Mesmer is licensed under a Modified Apache License 2.0 for non-commercial, academic use only. See [LICENSE](https://github.com/Cell-Painting/cpDistiller/blob/main/LICENSE-MODIFIED-APACHE-2.0) for full details.

Other parts are licensed under MIT License.

## Acknowledgement

We sincerely thank the authors of the following open-source projects:

- [Mesmer](https://github.com/vanvalenlab/intro-to-deepcell, https://doi.org/10.1038/s41587-021-01094-0)