
Tutorial
========

Preparation of dataset
-------------------------

Download the features extracted with the CellProfiler software from Cell Painting images and convert them into h5ad format.

* `Preparation of an example dataset <csv2h5ad/makeh5ad.ipynb>`_

Feature extraction with cpDistiller
-------------------------------------

cpDistiller could extract high-level deep learning features from Cell Painting images using ``cpDistiller.prepare_union.tiff2npz`` and ``cpDistiller.prepare_union.npz2embedding``

* `Feature extraction using cpDistiller <prepare_union/tiff2npz.ipynb>`_

Well position effect correction with cpDistiller
-------------------------------------------------

cpDistiller could correct well position effects (both row and column effects) using ``cpDistiller.main.cpDistiller_Model`` by setting ``mod`` to 0.

* `Well position effect correction using cpDistiller <row_col/cpDistiller_r_c.ipynb>`_

Triple effect correction with cpDistiller
-----------------------------------------

cpDistiller could correct triple effects (including batch, row, and column effects) using ``cpDistiller.main.cpDistiller_Model`` by setting ``mod`` to 1.

* `Triple effect correction using cpDistiller <batch_row_col/cpDistiller_b_r_c.ipynb>`_

Detailed results of Fig. 5
-----------------------------------------

To further demonstrate cpDistiller's ability to uncover gene relationships, we performed biological analyses on its gene embeddings for Group_A, Group_B, and Group_C.

The detailed results of the gene sets analyzed using GeneMANIA are available at (same as Fig. 5): https://genemania.org/search/homo-sapiens/CD3D/NCR3/FGFBP2/PRF1/FASLG/ELMO1/JAML/INPP5D/PRKCB/SPN/VAV1/PPP1R16B/KLHL6/TAGAP/, https://genemania.org/search/homo-sapiens/MRPS35/GLRX3/PSMA3/UCHL3/MRPL1/COA6/MRPL47/AIMP1/TPRKB/CD3D/NCR3/FGFBP2/PRF1/FASLG/ and https://genemania.org/search/homo-sapiens/MRPS35/GLRX3/PSMA3/UCHL3/MRPL1/COA6/MRPL47/AIMP1/TPRKB/ELMO1/JAML/INPP5D/PRKCB/SPN/VAV1/PPP1R16B/KLHL6/TAGAP/



.. toctree::
    :maxdepth: 1
    :hidden:

    csv2h5ad/makeh5ad.ipynb
    batch_row_col/cpDistiller_b_r_c.ipynb
    prepare_union/tiff2npz.ipynb
    row_col/cpDistiller_r_c.ipynb

   
