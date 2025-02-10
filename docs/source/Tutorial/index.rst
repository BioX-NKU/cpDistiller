
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


.. toctree::
    :maxdepth: 1
    :hidden:

    csv2h5ad/makeh5ad.ipynb
    batch_row_col/cpDistiller_b_r_c.ipynb
    prepare_union/tiff2npz.ipynb
    row_col/cpDistiller_r_c.ipynb

   
