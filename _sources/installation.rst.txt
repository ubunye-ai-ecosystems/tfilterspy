Installation
============

From PyPI
---------

.. code-block:: bash

   pip install tfilterspy

From Source
-----------

.. code-block:: bash

   git clone https://github.com/ubunye-ai-ecosystems/tfilterspy.git
   cd tfilterspy
   pip install -e .[dev]

Requirements
------------

- **Python** >= 3.8
- **NumPy** -- array operations and linear algebra
- **SciPy** -- efficient matrix solves (``scipy.linalg.solve``)
- **Dask** -- parallel ensemble/particle propagation (installed automatically)

Optional:

- **matplotlib** -- plotting in examples and notebooks
- **scikit-learn** -- used in example notebooks for classification benchmarks
