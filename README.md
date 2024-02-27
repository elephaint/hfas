# HFAS <img src="assets/airlab_logo.png" width="300" alt="Airlab Amsterdam" align="right"> 

_Hierarchical Forecasting at Scale_ (HFAS) employs sparse matrix operations to enable computationally efficient hierarchical forecasting. 

### Reference ###
[Olivier Sprangers](mailto:o.r.sprangers@uva.nl), Wander Wadman, Sebastian Schelter, Maarten de Rijke. [Hierarchical Forecasting at Scale](https://arxiv.org/abs/2310.12809). Under submission.

The experiments on the public M5 dataset from our paper can be replicated by running the scripts in the [src](https://github.com/elephaint/hfas/tree/main/src/) folder. Steps to reproduce the M5 experiments from the paper:

1. Clone this repository: `git clone https://github.com/elephaint/hfas.git`
2. Open a terminal and navigate to the location where you cloned the repository.
3. Create a new Python virtual environment using [conda](https://docs.anaconda.com/free/miniconda/miniconda-install/): `conda create -n hfas python=3.9`
4. Activate the newly created environment: `conda activate hfas`
5. Install the required dependencies in the environment: `pip install -r requirements.txt`
6. Run `preprocessing.py` to preprocess the M5 dataset: `python src\exp_m5\data\preprocessing.py`. This will create a file called `m5_dataset_products.parquet` in the folder `src\exp_m5\data`.
7. Run the LightGBM experiments: `python src\exp_m5\run_experiments.py`
8. Run the traditional statistical model experiments: `python src\exp_m5\train_traditional.py`
9. Evaluate the results: `python src\exp_m5\evaluate.py`. This will create five files: `rmse_mean.csv`, `rmse_std.csv`, `mae_mean.csv`, `mae_std.csv` and `bu_error_by_7d_period.csv`. These files contain the numerical values used to construct the tables in the paper. The tables are contained in the file `src\exp_m5\exp2_allstores\lr0.05\tables.xlsx`.
10. The code for the figures in the paper is contained in `src\exp_m5\evaluate.py`.

### License ###
This project is licensed under the terms of the [Apache 2.0 license](https://github.com/elephaint/hfas/blob/main/LICENSE).

### Acknowledgements ###
This project was developed by [Airlab Amsterdam](https://www.icai.ai/labs/airlab-amsterdam).
