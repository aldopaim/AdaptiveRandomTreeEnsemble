# AdaptiveRandomTreeEnsemble
This is the repository for the Adaptive Random Tree Ensemble (ARTE) algorithm. 

Data stream mining with concept drift is a significant challenge in machine learning because this scenario requires the ability to handle unlimited and ever-changing data and real-time processing. An often employed strategy in data stream mining involves utilizing ensembles due to their capability to tackle concept drift and attain remarkably accurate predictions. However, developing a precise and efficient ensemble for data stream mining poses a significant challenge, as state-of-the-art algorithms are often highly inefficient, consuming excessive memory and processing time. In this study, we propose a novel ensemble-based classification algorithm for data streams named Adaptive Random Tree Ensemble (ARTE). The algorithm explores approaches that promote high prediction accuracy using a random-sized feature subspace for each element of the ensemble, online bagging, random choice of the cut-point for splitting the trees, and a method of classifier selection for final ensemble voting. This study also presents analyses on the contribution of the choice of subspace size and the random cut-point for splitting the tree’s nodes to the ensemble’s diversity. Following an extensive experimental investigation, ARTE exhibited high predictive performance and outperformed state-of-the-art ensembles on data streams for real and synthetic datasets while requiring fewer computational resources.

## Datasets
The datasets and synthetic data streams are available in the `\dataset` directory.

## Using ARTE
Download the pre-compiled jar files or import the project source code into [MOA](https://github.com/Waikato/moa). 

## Citation
```
@article{paim2025adaptive,
  title={Adaptive random tree ensemble for evolving data stream classification},
  author={Paim, Aldo M and Enembreck, Fabr{\'\i}cio},
  journal={Knowledge-Based Systems},
  volume={309},
  pages={112830},
  year={2025},
  publisher={Elsevier}
}
```
