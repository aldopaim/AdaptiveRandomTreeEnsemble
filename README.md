# Adaptive Random Tree Ensemble for Evolving Data Stream Classification
This is the repository for the Adaptive Random Tree Ensemble (ARTE) algorithm. 

Data stream mining with concept drift is a significant challenge in machine learning because this scenario requires the ability to handle unlimited and ever-changing data and real-time processing. An often employed strategy in data stream mining involves utilizing ensembles due to their capability to tackle concept drift and attain remarkably accurate predictions. However, developing a precise and efficient ensemble for data stream mining poses a significant challenge, as state-of-the-art algorithms are often highly inefficient, consuming excessive memory and processing time. In this study, we propose a novel ensemble-based classification algorithm for data streams named Adaptive Random Tree Ensemble (ARTE). The algorithm explores approaches that promote high prediction accuracy using a random-sized feature subspace for each element of the ensemble, online bagging, random choice of the cut-point for splitting the trees, and a method of classifier selection for final ensemble voting. This study also presents analyses on the contribution of the choice of subspace size and the random cut-point for splitting the tree’s nodes to the ensemble’s diversity. Following an extensive experimental investigation, ARTE exhibited high predictive performance and outperformed state-of-the-art ensembles on data streams for real and synthetic datasets while requiring fewer computational resources.

## Post-Release Updates

### Detection of Underutilized Classifiers

The classifier selection mechanism was extended to identify **underutilized classifiers**, defined as base learners whose participation in the ensemble vote falls below 10% over the last *W* instances. Once flagged, these classifiers are signaled to the training layer for selective reinitialization, allowing the ensemble to recover capacity from dormant members rather than carrying them indefinitely.

### Random Binary Splits for Nominal Attributes

ARTE now handles nominal attributes via **random binary splits**. When a candidate leaf considers a nominal attribute for splitting, the distinct observed values at that leaf are randomly partitioned into two mutually exclusive and non-empty subsets, *A₁* and *A₂*, such that each value belongs to exactly one group. Instances whose attribute value falls in *A₁* are routed to the left branch; those in *A₂* to the right. This approach guarantees valid and complete splits while introducing stochastic variability that increases tree diversity, without relying on impurity-based criteria to guide the partition.

## Datasets
The datasets and synthetic data streams are available in the `\dataset` directory.

## Using ARTE
Download the pre-compiled JAR files and run:
```
java -javaagent:sizeofag-1.0.4.jar -cp moa-custom-ARTE-2.0.jar moa.DoTask "EvaluatePrequential -l moa.classifiers.meta.ARTE -s generators.AgrawalGenerator -i 1000000 -f 10000"
```
Alternatively, import the project source code directly into [MOA](https://github.com/Waikato/moa).

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
