## A New Non-intrusive Load Monitoring (NILM) Method for Residential Appliances
This project is to improve accuracy performance of the existing NILM systems. 
The method is composed of 7 steps across the 3 phases

### Pre-phase
- [Pre-verification](### 1. Pre-verification)
- Algorithm Selection

### Build phase
- Preprocessing
- Feature Extraction
- Training/Evaluation

### Use phase
- Prediction (Disaggregation)
- Result Post-processing

## [Prerequisite] Installation of NIMLTK Environement

### python version 3.7 is required



### NILMTK Setting 
[REF] https://github.com/nilmtk/nilmtk
-	conda create -n nilm -c conda-forge -c nilmtk nilmtk-contrib
-	conda activate nilm


#### * Tasks to be executed in created environment (nilm)
-	(nilm) python -m ipykernel install --user –name nilm
-	(nilm) pip install h5py==2.10.0
-	(nilm) pip install tensorflow
-	(nilm) conda install graphviz
-	(nilm) pip install graphviz
-	(nilm) pip install pydot
-	(nilm) pip install tslearn

#### Trouble Shooting
-	from keras.utils import plot_model --> from keras.utils.vis_utils import plot_model

## [Notebook] The most important steps for new NILM method

### 1. Pre-verification
We examine the pre-verification of candidate appliances in which appliances that cannot be recognized by NILM are excluded.
Pre-verification is performed prior to NILM model development. NILM models can only work when the training data they receive is produced by appliances with consistent electrical consumption patterns. This was confirmed using the dynamic time warping (DTW) technique, which is better suited than the Euclidian distance method even if the lengths or starting points of the comparison targets are different

### 2. Algorithm Selection
Various algorithms have been utilized by NILM systems. The benchmark project used adopted neural network-based algorithms. The neural network-based algorithms performed better than existing combinational optimization and hidden Markov models. Thus, we compared the performance of the models produced by the proposed method with the neural network-based algorithms. We used random forest and gradient boosting machine (GBM), the representative decision tree-based ensemble models, in the proposed method.

### 3. Experiments
NILM performance was evaluated in terms of precision, recall, accuracy, and F1 score, which are commonly used performance indicators for classification models, and whether the results satisfied the target level (all metrics are over 70%).

### 4. Result Post-processing
NILM model results cannot be used directly, so they must be post-processed. The find_peaks function in the Python scipy package was used to identify when the appliance was activated. The proper input parameters were set to detect peaks that indicated appliance operation.
