## An Experimental Study on Performance of Non-intrusive Load Monitoring
This project is to review on performance of the existing NILM systems. 
The study is composed of several experiments

### Experiments
- [on Algorithms](#action1)
- [on Sampling Period](#action2)
- [on Feature Extraction](#action3)
- [on Datasets](#action4)
- [on Model Architecture](#action5)
- [on Post-processing](#action6)

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

## [Notebooks] The most important steps for new NILM method

### <a name="action1" /> 1. Algorithms
The following seven machine learning algorithms are evaluated for NILM performance comparison.
- Recurrent Neural Net (RNN)
- Gated Recurrent Unit (GRU)
- Window GRU (WGRU)
- Denosing Autoencoder (DAE)
- Sequence to Point (S2P)
- Random Forest (RF)
- Gradient Boost Machine (GBM)

### <a name="action2" /> 2. Sampling Period
To examine the effect of the sampling period of data on NILM performance, experiments are conducted with different sample period data.

### <a name="action3" /> 3. Feature Extraction
NILM performance is evaluated when models are trained with extracted features. In the previous experiment, only one input feature was trained.

### <a name="action4" /> 4. Datasets
It should be verified whether the performance depends on datasets or not. For this purpose, it is necessary to check whether the NILM model developed through the previous process with Enertalk dataset shows similar performance results in other dataset, UK-DALE. 

### <a name="action5" /> 5. Model Architecture
NILM as multi-label classification problem can be generally implemented through either binary relevance or chain classifier methods.

### <a name="action6" /> 6. Result Post-processing
NILM model results cannot be used directly, so they must be post-processed. The find_peaks function in the Python scipy package was used to identify when the appliance was activated. The proper input parameters were set to detect peaks that indicated appliance operation.
