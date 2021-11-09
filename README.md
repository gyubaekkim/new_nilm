## A New Non-intrusive Load Monitoring (NILM) Method for Residential Appliances
This project is to improve accuracy performance of the existing NILM systems. 
The method is composed of 7 steps across the 3 phases

### Pre-phase
- Pre-verification
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
