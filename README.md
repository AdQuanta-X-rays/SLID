this is just code collection that we used in this project without the databases, networks weight etc. 
to generate a database one needs to run  G4_Nanophotonic_Scintillator/utils/generate_database.py
(this will work assuming you have installed the geant4 simulation tool)
to train models on this database, run neural_network_interface.py in the same folder
once the models are trained, you can use them in END2END_optim_v02.py to create a full prediction for the photonic emission 
this can be evaluated useing optimization_evaluation.py (for the complete process) and model_performance_estimation.py (for model estimation) 
