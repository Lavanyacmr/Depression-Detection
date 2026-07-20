TDD-Net: Transformer-Based Depression Detection Network
Overview
TDD-Net is a transformer-based multimodal deep learning framework for depression risk detection using textual and metadata features. The proposed architecture integrates contextual language representations with participant metadata to improve binary classification performance for depression-risk screening. This repository contains the complete implementation of TDD-Net, including data pre-processing, model training, evaluation, and utility scripts.
Repository Structure
•	main.py  Main execution script
•	train.py – Model training
•	evaluate.py – Model evaluation
•	model.py – TDD-Net architecture
•	preprocessing.py – Data pre-processing pipeline
•	utils.py – Utility functions
•	config.py – Model configuration
•	requirements.txt – Python dependencies
Features
•	Transformer-based text encoder
•	Metadata feature fusion
•	Binary depression-risk classification
•	End-to-end pre-processing pipeline
•	Training and evaluation framework
•	Configurable hyper parameters
•	Modular implementation
Requirements
•	Python 3.10 or later
•	PyTorch
•	Transformers
•	NumPy
•	Pandas
•	Scikit-learn
•	Matplotlib
•	tqdm
Install the required packages using:
pip install -r requirements.txt
Training
Run the following command to train the model:
python train.py
Evaluation
Run the following command to evaluate the trained model:
python evaluate.py
Configuration
Model architecture and training parameters can be modified in the config.py file.
Dataset Availability
The dataset used in this study contains sensitive participant-generated mental health and social media information. To protect participant privacy and confidentiality, the dataset is not included in this repository and is not publicly distributed. This repository provides the complete implementation of the proposed framework, including preprocessing, model architecture, training scripts, evaluation scripts, and configuration files. Researchers may reproduce the proposed methodology using their own ethically collected dataset with a comparable structure.
Ethics Statement
Participants voluntarily provided informed consent before contributing data to this study. All collected data were anonymized prior to analysis, and personally identifiable information was removed to protect participant privacy. Formal institutional ethics approval was not obtained prior to data collection, and this is acknowledged as a limitation of the study. The proposed model is intended solely for research on depression-risk screening and should not be interpreted as a clinical diagnostic tool.
Reproducibility
This repository includes:
•	Complete source code
•	Model implementation
•	Data pre-processing pipeline
•	Training scripts
•	Evaluation scripts
•	Configuration files
The original participant dataset is intentionally excluded because it contains sensitive human-subject information.
Citation
If you use this repository in your research, please cite the associated publication.
License
This repository is released for academic and research purposes only. Users are responsible for ensuring compliance with applicable ethical guidelines, institutional policies, and local regulations when applying this code to human-subject data.
