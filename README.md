# **ADIDoS: Automatic Detection and Identification of DDoS**

This repository provides the necessary codes to reproduce the work done in the CMPT980 project.
A pre-print will be provided the initial version is submitted to ICCST2020


## Dependencies

The code has been tested using python **3.7.4**
The requirements.txt provides the required packages.

## Folder

Our codes are provided in the **code** folder.
The reference paper we used in the evaluations can be reproduced using the codes in the **reference** folder. This code was clone and modified from the following repository:
We ran our codes using computecanada, therefore we provided the scripts to submit jobs to the large GPU nodes in the **cedar** folder.
Additionally, we provide 2 pre-trained models to test the code.

## Instructions
First create a new virtual environment
Then install the dependencies

Next, create a folder to hold the outputs

Modify the following directories in the code to point to the actual directories.

[Optional] For quick testing of the code, restrict the number of rows read from the files to a small number using:

In order to train the network use the following command

For just testing the pre-trained network, use the following command
Just replace the model with your saved model.

### Running the reference code ###
To run the reference code you need to do the following:
We only managed to run this code using a maximum of 1M rows only from each file, otherwise we get out of memory errors.




Added new files to the new_approach folder
ML_binary_classifier: this file uses the same approach as the reference code
ML_category_classifier: this file combines all test and train data and tries to classify the correct attack label

Note:
To include more features, just modify the features file.

Note: 
Test and train files are included automatically.

Note: 
labels with similar names are combined, for instance DrDoS_ is removed and UDP-lag and UDPLag are considered as one.

Required:
Fine tune the ML and select the appropriate features.

Cedar:
2 scripts are included to schedule jobs and return the outputs in the cedar_logs folder.

Model outputs and weights are saved in the same folder for future use.

The code has been tested using python 3.7.4 and computecanada 4 GPU nodes.
To run the code you need to do the following:
- Initilaize a new vitual environment with python3
- Activate the environment
- Install the requirement.txt packages
- Modify the directories 
- Dataset Directory and output directory where the model and results will be saved
- Next run the code using 
where features.txt is the file containing the features, so feel free to modify.

In order to execute the code on computecanada, you need to acquire access to the cluster and modify the email and pathes in the bash script to point to your local directories.

