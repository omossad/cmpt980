
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

