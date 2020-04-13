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

