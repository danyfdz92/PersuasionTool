# PersuasionTool
Persuasion Checker tool developed by BITSLAB under the supervision of Professor Lu Xiao

This persuasion tool was developed at bitslab (Syracuse University) under the supervision of Professor Lu Xiao.

TFIDF.py -> this python script calculates the TFIDF (term frequency–inverse document frequency) for each comment. It is necessary that the csv file contains the root comment and the original post in order to make the comparison. 

OtherFeatures.py -> it is a python script that calculates different features that are necessary to run the model.

DecisionModels.py -> after obtaining all the features for the comments, this script splits the dataset into training and testing to create different decision models.

persuasiveTool.py -> user interface to test the persuasiveness of different comments.
