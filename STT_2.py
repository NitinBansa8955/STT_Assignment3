# Importing necessary libraries
import json  
import numpy as np  
import pandas as pd 

# 1.NLP dataset
# Defining label categories for the NLP_dataset annotations
LABELS = ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "ADP", "PRON", "DET", "CONJ", "PART", "PRON_WH", "PART_NEG", "NUM", "X"]

# Function for loading JSON file and returning its content
def load_annotations(file_path):
    with open(file_path, "r", encoding="utf-8") as file:  
        return json.load(file)  # Parsing and returning JSON data

# Function for building confusion matrix from the annotations of two raters
def build_confusion_matrix(rater1, rater2, labels):
    numlab = len(labels)  # Number of labels (categories)
    labtoindx = {}  # Dictionary for mapping label to index for matrix
    for idx in range(len(LABELS)):  # Assigning indices to each label
        labtoindx[LABELS[idx]] = idx

    # Initializing confusion matrix
    confusion_matrix = np.zeros((numlab, numlab), dtype=int)  

    for i, j in zip(rater1, rater2):  # Iterating over pairs of annotations from the raters
        for lab1, lab2 in zip(i["label"], j["label"]):  # For each label from both raters
            lab1_tag = lab1["labels"][0]  # Extracting label from rater 1
            lab2_tag = lab2["labels"][0]  # Extracting label from rater 2
            if lab1_tag in labtoindx and lab2_tag in labtoindx:  # Checking if the label exists in the label map
                i = labtoindx[lab1_tag]  # Getting the index of label 1
                j = labtoindx[lab2_tag]  # Getting the index of label 2
                confusion_matrix[i][j] += 1  # Updating the confusion matrix with the count

    return confusion_matrix  # Returnning the built confusion matrix

# Function to calculate Cohen's Kappa from the confusion matrix
def calculate_cohens_kappa_from_matrix(confusion_matrix):
    total = np.sum(confusion_matrix)  # Total number of annotations
    obs_agrm = np.trace(confusion_matrix) / total  # Observed agreement (diagonal elements)

    row_sums = np.sum(confusion_matrix, axis=1)  # Row sums (total annotations per label)
    col_sums = np.sum(confusion_matrix, axis=0)  # Column sums (total counts per category)
    excp_agrm = np.sum((row_sums * col_sums) / total**2)  # Expected agreement

    kappa = (obs_agrm - excp_agrm) / (1 - excp_agrm)  # Cohen's Kappa formula
    return kappa  # Returnning calculated Kappa value

# Loading the annotations from two JSON files
rater1 = load_annotations(r"C:\Users\nitin\OneDrive\Desktop\Assignment 3\Assignment 3\annotator1.json")
rater2 = load_annotations(r"C:\Users\nitin\OneDrive\Desktop\Assignment 3\Assignment 3\annotator2.json")

# Building confusion matrix using the annotations from two raters
confusion_matrix = build_confusion_matrix(rater1, rater2, LABELS)

# Displaying the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix)

# Calculating Cohen's Kappa from the confusion matrix
kappa = calculate_cohens_kappa_from_matrix(confusion_matrix)
print(f"Cohen's Kappa: {kappa}")


# 2.CV dataset
# Reading the CSV files
df1 = pd.read_csv(r"Assignment 3/CV Task - Sheet1.csv")
df2 = pd.read_csv(r"Assignment 3/CV_task_Dhruv - Sheet1.csv")
df3 = pd.read_csv(r"Assignment 3/CV_task_Nitin - Sheet1.csv")

# Merging the three dataframes on the "IMAGE NUMBER" column
merged = pd.merge(df1, df2, on="IMAGE NUMBER", suffixes=('_1', '_2'))
merged = pd.merge(merged, df3, on="IMAGE NUMBER")
# Renaming the "TRUCK/ NO TRUCK" column from the third dataframe
merged.rename(columns={'TRUCK/ NO TRUCK': 'TRUCK/ NO TRUCK_3'}, inplace=True)

# Converting categorical values ("NO TRUCK" = 0, "TRUCK" = 1) to numerical format
label_map = {'NO TRUCK': 0, 'TRUCK': 1}
merged.replace({'TRUCK/ NO TRUCK_1': label_map, 'TRUCK/ NO TRUCK_2': label_map, 'TRUCK/ NO TRUCK_3': label_map}, inplace=True)

# Initializing an empty array to count occurrences of each category per image
category_counts = np.zeros((merged.shape[0], 2), dtype=int)

# Loop through each row in the merged dataset and counting the occurrences of "NO TRUCK" and "TRUCK"
for i, row in merged.iterrows():
    counts = np.bincount(row[1:].values, minlength=2)
    category_counts[i] = counts

# Function for manually calculating Fleiss' Kappa
def calculate_fleiss_kappa(matrix):
    N, k = matrix.shape  # N = number of items(20), k = number of categories(0,1)

    # Calculating proportion of raters that agreed for each item
    P_i = np.sum(matrix**2, axis=1) - matrix.sum(axis=1)
    P_i = P_i / (matrix.sum(axis=1) * (matrix.sum(axis=1) - 1))

    # Calculating mean agreement (P-bar)
    P_O = np.mean(P_i)

    # Calculating expected agreement (P-e)
    category_proportions = np.sum(matrix, axis=0) / np.sum(matrix)
    P_e = np.sum(category_proportions**2)

    # Calculating Fleiss' Kappa
    kappa = (P_O - P_e) / (1 - P_e)
    return kappa  

# Computing Fleiss' Kappa for the category counts matrix
fleiss_kappa_score = calculate_fleiss_kappa(category_counts)

# Function to determining agreement level based on the sum of values for each row
def agreement_level(row):
    total_sum = row.sum()  # Calculating the total sum of "TRUCK" and "NO TRUCK" annotations
    if total_sum == 0 or total_sum == 3:  # If the sum is 0 or 3, it's full agreement
        return "Full Agreement"
    elif total_sum == 1 or total_sum == 2:  # If the sum is 1 or 2, it's partial disagreement
        return "Partial Disagreement"

# Applying the agreement_level function to each row of the merged DataFrame and add the result as a new column
merged['Agreement Level'] = merged[['TRUCK/ NO TRUCK_1', 'TRUCK/ NO TRUCK_2', 'TRUCK/ NO TRUCK_3']].apply(agreement_level, axis=1)

# Printing the merged DataFrame, including the Agreement Level
print(merged[['TRUCK/ NO TRUCK_1', 'TRUCK/ NO TRUCK_2', 'TRUCK/ NO TRUCK_3', 'Agreement Level']])

# Optionally save the updated DataFrame with the "Agreement Level" column to a new CSV file
merged.to_csv(r"C:\Users\nitin\OneDrive\Desktop\Assignment 3\CV Task - Updated.csv", index=False)

# Printing the calculated Fleiss' Kappa score
print(f"Fleiss' Kappa: {fleiss_kappa_score}")

# interpetation for Fleiss' Kappa

# Fleiss' Kappa (CV dataset - 0.682):
# Fleiss' Kappa is used to assess the agreement between multiple raters (in our case, three raters labeling the images ,two from our team and 1 from another team) on categorical items. 

# Kappa scale for interpretation (based on common conventions):
# < 0: No agreement
# 0.01 - 0.20: Slight agreement
# 0.21 - 0.40: Fair agreement
# 0.41 - 0.60: Moderate agreement
# 0.61 - 0.80: Substantial agreement
# 0.81 - 1.00: Almost perfect agreement

# Our value of Fleiss' Kappa (0.682) indicates Substantial Agreement between the raters. This value suggests that the raters are in strong agreement about the classifications, such as "TRUCK" vs. "NO TRUCK." While the agreement is not perfect, it is much better than random chance, which indicates that the raters are fairly consistent in their decisions.

# However, the value still leaves room for slight improvements in consistency. Substantial Agreement typically reflects situations where raters are performing well, but occasional discrepancies can occur due to subjective factors or the complexity of the task at hand.

# interpetation for Cohen's Kappa
# Cohen's Kappa (NLP dataset - 0.714):
# Cohen's Kappa is a similar metric but is specifically for evaluating the agreement between two raters. It has the same range and interpretation scale as Fleiss' Kappa.

# Our value of Cohen's Kappa(0.714) suggests substantial agreement between the two raters in the NLP dataset. The raters are in strong agreement, but there is still room for some disagreement. A value of 0.71 is quite good, indicating that the two raters are consistent in their classification of the items.

# It suggests that the two raters generally agree on the labels, though some differences might still be present. This level of agreement is usually considered high and indicates that the labeling process is reliable, but improvements could still be made to achieve a more perfect match.

