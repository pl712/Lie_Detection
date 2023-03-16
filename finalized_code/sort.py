import os
import shutil
import pandas as pd

#MU3D  
# # Set the source directory where your files are located
# src_dir = "/Users/peter/Desktop/Lie_Detection/processed/"

# # Set the destination directories for the two groups of files
# group_1_dir = "/Users/peter/Desktop/Lie_Detection/finalized_code/data/mu3d/truth/"
# group_2_dir = "/Users/peter/Desktop/Lie_Detection/finalized_code/data/mu3d/lie/"

# # Loop through each file in the source directory
# for filename in os.listdir(src_dir):
#     # Check if the file is a CSV file and has the correct naming convention
#     if filename.endswith(".csv") and "_" in filename:
#         # Split the filename into its components
#         name_parts = filename.split("_")
#         # Determine which group the file belongs to based on the last two characters of the second component
#         if name_parts[1][-6:-4] in ["PT", "NL"]:
#             shutil.move(os.path.join(src_dir, filename), os.path.join(group_1_dir, filename))
#         elif name_parts[1][-6:-4] in ["NT", "PL"]:
#             shutil.move(os.path.join(src_dir, filename), os.path.join(group_2_dir, filename))


#BOL
# csv = pd.read_csv("./mapping.csv")

# src_dir = "/Users/peter/Desktop/processed/"

# # # Set the destination directories for the two groups of files
# group_1_dir = "/Users/peter/Desktop/Lie_Detection/finalized_code/data/BOL/truth/"
# group_2_dir = "/Users/peter/Desktop/Lie_Detection/finalized_code/data/BOL/lie/"

# result = csv['truth']
# name = csv['video']

# for i in range(len(name)):
#     if name[i][18].isdigit():
#         speaker = name[i][17:19]
#         count = name[i][24:25]
#         truth = result[i]
#     else:
#         speaker = name[i][17]
#         count = name[i][23:24]
#         truth = result[i]

#     if truth == 0:
#         filename = f"User_{speaker}_run_{count}.csv"
#         shutil.move(os.path.join(src_dir, filename), os.path.join(group_2_dir, filename))
#     elif truth == 1:
#         filename = f"User_{speaker}_run_{count}.csv"
#         shutil.move(os.path.join(src_dir, filename), os.path.join(group_1_dir, filename))

