# Python file to parse the csv file with the x,y coordinates
# of the landmarks from Zooniverse and create a simplified
# csv file with just the coordinates

import pandas as pd

# could either store each image as an instance of a class, as shown here:
# to have methods to do things later on if needed


class oakImage:

    def __init__(self, id, blade_tip, sinus_major, lobe_tip_margin, petiole_tip,
                 petiole_blade, major_secondary, minor_secondary, max_width,
                 min_width, next_width):
        self.id = id  # int
        self.blade_tip = blade_tip  # tuple
        self.sinus_major = sinus_major  # dictionary
        self.lobe_tip_margin = lobe_tip_margin  # dictionary
        self.petiole_tip = petiole_tip  # tuple
        self.petiole_blade = petiole_blade  # tuple
        self.major_secondary = major_secondary  # dictionary
        self.minor_seconadary = minor_secondary  # dictionary
        self.max_width = max_width  # 4-tuple
        self.min_width = min_width  # 4-tuple
        self.next_width = next_width  # 4-tuple


# OR
# could store each instance of a class as a dictionary, with lists of tuples for each coordinate
l_counter = {"blade_tip": [(0, 0)], "sinus_major": [(0, 0)],
             "lobe_tip_margin": [(0, 0)],
             "petiole_tip": [(0, 0)], "petiole_blade": [(0, 0)],
             "major_secondary": [(0, 0)],
             "minor_seconadary": [(0, 0)],
             "max_width": [(0, 0)], "min_width": [(0, 0)], "next_width": [(0, 0)]}

# List of landmarks names, matching exactly what they are in csv file
# convert csv file to pandas dataframe for easier access
df = pd.read_csv("unreconciled.csv")
subject = df.columns[0]
print(df.loc[0][0])

# count the amount of columns each landmark has
# how to handle the case where each subject is going to have a
# different number of landmarks?
l_counter = {"Tip of Blade": 0, "Each sinus": 0,
             "Each lobe tip where vein reaches margin": 0,
             "Start of petiole": 0, "Petiole meets blade": 0,
             "Each midrib/minor secondary vein": 0,
             "Each midrib/major secondary vein intersection": 0,
             "Width": 0, "Min. sinus width": 0, "Sinus next Length": 0}

# step 1:
# for loop to increment values of landmark counter to correspond with
# the amount of columns present in csv file for each landmark AND
# to figure out the starting index
start_index = 0
for i in landmarks:
    for name in df.columns:
        if name[0:12] == "Tip of Blade":
            l_counter["Tip of Blade"] += 1
            start_index = df.loc[name]
        elif i == name[0:len(i)]:
            l_counter[i] += 1

print("starting index for first run is:", start_index)

# Step 2:
# extract values from CSV file / dataframe by accessing each index,
# starting at the index (i, 0), where i is the number of the image
# are on, and 0 is the first column to get the subject ID.
# Then, start at the start_index to get the first landmark coordinate
# will be the blade_tip, then continue to sinus, have for loop which
# will have the range of the corresponding counter value in the
# counter dictionary

# feel that i should store the oak in the dictionary, so that
# i can make them all at once and not have to use a variable in
# the name of the Oak instance
# a dictionary would just have A LOT of nested items


def makeOaks(num_images):
    sinus_list = []
    curr = start_index
    for i in range(num_images):
        subject_id = df.loc(i, 0)
        # blade_tip_X = df.loc(i, curr). parse.nextInt??,
        # blade_tip_y = df.loc(i, curr). parse.nextInt??,
        curr += 1
        # create a list of tuples for the coordinates of
        # each sinus point
        for j in range(l_counter["Each sinus"]):
           # TODO: look up how to parse for integer in the entry of a csv file / dataframe
           # x = df.loc(i, curr). parse.nextInt??
           # y = df.loc(i, curr).parse.nextInt??
            curr += 1
            sinus_list.append(x, y)

# return OakDict, or Oak_num1
