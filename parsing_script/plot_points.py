# Script to plot the points from parsing_script onto an image

import matplotlib.pyplot as plt
import parsing_script
import matplotlib.image as mpimg

# test if parsing_script works in this one

myOak = parsing_script.makeOaks(0)

# image name is hardcoded for now, can later modify to match up
# subject id with the image name, and pull from the whole folder
myImage = mpimg.imread('test_image.jpeg')
# print(myImage.shape[0])
plt.figure(figsize=(10, 10))
plt.imshow(myImage)


#sinus_dict = getattr(myOak, 'sinus_major')
# print(sinus_dict[1])


def plot_points(oak, feature, color):
    my_dict = getattr(oak, feature)
    for i in range(len(my_dict)):
        curr_tup = my_dict[i+1]
        plt.plot(curr_tup[0], curr_tup[1], color)
        #print(f"Plotted at {curr_tup[0]}, {curr_tup[1]}")


def plot_line(oak, feature, color):
    my_tup = getattr(oak, feature)
    # this assumes that the tuple is in the order
    # (x1, y1, x2, y2)
    x = [my_tup[0], my_tup[2]]
    y = [my_tup[1], my_tup[3]]
    plt.plot(x, y, c=color)

# we have to call the plotting function for each landmark,
# it is a separate function for points vs lines


plot_points(myOak, 'blade_tip', 'r.')
plot_points(myOak, 'sinus_major', 'm.')
plot_points(myOak, 'lobe_tip_margin', 'y.')
plot_points(myOak, 'petiole_tip', 'g.')
plot_points(myOak, 'petiole_blade', 'y.')
plot_points(myOak, 'major_secondary', 'b.')
plot_points(myOak, 'minor_secondary', 'k.')
plot_line(myOak, 'max_width', 'b')
plot_line(myOak, 'min_width', 'r')
plot_line(myOak, 'next_width', 'y')

plt.show()
