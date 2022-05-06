import os

PATH = "data/jsl"
file_obj = open("annotations.csv", "a")
# Read from folder
for im_name in os.listdir(PATH):
    i = len(im_name) - 5
    while i > 0 and im_name[i].isnumeric():
        i -= 1
    category = "j" + im_name[i + 1:len(im_name) - 4]
    annotation = PATH + "/" + im_name + ",japanese," + category + "\n"
    file_obj.write(annotation)
    print("writing ", annotation)

