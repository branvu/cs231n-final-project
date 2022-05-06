import os

PATH = "data/jsl"
PATH2 = "data/isl"
PATH3 = "data/asl"
file_obj = open("annotations.csv", "a")
# american - 0, japanese - 1, irish - 2
# prefix for category is language

for im_name in os.listdir(PATH3):
    # asl-0-1
    liststr = im_name.split('-')
    category = "0" + liststr[1]
    annotation = PATH3 + "/" + im_name + ",0," + category + "\n"
    file_obj.write(annotation)
    print("writing ", annotation)

# Read from irish
for im_name in os.listdir(PATH2):
    # isl-0
    liststr = im_name.split('-')
    category = "2" + liststr[1]
    annotation = PATH2 + "/" + im_name + ",2," + category + "\n"
    file_obj.write(annotation)
    print("writing ", annotation)

# # Read from folder JSL
for im_name in os.listdir(PATH):
    liststr = im_name.split('-')
    category = "1" + liststr[1]
    annotation = PATH + "/" + im_name + ",1," + category + "\n"
    file_obj.write(annotation)
    print("writing ", annotation)
