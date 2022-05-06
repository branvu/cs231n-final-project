import os

PATH = "data/jsl"
PATH2 = "data/isl"
PATH3 = "data/asl"
file_obj = open("annotations.csv", "a")
# american - 0, japanese - 1, irish - 2
# prefix for category is language

# for im_name in os.listdir(PATH3):
#     # asl-0-1
#     category = "0" + im_name[4]
#     annotation = PATH + "/" + im_name + ",0," + category + "\n"
#     file_obj.write(annotation)
#     print("writing ", annotation)

# # Read from irish
# for im_name in os.listdir(PATH2):
#     # isl-0
#     category = "2" + im_name[4]
#     annotation = PATH + "/" + im_name + ",2," + category + "\n"
#     file_obj.write(annotation)
#     print("writing ", annotation)

# # Read from folder JSL
for im_name in os.listdir(PATH):
    i = len(im_name) - 5
    while i > 0 and im_name[i].isnumeric():
        i -= 1
    category = "1" + im_name[i + 1:len(im_name) - 4]
    annotation = PATH + "/" + im_name + ",1," + category + "\n"
    file_obj.write(annotation)
    print("writing ", annotation)

