import os  #通过os模块调用系统命令

file_path = "./Annotations/"  #文件路径
path_list = os.listdir(file_path) #遍历整个文件夹下的文件name并返回一个列表

path_name = []#定义一个空列表

for i in path_list:
    path_name.append(i.split(".")[0]) #若带有后缀名，利用循环遍历path_list列表，split去掉后缀名

#path_name.sort() #排序

for file_name in path_name:
    # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
    with open("./ImageSets/Main/train.txt", "a") as file:
        file.write(file_name + "\n")
        print(file_name)
    file.close()

