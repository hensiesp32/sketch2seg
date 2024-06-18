import os
import json

def get_file_list(photo_path,sketch_path):
    image_path_dict = {}
    sk_path_dict = {}
    for _,dirs,_ in os.walk(photo_path):
        for dir in dirs:
            image_class_path = os.path.join(photo_path,dir)
            sk_class_path = os.path.join(sketch_path,dir)
            for _,_,files in os.walk(image_class_path):
                image_path_list = []
                sk_path_list = []
                for file in files:
                    image_path = os.path.join(image_class_path,file)
                    image_path_list.append(image_path)
                    file_name = file.split(".")[0]
                    for _,_,sks in os.walk(sk_class_path):
                        for sk in sks:
                            if file_name in sk:
                                sk_path = os.path.join(sk_class_path,sk)
                                sk_path_list.append(sk_path)
                                break
            image_path_dict[dir] = image_path_list.copy()
            sk_path_dict[dir] = sk_path_list.copy()
    return image_path_dict,sk_path_dict


# if __name__ =="__main__":
#     photo_path = "/root/code_dir/ControlNet_retrival/datasets/new_data_retirval_train/photo"
#     sketch_path = "/root/code_dir/ControlNet_retrival/datasets/new_data_retirval_train/sketch"
#     image_path_dict,sk_path_dict = get_file_list(photo_path,sketch_path)

#     # with open("image_path.json","w") as fp:
#     #     json.dump(image_path_dict,fp)
#     #     fp.close()
#     with open("sketch_path.json","w") as fp:
#         json.dump(sk_path_dict,fp)
#         fp.close()

"""
step1 : 遍历所有的class dir
step2 : 遍历每个dir中的image
step3 ：把所有的image的path放入list中
"""