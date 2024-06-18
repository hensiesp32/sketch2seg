import os
def num_check(photo_path,sketch_path):
    no_same_class =[]
    for _,dirs,_ in os.walk(photo_path):
        for dir in dirs:
            img_class_dir = os.path.join(photo_path,dir)
            for _,_,files in os.walk(img_class_dir):
                image_num = len(files)

            sk_class_dir = os.path.join(sketch_path,dir)
            for _,_,sks in os.walk(sk_class_dir):
                sketch_num = len(sks)

            if image_num != sketch_num:
                no_same_class.append(dir)
    
    abnormal_img = []
    for dir in no_same_class:
        img_class_dir = os.path.join(photo_path,dir)
        for _,_,files in os.walk(img_class_dir):
            for file in files:
                image_name = file.split(".")[0]
                sk_class_dir = os.path.join(sketch_path,dir)
                for _,_,sks in os.walk(sk_class_dir):
                    flag = False
                    import pdb
                    pdb.set_trace()
                    for sk in sks:
                        if image_name in sk:
                            flag = True
                            break
                if not flag:
                    img = os.path.join(img_class_dir,file)
                    abnormal_img.append(img)
    return abnormal_img


if __name__ == "__main__":
    photo = "/root/code_dir/ControlNet_retrival/datasets/50_data_retrival_train/photo"
    sketch = "/root/code_dir/ControlNet_retrival/datasets/50_data_retrival_train/sketch"
    abnormal_img = num_check(photo_path=photo,sketch_path=sketch)
    with open("unusual_img.txt","w") as fp:
        for line in abnormal_img:
            fp.write(line+'\n')
        fp.close()

"""
step 1 : 获取每个photo_dir 下的class_dir
step 2 : 获取每个class_dir下image数量，sketch数量
step 3 ：判断image数量与sketch数量是否相同，不同的class取出
step 4 : 对于不同的class，找出image_name
"""