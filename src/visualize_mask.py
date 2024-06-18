from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def save_image(original_image,mask_image,save_path):
    label_img = transPNG(mask_image)
    label_img = np.array(label_img)
    fig = plt.figure(figsize=(256/80, 256/80))
    plt.imshow(original_image,alpha=1.0)
    plt.imshow(label_img,alpha=0.6)
    plt.axis('off')
    fig.savefig(save_path,bbox_inches="tight",pad_inches = 0,transparent=True,dpi = 104)
    plt.close()

def get_color(image_path,seg_path):
    image1 = Image.open(image_path)
    mask = np.load(seg_path)
    image2 = image2 = np.zeros((256,256,3))
    ## 设置分割图的颜色 rgb
    image2[:,:,0] = np.where(mask==1,255,255)
    image2[:,:,1] = np.where(mask == 1,255,255)
    image2[:,:,2] = np.where(mask ==1,0,255)
    image = Image.fromarray(np.uint8(image2))
    return image,image1

def transPNG(img):
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = list()
    for item in datas:
        if item[0] > 220 and item[1] > 220 and item[2] > 220:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img

if __name__ =="__main__":
    img_root_path = "/root/code_dir/ControlNet_Seg/datasets/PASCAL/pascal_voc_256/photo"
    save_path = "/root/code_dir/ControlNet_Seg/exp_result/exp23_voc/visualization"
    gt_path = "/root/code_dir/ControlNet_Seg/datasets/PASCAL/pascal_voc_256/anno_npy"
    pre_path = "/root/code_dir/ControlNet_Seg/exp_result/exp23_voc/predictions"
    for _,dirs,_ in os.walk(img_root_path):
        for dir in dirs:
            dir_path = os.path.join(img_root_path,dir)
            gt_dir_path = os.path.join(gt_path,dir)
            for _,_,files in os.walk(dir_path):
                for file in files:
                    gt_save_name = "gt_" + file
                    pre_save_name = "pre_" + file
                    npy_file_name = file.split(".")[0] + ".npy"
                    file_path = os.path.join(dir_path,file)
                    gt_mask_path = os.path.join(gt_dir_path,npy_file_name)
                    pre_mask_path = os.path.join(pre_path,npy_file_name)
                    image,image1 = get_color(file_path,pre_mask_path) ## ground_truth和prediction使用不同的颜色
                    gt_save_path = os.path.join(save_path,gt_save_name)
                    pre_save_path = os.path.join(save_path,pre_save_name)
                    save_image(image1,image,pre_save_path)  ## 

