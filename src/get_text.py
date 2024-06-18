import os
import json

image_path = "/root/code_dir/ControlNet_Seg/datasets/25_test_data/anno"
save_path = "/root/code_dir/ControlNet_Seg/datasets/25_test_data/text/text.json"

plane_json = "/root/code_dir/ControlNet_Seg/datasets/dataset/text/airplane.json"
car_json = "/root/code_dir/ControlNet_Seg/datasets/dataset/text/car_sedan.json"
cat_json = "/root/code_dir/ControlNet_Seg/datasets/dataset/text/cat.json"
bird_json = "/root/code_dir/ControlNet_Seg/datasets/dataset/text/songbird.json"
bottle_json = "/root/code_dir/ControlNet_Seg/datasets/dataset/text/wine_bottle.json"
sofa_json = "/root/code_dir/ControlNet_Seg/datasets/dataset/text/couch.json"
chair_json = "/root/code_dir/ControlNet_Seg/datasets/dataset/text/chair.json"
horse_json = "/root/code_dir/ControlNet_Seg/datasets/dataset/text/horse.json"
dog_json = "/root/code_dir/ControlNet_Seg/datasets/dataset/text/dog.json"
cow_json = "/root/code_dir/ControlNet_Seg/datasets/dataset/text/cow.json"
boat_json = "/root/code_dir/ControlNet_Seg/datasets/dataset/text/sailboat.json"
bike_json = "/root/code_dir/ControlNet_Seg/datasets/dataset/text/bicycle.json"

text_json ={}
with open(plane_json,"r",encoding='utf-8') as pl:
    plane_info = json.load(pl)
with open(car_json,"r",encoding='utf-8') as ca:
    car_info = json.load(ca)    
with open(cat_json,"r",encoding='utf-8') as ct:
    cat_info = json.load(ct)
with open(bird_json,"r",encoding='utf-8') as bd:
    bird_info = json.load(bd)
with open(bottle_json,"r",encoding='utf-8') as bo:
    bottle_info = json.load(bo)
with open(sofa_json,"r",encoding='utf-8') as sf:
    sofa_info = json.load(sf)
with open(chair_json,"r",encoding='utf-8') as ch:
    chair_info = json.load(ch)
with open(horse_json,"r",encoding='utf-8') as ho:
    horse_info = json.load(ho)
with open(dog_json,"r",encoding='utf-8') as dg:
    dog_info = json.load(dg)
with open(cow_json,"r",encoding='utf-8') as cw:
    cow_info = json.load(cw)
with open(boat_json,"r",encoding='utf-8') as bt:
    boat_info = json.load(bt)
with open(bike_json,"r",encoding='utf-8') as bk:
    bike_info = json.load(bk)
for _,_,files in os.walk(image_path):
    for file in files:
        image_name = file.split(".")[0]
        if file not in list(text_json.keys()):
            for block in plane_info:
                if block["image_id"] == image_name:
                    text_json[file] = block["caption"]
                    break
        if file not in list(text_json.keys()):       
            for block in car_info:
                if block["image_id"] == image_name:
                    text_json[file] = block["caption"]
                    break
        if file not in list(text_json.keys()):        
            for block in cat_info:
                if block["image_id"] == image_name:
                    text_json[file] = block["caption"]
                    break
        if file not in list(text_json.keys()):
            for block in bird_info:
                if block["image_id"] == image_name:
                    text_json[file] = block["caption"]
                    break
        if file not in list(text_json.keys()): 
            for block in bottle_info:
                if block["image_id"] == image_name:
                    text_json[file] = block["caption"]
                    break
        if file not in list(text_json.keys()): 
            for block in sofa_info:
                if block["image_id"] == image_name:
                    text_json[file] = block["caption"]
                    break
        if file not in list(text_json.keys()): 
            for block in chair_info:
                if block["image_id"] == image_name:
                    text_json[file] = block["caption"]
                    break
        if file not in list(text_json.keys()): 
            for block in horse_info:
                if block["image_id"] == image_name:
                    text_json[file] = block["caption"]
                    break
        if file not in list(text_json.keys()): 
            for block in dog_info:
                if block["image_id"] == image_name:
                    text_json[file] = block["caption"]
                    break
        if file not in list(text_json.keys()): 
            for block in cow_info:
                if block["image_id"] == image_name:
                    text_json[file] = block["caption"]
                    break
        if file not in list(text_json.keys()): 
            for block in boat_info:
                if block["image_id"] == image_name:
                    text_json[file] = block["caption"]
                    break
        if file not in list(text_json.keys()): 
            for block in bike_info:
                if block["image_id"] == image_name:
                    text_json[file] = block["caption"]
                    break
with open(save_path,"w",encoding='utf-8') as tx:
    json.dump(text_json,tx)
tx.close()
pl.close()
ca.close()
ct.close()
bd.close()
bo.close()
sf.close()
ch.close()
ho.close()
dg.close()
cw.close()
bt.close()
bk.close()
