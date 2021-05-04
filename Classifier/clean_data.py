import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from PIL import Image

classes = ["face_mask","face"]
def convert_annotation(image_id):
        flag = 0
        in_file = open('train/annotation/%s.xml' % (image_id))
        img_rgb = Image.open('train/img/%s.jpg' % (image_id))
        img = img_rgb.convert('RGB')
        tree = ET.parse(in_file)
        root = tree.getroot()

        size = root.find('size')


        output = ''
        i = 0
        for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                        continue

                flag = 1
                cls_id = classes.index(cls) + 1
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))
                #bb = convert((w, h), b)
                img_name = image_id +'_' +str(i)
                img1 = img.crop((b[0], b[2], b[1], b[3]))
                if cls_id == 1:
                        img1.save("train/labels/with_mask/%s.jpg" %(img_name))
                else:
                        img1.save("train/labels/without_mask/%s.jpg" % (img_name))
                output = output + str(cls_id) + " " +" ".join([str(a) for a in b]) +"\n"
                i += 1

        #out_file = open('train/%s.txt' % (image_id), 'w')
        #out_file.write(output)

        return flag
def convert_annotation1(image_id):
        flag = 0
        in_file = open('train/annotation/%s.xml' % (image_id))
        img_rgb = Image.open('train/img/%s.jpg' % (image_id))
        img = img_rgb.convert('RGB')
        tree = ET.parse(in_file)
        root = tree.getroot()

        size = root.find('size')


        output = ''
        i = 0
        label_list = []
        for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                        continue

                flag = 1
                cls_id = classes.index(cls) + 1
                if len(label_list) != 0 and cls_id not in label_list:
                        return
                label_list.append(cls_id)
        if len(label_list) == 0:
                return
        if label_list[0] == 1:
                img.save("train/new/with_mask/%s.jpg" % (image_id))
        else:
                img.save("train/new/without_mask/%s.jpg" % (image_id))

        #out_file = open('train/%s.txt' % (image_id), 'w')
        #out_file.write(output)

        return flag

def getFileID(file_dir):
    L=[]
    for file in os.listdir(file_dir):
        L.append(os.path.splitext(file)[0])
    return L

wd = os.getcwd()

image_ids = getFileID('train/img/')
for img in image_ids:
        convert_annotation1(img)

