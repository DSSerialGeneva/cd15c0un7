#!/usr/bin/python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output

import io
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data

print(check_output(["ls", "../input"]).decode("utf8"))

data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))

prod_to_category = dict()

for c, d in enumerate(data):
    product_id = d['_id']
    category_id = d['category_id'] # This won't be in Test data
    print ("Product ID = ["+str(product_id)+"] - Category ID = ["+str(category_id)+"] - NB images = ["+str(len(d['imgs']))+"]")
    prod_to_category[product_id] = category_id
    img_index = 1
    for e, pic in enumerate(d['imgs']):
        picture = imread(io.BytesIO(pic['picture']))
        #plt.imshow(picture)
        #print (str(picture))

        pixel_index = 0
        print ("  Image ["+str(img_index)+"] size = "+str(len(pic['picture']))+" bytes")
        #for pixel in pic['picture']: 
        #    print (str(pixel_index)+": "+str(pixel))
        #    pixel_index += 1

        #newFileByteArray = pic['picture']
        #newFile = open("../images/"+str(product_id)+"_img_"+str(img_index)+".jpg", "wb")
        #newFile.write(newFileByteArray)
        
        img_index += 1
        #break
    #break

    #prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
    #prod_to_category.index.name = '_id'
    #prod_to_category.rename(columns={0: 'category_id'}, inplace=True)

    #prod_to_category.head()
