#!/usr/bin/env python
# coding: utf-8
import os
import cv2
from glob import glob
from osgeo import gdal, ogr, osr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import earthpy as et
from skimage import data
from skimage import filters
from skimage import exposure
from skimage import morphology
from skimage import measure
import re
import time
import json

print(dir(os))
# print(dir(np))
# print(dir(gdal))
np.seterr(all="ignore")  # Take no action when the exception occurs
# In[37]:
os.chdir(r"D:\department_daily\数据挖掘\影像云掩膜\算法上交\20210803\数据\H1C_OPER_CZI_L1C_20200130T032809_20200130T032904_07311_10 ")   # 用于改变当前工作目录到指定的路径
in_path = 'H1C_OPER_CZI_L1C_20200130T032809_20200130T032904_07311_10.tiff'
ds = gdal.Open(in_path)
dir(ds)
# In[51]:
# print(ds)

def tiff_sta(in_path):
    # 函数说明
    dataset=gdal.Open(in_path)    #打开文件
    proj=dataset.GetProjection()
    print(proj)
    rows=dataset.RasterXSize()
    cols=dataset.RasterYSize()
    print(rows)
    print(cols)

    for b in range(dataset.RasterCount):       #波段数
        band=dataset.GetRasterBand(b + 1)      #读取一个波段，其参数为波段的索引号
        nodata=band.GetNoDataValue()
        #         nodata = 0
        band_array = band.ReadAsArray()
        num_nodata = (band_array == nodata).sum()
        band_array = band_array[band_array != nodata]
        hist, bins = np.histogram(band_array, 20, [np.min(band_array), np.max(band_array)])  # , density=True

        sta = {
            'All_Number_of_Bands': str(ds.RasterCount),
            'Number_of_Bands': "Band_" + str(b + 1),
            'Data_Type': str(gdal.GetDataTypeName(band.DataType)),
            'NoData_Value': str(nodata),
            'Num_NoData': str(num_nodata),
            'Min': str(np.min(band_array)),
            'Max': str(np.max(band_array)),
            'Mean': str(np.mean(band_array)),
            'Median': str(np.median(band_array)),
            'Var': str(np.var(band_array)),
            'Std': str(np.std(band_array, ddof=1)),
            #             'Hist': hist.tolist(),
            #             'Bins': bins.tolist()
        }

        js_sta = json.dumps(sta, indent=2, ensure_ascii=False)
        print(js_sta)

        fig, ax1 = plt.subplots()
        plt.set_cmap('RdBu')
        x = np.arange(len(bins) - 1)
        plt.bar(x, hist)
        plt.xticks(range(len(bins)), bins, rotation=50)
        title = "Histogram  " + "Band_" + str(b + 1)
        plt.xlabel(u'Data Value')
        plt.ylabel(u'Count')
        plt.title(title)
        plt.rcParams['figure.facecolor'] = 'white'
        print("This image has " + str(ds.RasterCount) + " bands, " + "Band_" + str(b + 1) + " statistics completed.")


# In[53]:

tiff_sta(in_path)
# In[54]:
rows = ds.RasterYSize
cols = ds.RasterXSize
bands = ds.RasterCount
print(f"rows:{rows}", f"cols:{cols}", f"bands:{bands}")
# In[55]:

blue = ds.GetRasterBand(1).ReadAsArray()
print(blue)
plt.figure(figsize=(13, 5))
plt.imshow(blue, cmap='gray', interpolation='nearest')
plt.colorbar()

# print(blue.dtype)
# blue[np.where(blue == 0)] = np.nan
# print(blue)
# plt.figure(figsize=(13, 5))
# plt.imshow(blue, cmap='gray', interpolation='nearest')
# plt.colorbar()


# In[56]:


# hist, bins_center = exposure.histogram(blue) [np.where(blue != 0)]


# In[62]:

iso = filters.threshold_isodata(blue)
print("iso:", iso)
# In[58]:

otsu = filters.threshold_otsu(blue)
print("otsu:", otsu)
# In[ ]:


# In[59]:
mean = filters.threshold_mean(blue)
print("mean:", mean)

# In[50]:


mome = cv2.moments(blue)
print("mome:", mome)

# In[48]:


mome = measure.moments(blue)
print("mome:", mome)


# In[67]:


def calcGrayHist(image):
    rows, cols = image.shape
    grayHist = np.zeros([256])
    for r in range(rows):
        for c in range(cols):
            grayHist[int(image[r, c])] += 1

    return grayHist


# In[65]:


def threshEntroy(image):
    rows, cols = image.shape
    # 获取直方图
    grayHist = calcGrayHist(image)
    # 得到概率直方图
    normgrayHist = grayHist / float(rows * cols)

    zeroCumuMoment = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            zeroCumuMoment[k] = normgrayHist[k]
        else:
            zeroCumuMoment[k] = zeroCumuMoment[k - 1] + normgrayHist[k]
    entropy = np.zeros([256], np.float32)
    # 计算熵
    for k in range(256):
        if k == 0:
            if normgrayHist[k] == 0:
                entropy[k] = 0
            else:
                entropy[k] = -normgrayHist[k] * np.log10(normgrayHist[k])
        else:
            if normgrayHist[k] == 0:
                entropy[k] = entropy[k - 1]
            else:
                entropy[k] = entropy[k - 1] - normgrayHist[k] * np.log10(normgrayHist[k])

    ft = np.zeros([256], np.float32)
    ft1, ft2 = 0., 0.
    totalEntropy = entropy[255]
    for k in range(255):
        # 找最大值
        maxfornt = np.max(normgrayHist[:k + 1])
        maxback = np.max(normgrayHist[k + 1:256])
        if (maxfornt == 0 or zeroCumuMoment[k] == 0 or maxfornt == 1 or zeroCumuMoment[k] == 1 or totalEntropy == 0):
            ft1 = 0
        else:
            ft1 = entropy[k] / totalEntropy * (np.log10(zeroCumuMoment[k]) / np.log10(maxfornt))
        if (maxback == 0 or 1 - zeroCumuMoment[k] == 0 or maxback == 1 or 1 - zeroCumuMoment[k] == 1):
            ft2 = 0
        else:
            if totalEntropy == 0:
                ft2 = (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
            else:
                ft2 = (1 - entropy[k] / totalEntropy) * (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
        ft[k] = ft1 + ft2
    # 找出最大值的索引，作为得到的阈值
    thresloc = np.where(ft == np.max(ft))
    thresh = thresloc[0][0]

    # 阈值处理
    threshold = np.copy(image)
    threshold[threshold > thresh] = 255
    threshold[threshold <= thresh] = 0
    # 返回分割图像，最大阈值，最大熵和熵
    return threshold, thresh, max(ft), entropy


# In[ ]:


b, a, c, e = threshEntroy(blue)
# 结果显示
# b=np.expand_dims(b,axis=-1)
# d=array_to_img(b)

# d.shape
# plt.imshow(d)
cv2.imshow("f", b)
cv2.waitKey()

# In[ ]:




