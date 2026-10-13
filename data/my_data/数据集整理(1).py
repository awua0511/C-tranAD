#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd


# In[24]:


import numpy as np


# In[25]:


import glob


# In[26]:


import tqdm


# In[ ]:


def read_data(file_name):
    with open(file_name, 'r') as f:
        ret = f.read()
    lines = ret.split('\n')
    ret = [line.split(',') for line in lines[4:]]
    df = pd.DataFrame(ret)
    df.drop(columns=0,inplace=True)
    df.drop(columns=28,inplace=True)
    df = df.iloc[:-1,]
    return df
    


# In[28]:


def scaler_df(df):
    df[df.columns[1:]] = df[df.columns[1:]].astype('float32')
    df[df.columns[1:]] = df[df.columns[1:]].apply(lambda x: (x - x.mean()) / x.std())
    return df
    


# In[29]:


def sliding_window(data, window_size=512, stride=64):
    """
    滑动窗口函数，将数据转换为（M，窗口大小，43）的格式
    
    Parameters:
    data (numpy.ndarray): 输入数据，形状为（n, 43）
    window_size (int): 窗口大小
    stride (int): 滑动步长
    Returns:
    numpy.ndarray: 转换后的数据，形状为（M，窗口大小，43）
    """
    # 计算可以形成的窗口数量
    num_windows = (len(data) - window_size) // stride + 1
    
    # 初始化一个空列表来存储窗口数据
    nor_data = []
    exp_data = []
    
    
    # 滑动窗口并收集数据
    for i in range(0, num_windows * stride, stride):
        window = data.iloc[i:i+window_size]

        a = window[1].str.contains('WER')
        tmp_value = sum(window[1].str.contains('WER'))

        # print(tmp_value)
        
        if tmp_value > 1:
            exp_data.append(window.iloc[:,1:].values)
        elif tmp_value <= 1:
            nor_data.append(window.iloc[:,1:].values)

    
    return np.array(nor_data).astype(np.float32), np.array(exp_data).astype(np.float32)


# In[30]:


# df = read_data('./样本\\group21\\006014.csv')
# df = scaler_df(df)
# nor_data, exp_data = sliding_window(df)


# In[31]:


# exp_data


# In[32]:


files = glob.glob("E:\project\咸鱼\丘成桐科学奖\programming\group21(1)\group21\*.csv")


# In[33]:


files


# In[34]:


all_nor = []
all_exp = []


# In[35]:


for file_name in tqdm.tqdm(files):
    try:
        df = read_data(file_name)
        df = scaler_df(df)
        nor_data, exp_data = sliding_window(df)
        all_nor.append(nor_data)
        all_exp.append(exp_data)
    except Exception as e:
        pass
    
    


# In[36]:


new_exp = []
for j in all_exp:
    if len(j) != 0:
        new_exp.append(j)


# In[ ]:





# In[37]:


all_nor = np.vstack(all_nor)


# In[38]:


all_exp = np.vstack(new_exp)


# In[39]:


all_nor.shape, all_exp.shape


# In[40]:


np.save('../../../../programming/all_nor.npy', all_nor)


# In[41]:


np.save('../../../../programming/all_exp.npy', all_exp)


# In[ ]:




