#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[5]:


df =pd.read_csv("veri.csv")


# In[13]:


df.head(3)


# In[8]:


veri.tail(5)


# In[9]:


df = veri.copy()


# In[14]:


df.info()


# In[15]:


df.dtypes


# In[8]:


df.shape


# In[9]:


df.columns


# In[12]:


df.describe().T


# In[7]:


sil = df.drop(labels =['mahalle_koy_uavt'],axis =1)


# In[8]:


sil.head()


# In[9]:


df[df['ilce_adi']!='ADALAR']


# In[14]:


sil.drop_duplicates()


# In[10]:


df.groupby(by=['ilce_adi']).mean()


# In[11]:


kategori = df.select_dtypes(include = ["object"])
kategori


# In[12]:


kategori2 = df.select_dtypes(include = ["int"])


# In[18]:


kategori2


# In[19]:


df.groupby('ilce_adi')['can_kaybi_sayisi'].count()


# In[23]:


df.groupby('ilce_adi')['icme_suyu_boru_hasari'].mean()


# In[24]:


df.groupby(["mahalle_koy_uavt"])["hastanede_tedavi_sayisi"].mean()


# In[26]:


df.groupby(["mahalle_adi"])["can_kaybi_sayisi"].mean()


# In[28]:


df.groupby(["mahalle_adi"])["hastanede_tedavi_sayisi"].mean()


# In[14]:


df.groupby("ilce_adi").aggregate([min, max])


# In[30]:


for name in df['gecici_barinma'][:14]:
    print(name)


# In[31]:


df["ilce_adi"].value_counts()


# In[32]:


df["ilce_adi"].value_counts().plot.barh()


# In[33]:


df["gecici_barinma"].fillna(0, inplace = True)


# In[34]:


df.isnull().sum()


# In[35]:


df.isnull().values.any()


# In[36]:


kategori.ilce_adi.unique()


# In[37]:


kategori["ilce_adi"].value_counts().count()


# In[38]:


df["atik_su_boru_hasari"].value_counts().plot.barh().set_title("atık suyu borusu hasar kaydı");


# In[43]:


df["dogalgaz_boru_hasari"].value_counts().plot.barh().set_title("doğalgaz borusu hasar grafiği");


# In[49]:


sns.barplot(x = "atik_su_boru_hasari", y = df.atik_su_boru_hasari.index, data= df);


# In[52]:


sns.barplot(x = "icme_suyu_boru_hasari", y = df.icme_suyu_boru_hasari.index, data= df);


# In[55]:


sns.barplot(x = "can_kaybi_sayisi", y = df.can_kaybi_sayisi.index, data= df);


# In[56]:


sns.catplot(x = "icme_suyu_boru_hasari", y = "ilce_adi", data = df);


# In[57]:


sns.catplot(x = "gecici_barinma", y = "ilce_adi", data = df);


# In[58]:


sns.catplot(x = "hastanede_tedavi_sayisi", y = "ilce_adi", data = df);


# In[59]:


get_ipython().run_line_magic('pinfo', 'sns.distplot')


# In[60]:


sns.kdeplot(df.cok_agir_hasarli_bina_sayisi, shade = True);


# In[61]:


sns.kdeplot(df.icme_suyu_boru_hasari, shade = True);


# In[62]:


sns.boxplot(x = "gecici_barinma", y = "ilce_adi", data = df);


# In[63]:


sns.lmplot(x = "gecici_barinma", y = "mahalle_koy_uavt", data = df);


# In[64]:


sns.lmplot(x = "icme_suyu_boru_hasari", y = "mahalle_koy_uavt", data = df);


# In[65]:


sns.catplot(x = "cok_agir_hasarli_bina_sayisi", y = "ilce_adi", data = df);


# In[66]:


sns.lineplot(x = "dogalgaz_boru_hasari", y = "ilce_adi", data = df);


# In[67]:


sns.lineplot(x = "gecici_barinma", y = "ilce_adi", hue = "cok_agir_hasarli_bina_sayisi", data = df);


# In[56]:


sns.pairplot(df);


# In[55]:


sns.pairplot(df, hue = "ilce_adi");


# In[68]:


sns.lineplot(x = "can_kaybi_sayisi", y = "ilce_adi", data = df);


# In[1]:


import researchpy as rp


# In[9]:


rp.summary_cont(df[["agir_yarali_sayisi", "hastanede_tedavi_sayisi", "can_kaybi_sayisi"]])


# In[10]:


rp.summary_cat(df[["agir_yarali_sayisi", "hastanede_tedavi_sayisi", "can_kaybi_sayisi"]])


# In[11]:


df[["cok_agir_hasarli_bina_sayisi", "can_kaybi_sayisi"]].cov()


# In[12]:


df[["hastanede_tedavi_sayisi", "can_kaybi_sayisi"]].cov()


# In[13]:


df[["hastanede_tedavi_sayisi", "can_kaybi_sayisi"]].corr()


# In[14]:


df[["cok_agir_hasarli_bina_sayisi", "can_kaybi_sayisi"]].corr()


# In[15]:


df[["agir_yarali_sayisi", "hastanede_tedavi_sayisi"]].corr()


# In[16]:


df.mean


# In[17]:


import seaborn as sns


# In[18]:


veri = sns.load_dataset("veri")


# In[19]:


df = veri.copy()
df


# In[20]:


veri= df["hafif_hasarli_bina_sayisi"]


# In[21]:


veri.head()


# In[22]:


sns.boxplot(x = veri);


# In[23]:


Q1 =veri.quantile(0.25)


# In[24]:


Q3 = veri.quantile(0.75)


# In[25]:


IQR = Q3-Q1


# In[26]:


Q1


# In[27]:


Q3


# In[28]:


IQR


# In[32]:


alt_sinir = Q1- 1.5*IQR


# In[33]:


alt_sinir


# In[30]:


ust_sinir = Q3 + 1.5*IQR


# In[31]:


ust_sinir


# In[34]:


aykiri_tf=(veri < alt_sinir) | (veri > ust_sinir)


# In[35]:


aykiri_tf


# In[36]:


veri[aykiri_tf]


# In[37]:


veri.isnull().sum()


# In[38]:


veri.notnull().sum()


# In[39]:


veri.isnull().sum().sum()


# In[40]:


veri.isnull()


# In[41]:


df[df.isnull().any(axis = 1)]


# In[42]:


df[df.notnull().all(axis = 1)]


# In[43]:


df[df["ilce_adi"].notnull() & df["can_kaybi_sayisi"].notnull()& df["gecici_barinma"].notnull()]


# In[44]:


df.dropna()


# In[45]:


df["agir_hasarli_bina_sayisi"]


# In[46]:


df["agir_hasarli_bina_sayisi"].mean()


# In[47]:


df["agir_hasarli_bina_sayisi"].fillna(df["agir_hasarli_bina_sayisi"].mean())


# In[48]:


df.groupby("ilce_adi")["can_kaybi_sayisi"].mean()


# In[49]:


df["gecici_barinma"].fillna(df.groupby("mahalle_koy_uavt")["cok_agir_hasarli_bina_sayisi"].transform("mean"))


# In[50]:


from sklearn.preprocessing import LabelEncoder


# In[51]:


lbe = LabelEncoder()


# In[52]:


lbe.fit_transform(df["mahalle_adi"])


# In[53]:


df["ad_index"] = lbe.fit_transform(df["mahalle_adi"])


# In[54]:


df


# In[ ]:




