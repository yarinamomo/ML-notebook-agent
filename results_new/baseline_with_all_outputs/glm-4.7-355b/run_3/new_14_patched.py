# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
df= pd.read_csv('data/Financials.csv')
df.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
#Sütun isimlerindeki boşlukları silmek için
df.columns= df.columns.str.strip()

#Temizlenmesi gereken parasal sütunlsrı belirlemek için
para_sutunlari= ['Units Sold','Manufacturing Price','Sale Price','Gross Sales','Discounts','Sales','COGS','Profit']

#Temizliyoruz
for col in para_sutunlari:
    #Sütun zaten sayı tipindeyse(float veya int) dokunmadan atlasın diye)
    if pd.api.types.is_numeric_dtype(df[col]):
        continue
    
    #önce metne çevirerek karakterleri temizliyoruz
    df[col]= df[col].astype(str)
    df[col]= df[col].str.replace('$','',regex=False)
    df[col]= df[col].str.replace(',','',regex=False)
    df[col]= df[col].str.replace(')','',regex=False)
    df[col]= df[col].str.replace('(','-',regex=False)

    #Tekrardan sayıya (float) çeviriyoruz./ Hata verirse değeri NaN ile değiştir.
    df[col]= pd.to_numeric(df[col], errors='coerce')

#kontrol
df.info()

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
print(df.isnull().sum())

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
#sayısal sütunlardaki boşlukları MEDYAN ile dolduralım.
for col in df.select_dtypes(include=['float64','int64']).columns:
    df[col]= df[col].fillna(df[col].mode()[0])

#Metin sütunlarını en çok tekrar edenle(mod) ile dolduralım garanti olsun
for col in df.select_dtypes(include=['object']).columns:
    df[col]= df[col].fillna(df[col].mode()[0])

#kontrol
df.isnull().sum()

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
import matplotlib.pyplot as plt
import seaborn as sns

#grafik boyutlandırma
plt.figure(figsize=(12, 6))

#sales ve profit sütunları için grafik
sns.boxplot(data=df[['Sales', 'Profit']])
plt.title("Aykırı Değer Kontolü(Noktalar = Aykırı Değerler)")
plt.show()

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# Aykırı değerleri sınırlara eşitleyen fonksiyon
def aykiri_degerleri_baskila(df, sutun_listesi):
    for col in sutun_listesi:
        #çeyreklçeri hesapla
        Q1= df[col].quantile(0.25)
        Q3= df[col].quantile(0.25)
        IQR= Q3 - Q1

        #Alt ve Üst sınırları belirle
        alt_sinir= Q1 - 1.5*IQR
        ust_sinir= Q3 + 1.5*IQR

        #Sınırlardan taşanları(Aykırı değerleri) sınırlara eşitle
        df.loc[df[col]< alt_sinir,col]= alt_sinir
        df.loc[df[col]> ust_sinir,col]= ust_sinir

        return df

#Hangi sütunları düzelteceğiz?(sayısal olanlar)
duzeltilecekler= ['Sales','Units Sold','Profit', 'Gross Sales','COGS','Manufacturing Price']

#Fonksiyonu Çalıştır
df= aykiri_degerleri_baskila(df, duzeltilecekler)

#Aykırı değerler traşlandı yani absürt fazla veya az değerleri normale çektik.

#Tekrar grafiğe bakıp kontrol edelim
sns.boxplot(data=df[['Sales','Profit']])
plt.show()

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# === BEFORE (original) ===
# def kesin_tiraslama(df, sutun_listesi):
#     for col in sutun_listesi:
#         alt_sinir= df['Profit'].quantile(0.05)
#         ust_sinir= df['Profit'].quantile(0.95)
# 
#         #sınırların dışındaki değerleri çekelim
#         df['Profit']= df['Profit'].clip(lower=alt_sinir, upper=ust_sinir)
#         return df,
# 
# #Sütunları tekrar belirleyelim
# duzeltilecekler= ['Sales','Units Sold', 'Gross Sale','COGS','Manufacturing Price']
# 
# #Traşla
# df= kesin_tiraslama(df, duzeltilecekler)
# 
# #grafiğe bakalım
# sns.boxplot(data=df[['Sales','Profit']])
# plt.show()

# === AFTER (edited) ===
def kesin_tiraslama(df, sutun_listesi):
    for col in sutun_listesi:
        alt_sinir= df['Profit'].quantile(0.05)
        ust_sinir= df['Profit'].quantile(0.95)


        df['Profit']= df['Profit'].clip(lower=alt_sinir, upper=ust_sinir)
        return df

duzeltilecekler= ['Sales','Units Sold', 'Gross Sales','COGS','Manufacturing Price']


df= kesin_tiraslama(df, duzeltilecekler)


sns.boxplot(data=df[['Sales','Profit']])
plt.show()