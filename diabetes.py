import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


df_ = pd.read_csv(r"C:\Users\ASUS\Desktop\Miuul\Özellik Mühendisliği hafta 6\feature_engineering-220803-214037\feature_engineering\feature_engineering\datasets\diabetes.csv")

df = df_

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df.head(10)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

##########################################################################################


##########################################################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)} {cat_cols}')
    print(f'num_cols: {len(num_cols)} {num_cols}')
    print(f'cat_but_car: {len(cat_but_car)} {cat_but_car}')
    print(f'num_but_cat: {len(num_but_cat)} {num_but_cat}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

##########################################################################################


# KATEGORİK DEĞİŞKENLERİN ANALİZİ
def cat_summary(dataframe, col_names, plot=False):

    if not isinstance(col_names, list):
        col_names = [col_names]


    for col_name in col_names:
        print(f"Analiz edilen sütun: {col_name}")
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.title(f"{col_name} için Frekans Grafiği")
            plt.show(block = True)

cat_summary(df, cat_cols, plot=True)


##########################################################################################

# NUMERİK DEĞİŞKENLERİN ANALİZİ
def num_summary(dataframe, numerical_col, plot=False):
    # Sayısal değişkenlerin özet istatistiklerini yazdır
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        # Histogram çiz
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)  # Birinci grafiği çiz
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(f"{numerical_col} Histogram")

        # Box plot çiz
        plt.subplot(1, 2, 2)  # İkinci grafiği çiz
        dataframe.boxplot(column=numerical_col)
        plt.title(f"{numerical_col} Box Plot")

        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

##########################################################################################


# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
def target_mean_by_num(dataframe, target, numerical_col):
    # Hedef değişken göre numerik sütunların ortalamasını gösterir.
    print(f"{target} değişkenine göre {numerical_col} ortalamaları:", end="\n\n")
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_mean_by_num(df, "Outcome", col)

##########################################################################################

# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ


def target_summary_with_cat(dataframe, target, categorical_cols):
    for col in categorical_cols:
        print(f"\n{col} sütununa göre {target} dağılımı:")
        print(pd.DataFrame({"Count": dataframe.groupby(col)[target].value_counts(),
                            "Ratio": dataframe.groupby(col)[target].value_counts(normalize=True)}))
        print("##########################################")


target_summary_with_cat(df, 'Outcome', cat_cols)



##########################################################################################


# Aykırı Değerler

def outlier_thresholds(dataframe, col_name, q1=0.15, q3=0.85):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    print(f"{col_name} = low limit: {low_limit} up limit= {up_limit}")

    if low_limit < 0:
        low_limit = 0
    print(f"{col_name} = new low limit: {low_limit} up limit= {up_limit}", end="\n\n")
    return low_limit, up_limit


for col in num_cols:
     low_limit, up_limit = outlier_thresholds(df, col)

##########################################################################################
# Aykırı Değer Var mı Yok mu?
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outliers = dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)]
    if len(outliers) > 0:
            return True
    else:
            return False


for col in num_cols:
    print(col, check_outlier(df, col), end="\n\n")


##########################################################################################
# Aykırı Değerlerin Kendilerine Erişmek
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    print(col, grab_outliers(df, col))

##########################################################################################
# 1. Yöntem : Silme

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


for col in num_cols:
    df = remove_outlier(df, col)
##########################################################################################
# 2. Yöntem : Baskılama Yöntemi (re-assignment with thresholds)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))
##########################################################################################

#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor (Bakılabilir)
#############################################


##########################################################################################


#Eksik gözlem analizi

df.isnull().sum()

##########################################################################################
"""Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
değerlere işlemleri uygulayabilirsiniz."""

#############################eksik değerleri outcome kırılımda dolduralım
for col in num_cols:
    df[col]=df[col].fillna(df.groupby("Outcome")[col].transform("mean"))

############################# Bu sütunlarda sıfır olan değerleri NaN ile değiştirme
df[["Glucose", "Insulin", "SkinThickness", "BloodPressure", "BMI"]] =df[["Glucose", "Insulin", "SkinThickness", "BloodPressure", "BMI"]].replace(0, np.nan)

print(df[["Glucose", "Insulin", "SkinThickness", "BloodPressure", "BMI"]].head())

############################# Ortalama ile eksik veri doldurma
df["Glucose"].fillna(df["Glucose"].mean(), inplace=True)
df["BMI"].fillna(df["BMI"].mean(), inplace=True)

############################# Medyan ile eksik veri doldurma
df["BloodPressure"].fillna(df["BloodPressure"].median(), inplace=True)





############################# 1. Değişkenlerin standartlaştırılması
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("Standartlaştırılmış veri:")
print(df_scaled.head())

# 2. KNN'in uygulanması
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df.columns)
print("KNN ile eksik değer doldurulmuş veri:")
print(df.head())

# 3. Standartlaştırılan veriyi orijinal ölçeğe geri döndürme
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
print("Orijinal ölçekli doldurulmuş veri:")
print(df.head())



##########################################################################################

#Eksik gözlem analizi
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df, True)

##########################################################################################

#############################################################################################################
# 1. Yöntem : Hızlıca silmek
#############################################################################################################

df.dropna().shape

##########################################################################################




"""
#############################################################################################################
# Çözüm 2.1: Basit Atama Yöntemleri ile Doldurmak
#############################################################################################################

df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

# df.apply(lambda x: x.fillna(x.mean()), axis=0)

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

df["Embarked"].fillna("missing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

#############################################################################################################
# 2.2. Yöntem :Kategorik Değişken Kırılımında Değer Atama
#############################################################################################################

#Farklı veri setinden örnek olarak konulmuştur.


df.groupby("Sex")["Age"].mean()

df["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

#######################################################################################################################################
# 3. Yöntem : Tahmine Dayalı Atama ile Doldurma
#######################################################################################################################################

#Farklı veri setinden örnek olarak konulmuştur.

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

# değişkenlerin standartlatırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]"""

##########################################################################################

###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "XXXXXX", na_cols)

##########################################################################################

##################################
# KORELASYON
##################################

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

##########################################################################################
############################################################################################################################
# BASE MODEL KURULUMU
############################################################################################################################

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)


rf_model_1 = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model_1.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

############################################################################################################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model_1, X)

############################################################################################################################

#####################################################################################
# ÖZELLİK ÇIKARIMI
#####################################################################################


# Yaş değişkenini kategorik hale getirme
df['Age_group'] = pd.cut(df['Age'],
                         bins=[20, 30, 40, 50, 60, 70, 80, 90],
                         labels=["21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81+"],
                         right=False)
print(df[['Age', 'Age_group']].head())
df.head()

############################################################################################################################


df["GLUCOSE_NEW"]= pd.cut(df["Glucose"], bins=[0, 50, 70, max(df["Glucose"])], labels=["low","normal","high"])

df["BMI_NEW"]=pd.cut(df["BMI"], bins=[18,25,32,max(df["BMI"])], labels=["Normal Weight","Overweight","Obese"])

df.loc[df["Insulin"]<=130,"INSULIN_NEW"]="normal"
df.loc[df["Insulin"]>130, "INSULIN_NEW"]="anormal"

df.groupby("GLUCOSE_NEW")["Outcome"].mean()

############################################################################################################################

##################################
# ENCODING
##################################

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols
############################################################################################################################
# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in cat_cols if df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df,col)
############################################################################################################################
# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols = [col for col in cat_cols if col not in binary_cols]

df = one_hot_encoder(df, cat_cols, drop_first=True)
############################################################################################################################
##################################
# STANDARTLAŞTIRMA
##################################

# minmaxscaling, standartscaling
from sklearn.preprocessing import RobustScaler

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

df[num_cols].head()

#####################################################################################################################

##################################
# MODELLEME
##################################

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")


#####################################################################################################################


lgbm_model= LGBMClassifier(random_state=42, verbosity=-1)
lgbm_model.fit(X_train, y_train)
y_pred_2 =lgbm_model.predict(X_test)
lgbm_accuracy= accuracy_score(y_pred_2, y_test)
lgbm_accuracy





des_model = DecisionTreeClassifier(random_state=42)
des_model.fit(X_train, y_train)
y_pred_2 = des_model.predict(X_test)
decison_sonuc = accuracy_score(y_pred_2, y_test)
decison_sonuc





log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)
y_pred_3 = log_model.predict(X_test)
logistic_sonuc = accuracy_score(y_pred_3, y_test)
logistic_sonuc





xgm_model= XGBClassifier(random_state=42)
xgm_model.fit(X_train, y_train)
y_pred_4 =xgm_model.predict(X_test)
xgb= accuracy_score(y_pred_4, y_test)
xgb






knn_model= KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_6 =knn_model.predict(X_test)
knn= accuracy_score(y_pred_6, y_test) # 0.84
knn






models = [rf_model, lgbm_model, des_model, log_model, xgm_model, knn_model]

best_model = None
best_accuracy = 0

for i, model in enumerate(models, 1):
    model.fit(X_train, y_train)
    y_pred_i = model.predict(X_test)
    accuracy_score_model = accuracy_score(y_pred_i, y_test)

    print(f'Model Name: {type(model).__name__}, Accuracy: {accuracy_score_model}\n')

    print("#" * 80)

    if accuracy_score_model > best_accuracy:
        best_accuracy = accuracy_score_model
        best_model = model

print(f"Best Model {best_model}, Best Accuracy {best_accuracy}")





def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block = True)
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_model, X_train)