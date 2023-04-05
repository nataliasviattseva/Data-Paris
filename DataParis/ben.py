import os
import pandas
import csv
import numpy
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import tools
import seaborn as sns

from .utils import get_graph 


def nettoyage_df() :
    df = pandas.read_csv("/Users/katsuji/Downloads/que-faire-a-paris-.csv", sep= ';', header = 0)
    #df = pd.read_csv("/Users/katsuji/Downloads/que-faire-a-paris-.csv", sep=';', header=0)
    df_propre = df.copy()

    df_propre.isnull().values.any()

    #gerer les valeurs vides
    df_propre.dropna(how= 'all', inplace= True)

    #supprimer ce qui n'est pas dans Paris
    indexVille = df_propre[ df_propre["address_city"] != "Paris" ].index
    df_propre.drop(indexVille, inplace= True)

    #df_propre.drop(columns = ["url",  ],  inplace = True)
    df_propre.drop(df_propre.columns.difference(['id', 'title', 'date_start', 'date_end', 'tags', 'address_name', 'address_street', 'address_zipcode', 'lat_lon', 'price_type']), axis=1, inplace=True)

    return(df_propre)
    #print(df_propre.isnull().values.any())

    #print(df_propre)

def creation_df_prix(df_propre) :
    arrondissement = [75001, 75002, 75003, 75004, 75005, 75006, 75007, 75008, 75009, 75010, 75011, 75012, 75013, 75014, 75015, 75016, 75017, 75018, 75019, 75020]

    df_price_rip = df_propre["price_type"]
    df_arr = df_propre["address_zipcode"]

    df_price = df_price_rip.replace("gratuit sous condition", "gratuit")
    print(df_price.value_counts())

    df_arrondissement = pandas.concat([df_price, df_arr], axis = 1)

    #df_arrondissement = df_arrondissement.reset_index()

    #print(df_arrondissement.value_counts())
    return(df_arrondissement)

def creation_hist_q2(df_arrondissement) :

    #df_arrondissement.hist(column= "address_zipcode", by = "price_type")
    # plt.show()

    #sns.countplot(x = "address_zipcode", hue ="price_type", data = df_arrondissement)
    threshold = 10
    zip_counts = df_arrondissement ["address_zipcode"].value_counts()
    valid_zips = zip_counts[zip_counts >= threshold].index
    df_valid = df_arrondissement[df_arrondissement["address_zipcode"].isin(valid_zips)]
    print(df_valid.value_counts())

    plt.switch_backend('AGG')       # added Katsuji

    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20
    mpl.rcParams['legend.fontsize'] = 20
    #plt.figure(figsize=(30,30))
    plt.figure(figsize=(10, 8))     # modified Katsuji

    fig = sns.countplot(x="address_zipcode", hue="price_type", data = df_valid, order = ["75001", "75002", "75003", "75004", "75005", "75006", "75007", "75008", "75009", "75010", "75011", "75012", "75013", "75014", "75015", "75016", "75017", "75018", "75019", "75020"])
    #fig.set(title = " Nombre d'évènements gratuits ou payants pas arrondissement ")
    plt.title(" Nombre d'évènements gratuits ou payants par arrondissement ", fontsize = 20)
    
    graph = get_graph()
    #plt.show()

    return graph