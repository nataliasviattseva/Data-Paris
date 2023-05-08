from django.shortcuts import render
from django.http import HttpResponse

import os
import pandas as pd
import csv
import numpy
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import seaborn as sns


from io import BytesIO
import base64


def nettoyage_df():
    # lecture de la base
    #df = pd.read_csv("static\data\que-faire-a-paris-.csv", sep=';', header=0)
    df = pd.read_csv("static/data/que-faire-a-paris-.csv", sep=';', header=0)
    df_propre = df.copy()

    df_propre.isnull().values.any()

    # gerer les valeurs vides
    df_propre.dropna(how='all', inplace=True)

    # supprimer ce qui n'est pas dans Paris
    indexVille = df_propre[df_propre["address_city"] != "Paris"].index
    df_propre.drop(indexVille, inplace=True)

    columns_to_keep = ['id', 'title', 'date_start', 'date_end', 'tags', 'address_name', 'address_street',
                       'address_zipcode', 'lat_lon', 'price_type']
    df_propre.drop(df.columns.difference(columns_to_keep), axis=1, inplace=True)

    return df_propre


def home(request):
    return render(request, "home.html")

def question1(request):
    df_propre = nettoyage_df()

    # creation  de la base pour arrondissement et prix
    df_price_rip = df_propre["price_type"]
    df_arr = df_propre["address_zipcode"]

    df_price = df_price_rip.replace("gratuit sous condition", "gratuit")
    print(df_price.value_counts())

    df_arrondissement = pd.concat([df_price, df_arr], axis=1)

    # filtre pour enlever les valeurs trop peu importantes
    threshold = 10
    zip_counts = df_arrondissement["address_zipcode"].value_counts()
    valid_zips = zip_counts[zip_counts >= threshold].index
    df_valid = df_arrondissement[df_arrondissement["address_zipcode"].isin(valid_zips)]
    print(df_valid.value_counts())

    # creation du df free pour l'affichage
    df_show_free = df_valid.loc[(df_valid["price_type"] == "gratuit") & (df_valid["address_zipcode"])]
    df_show_free = df_show_free.value_counts()
    df_show_free = df_show_free.to_frame()
    df_show_free = df_show_free.sort_values(by=["address_zipcode", "price_type"], ascending=True)
    # df_show_free = df_show_free.transpose()
    df_html_free = df_show_free.to_html()

    # creation du df payant pour l'affichage
    df_show_pay = df_valid.loc[(df_valid["price_type"] == "payant") & (df_valid["address_zipcode"])]
    df_show_pay = df_show_pay.value_counts()
    df_show_pay = df_show_pay.to_frame()
    df_show_pay = df_show_pay.sort_values(by=["address_zipcode", "price_type"], ascending=True)
    # df_show_pay = df_show_pay.transpose()
    df_html_pay = df_show_pay.to_html()

    # parametres du countplot
    plt.switch_backend('AGG')  # added Katsuji
    
    mpl.rcParams['axes.labelsize'] = 7
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7
    mpl.rcParams['legend.fontsize'] = 7
    # plt.figure(figsize=(30,30))
    plt.figure(figsize=(6.5, 5))  # modified Katsuji

    # creation du countplot
    fig = sns.countplot(x="address_zipcode", hue="price_type", data=df_valid,
                        order=["75001", "75002", "75003", "75004", "75005", "75006", "75007", "75008", "75009", "75010",
                               "75011", "75012", "75013", "75014", "75015", "75016", "75017", "75018", "75019",
                               "75020"])
    # fig.set(title = " Nombre d'évènements gratuits ou payants pas arrondissement ")
    plt.title(" Nombre d'évènements gratuits ou payants par arrondissement ", fontsize=11)
    plt.xticks(rotation=45)


    countplot_file = "static/graph_images/q1_countplot.png"
    fig.get_figure().savefig(countplot_file)

    return render(request, "question1.html", {"graph": countplot_file, "df_show": df_html_free, "df_show1": df_html_pay})


#-------------------------------------

def question2(request):
    df_propre = nettoyage_df()

    # Garder le premier mot sur serie de tags
    df_propre['tags'] = df_propre['tags'].str.split(';').str[0]

    liste_types_d_evenements_pas_doublane = []
    liste_types_d_evenements_doublane = []
    dict = {}

    for i in df_propre['tags']:
        liste_types_d_evenements_doublane.append(i)
        if i not in liste_types_d_evenements_pas_doublane:
            liste_types_d_evenements_pas_doublane.append(i)

    for i in liste_types_d_evenements_doublane:
        if i in dict:
            dict[i] += 1
        else:
            dict[i] = 1


    
    
    df_tags = pd.DataFrame(list(dict.items()), columns=['Évènements', 'Occurrences'])
    

    # Retirer les Evenements marqués comme "NaN"
    # Filter NAN Data Selection column of strings by not (~) operator is used to negate the statement.
    df_tags = df_tags[~pd.isnull(df_tags['Évènements'])]

    
    html_df_tags = df_tags.to_html()
    # html_df_tags = df_tags.transpose().to_html()


    df_filtered = df_tags[df_tags['Occurrences'] > 33]

    # close all open figures and set the Matplotlib backend. AGG for png images
    plt.switch_backend('AGG')

    # Create the pie chart
    
    fig1, ax = plt.subplots()
    # fig1, ax = plt.subplots(figsize = (7, 5))

    ax.pie(df_filtered['Occurrences'], labels=df_filtered['Évènements'], textprops={'fontsize': 10}, rotatelabels=15, startangle=90)
    # ax.pie(df_filtered['Occurrences'], labels=df_filtered['Évènement'], autopct='%1.1f%%')

    # Add a title
    # ax.set_title('Pie Chart for Évènements')
    
    
    pie_graph_file = "static/graph_images/q2_pie.png"
    plt.savefig(pie_graph_file)
    plt.close(fig1)

    # fig2, sns = plt.subplots()
    
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8.5
    mpl.rcParams['ytick.labelsize'] = 8.5
    mpl.rcParams['legend.fontsize'] = 10
    plt.figure(figsize=(6, 5)) 

    sns_plot = sns.barplot(x='Occurrences', y='Évènements', data=df_filtered)
    sns_plot.set_title('Barplot for Évènements')
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=60)
    barplot_file = "static/graph_images/q2_barplot.png"
    sns_plot.get_figure().savefig(barplot_file)

    return render(request, 'question2.html', {'html_df_tags': html_df_tags,
                                              'pie_graph_file': pie_graph_file,
                                              'barplot_file': barplot_file})

def question3(request):
    df_propre = nettoyage_df()

    #--------------------------------------
    # added on 17April2023 by Katsuji
    #
    df_propre = df_propre.reset_index()

    print("------------ df_propre ")
    print(df_propre.isnull().sum())

    df_tmp_2 = convert_to_datetime(df_propre)

    print("------------ df_tmp_2 ")
    print(df_tmp_2.isnull().sum())

    # calculate the duration of event and add it to DataFrame.
    #   column added : "duration(days)"
    df_tmp_3 = calculate_duration(df_tmp_2)

    #print("++++++++++  df_tmp_3 ")
    #print(df_tmp_3.dtypes)
    #print(df_tmp_3.groupby('saison').count())


    # replace all "NaT" (DateTimeFormat) data with "0"
    # dtype will be changed to "object" and cannot calculate anymore but, anytime it is possible to be change it back to
    # DateTime type by to_datatime() and can calculate
    #
    #print(f"df_tmp_3.isnull().sum()={df_tmp_3.isnull().sum()}")
    #df_tmp_3.fillna("0", inplace=True)
    # df_tmp_3.isnull().sum()
    df_propre_v2 = df_tmp_3.copy()

    #print("++++++++++  df_propre_v2 ")
    #print(df_propre_v2.groupby('saison').count())

    # df_propre_v2.to_csv('./tmp.csv', header=True, index=False)

    # Free and non-free events by season
    condition_1 = 'price_type'
    price_type_lt = ['gratuit', 'payant']
    condition_3 = 'saison'
    #graph = construct_graph_bar(df_propre_v2, condition_1, price_type_lt, condition_3)
    graph, df_tmp = construct_graph_bar(df_propre_v2, condition_1, price_type_lt, condition_3)  # correction 17April2023
    #print(f"---------  df_tmp ------")
    #print(df_tmp)
    
    table = construct_table_img(df_tmp) # correction 17April2023

    #return render(request, 'question3.html', {'graph': graph})
    #return render(request, 'main/Q3.html', {'graph': graph, 'table': table}) # correction 17April2023
    return render(request, 'question3.html', {'graph': graph, 'table': table})

def convert_to_datetime(df_in):
    sr1 = df_in['date_start']
    sr1 = pd.to_datetime(sr1, errors="coerce")
    sr1 = pd.to_datetime(sr1, utc=True)
    sr1.name = "date_start(DateTimeFormat)"

    sr2 = df_in['date_end']
    sr2 = pd.to_datetime(sr2, errors="coerce")
    sr2 = pd.to_datetime(sr2, utc=True)
    sr2.name = "date_end(DateTimeFormat)"

    df_out = pd.concat([df_in, sr1, sr2], axis=1)

    # convert to offset aware
    timezone = pytz.timezone("UTC")

    d1 = datetime.strptime('2023-03-21', '%Y-%m-%d')
    d1_aware = timezone.localize(d1)

    d2 = datetime.strptime('2023-06-21', '%Y-%m-%d')
    d2_aware = timezone.localize(d2)

    d3 = datetime.strptime('2023-09-21', '%Y-%m-%d')
    d3_aware = timezone.localize(d3)

    d4 = datetime.strptime('2023-12-21', '%Y-%m-%d')
    d4_aware = timezone.localize(d4)

    #mid = sr1 + (sr2 - sr1)
    mid = sr1 + (sr2 - sr1) / 2 # correction 17April2023

    saison = []

    '''
    for i in range(df_out.shape[0]):
        if pd.isnull(mid[i]):
            saison.append("None")
        elif mid[i] >= d1_aware and mid[i] < d2_aware:
            saison.append("printemps")
        elif mid[i] >= d2_aware and mid[i] < d3_aware:
            saison.append("ete")
        elif mid[i] >= d3_aware and mid[i] < d4_aware:
            saison.append("automne")
        else:
            saison.append("hiver")
    '''
    
    # correction 17April2023
    for i in range(df_out.shape[0]):
        if pd.isnull(mid.iloc[i]):
            saison.append("None")
        elif mid.iloc[i] >= d1_aware and mid.iloc[i] < d2_aware:
            saison.append("printemps")
        elif mid.iloc[i] >= d2_aware and mid.iloc[i] < d3_aware:
            saison.append("ete")
        elif mid.iloc[i] >= d3_aware and mid.iloc[i] < d4_aware:
            saison.append("automne")
        else:
            saison.append("hiver")

    tmp_saison = pd.Series(saison)
    tmp_saison.name = "saison"
    df_out = pd.concat([df_out, tmp_saison], axis=1)

    #print(f"----------- df_out ")
    #print(df_out.groupby('saison').count())

    return df_out


def calculate_duration(df_in):
    sr1 = df_in["date_start(DateTimeFormat)"]

    sr2 = df_in["date_end(DateTimeFormat)"]

    sr3 = sr2 - sr1
    sr3.name = "duration(days)"
    df_out = pd.concat([df_in, sr3], axis=1)

    return df_out


def construct_graph_bar(df, condition_1, condition_lt, condition_3):
    # from https://www.youtube.com/watch?v=jrT6NiM46jk&t=185s

    # print(df.columns)
    # remove rows with 'saison' value = "None"
    df = df[df['saison'] != "None"]
    #df = df[df['saison'] != "0"]

    #print(f"----------- df ")
    #print(df.groupby('saison').count())

    #
    #   drop rows with price_type = 'gratuit sous condition'
    #
    i = 0
    for condition_tmp in condition_lt:
        df_tmp = df[df[condition_1] == condition_tmp]
        df_tmp = df_tmp.groupby(condition_3).count()
        df_tmp = pd.DataFrame(df_tmp['id'])
        df_tmp.reset_index(inplace=True)
        df_tmp['type'] = condition_tmp
        if i == 0:
            df_tmp_2 = df_tmp.copy()
            i += 1
        else:
            df_tmp_2 = pd.concat([df_tmp_2, df_tmp], axis=0)
    df_tmp_2 = df_tmp_2.reset_index()
    df_tmp_2 = df_tmp_2.drop(columns='index')
    df_tmp_2 = df_tmp_2.reindex(columns=["type", "saison", "id"])
    df_tmp = df_tmp_2.copy()

    new_name = "Nombre d'évènements"
    df_tmp.rename(columns={"id": new_name}, inplace=True)

    sns.set()

    plt.switch_backend('AGG')
    plt.figure(figsize=(6, 5), facecolor="w")
    sns.barplot(data=df_tmp, x="type", y=new_name, hue="saison", hue_order=["printemps", "ete", "automne", "hiver"])
    title = "Nombre d'évènements par type de paiement"
    plt.title(title)

    plt.legend(loc='upper right')

    print("************** ploting ***************")
    # plt.show()

    # buf = io.BytesIO()
    # plt.savefig(buf, format='svg', bbox_inches='tight')
    # s = buf.getvalue()
    # buf.close()

    # output graph plotted by seaborn/matplotlib as image data
    graph = get_graph()

    #return graph
    # return render(request, 'DataParis/home.html', {"chart":chart})
    return graph, df_tmp    # correction 17April2023


# from deleted file utils.py
def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


# Natalia: functions below from deleted file ben.py, I did'n find where they are called
def creation_df_prix(df_propre):
    arrondissement = [75001, 75002, 75003, 75004, 75005, 75006, 75007, 75008, 75009, 75010, 75011, 75012, 75013, 75014,
                      75015, 75016, 75017, 75018, 75019, 75020]

    df_price_rip = df_propre["price_type"]
    df_arr = df_propre["address_zipcode"]

    df_price = df_price_rip.replace("gratuit sous condition", "gratuit")
    print(df_price.value_counts())

    df_arrondissement = pd.concat([df_price, df_arr], axis=1)

    # df_arrondissement = df_arrondissement.reset_index()

    # print(df_arrondissement.value_counts())
    return (df_arrondissement)


def creation_hist_q2(df_arrondissement):
    # df_arrondissement.hist(column= "address_zipcode", by = "price_type")
    # plt.show()

    # sns.countplot(x = "address_zipcode", hue ="price_type", data = df_arrondissement)
    threshold = 10
    zip_counts = df_arrondissement["address_zipcode"].value_counts()
    valid_zips = zip_counts[zip_counts >= threshold].index
    df_valid = df_arrondissement[df_arrondissement["address_zipcode"].isin(valid_zips)]
    print(df_valid.value_counts())

    plt.switch_backend('AGG')  # added Katsuji

    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    # plt.figure(figsize=(30,30))
    plt.figure(figsize=(7, 5))  # modified Katsuji

    fig = sns.countplot(x="address_zipcode", hue="price_type", data=df_valid,
                        order=["75001", "75002", "75003", "75004", "75005", "75006", "75007", "75008", "75009", "75010",
                               "75011", "75012", "75013", "75014", "75015", "75016", "75017", "75018", "75019",
                               "75020"])
    # fig.set(title = " Nombre d'évènements gratuits ou payants pas arrondissement ")
    plt.title(" Nombre d'évènements gratuits ou payants par arrondissement ", fontsize=14)

    graph = get_graph()
    # plt.show()

    return graph

#
#   added on 17April2023 by Katsuji
#
def construct_table_img(df):

    fig, ax = plt.subplots(figsize=(6,5))
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=df.values,
         colLabels=df.columns,
         colColours =["gold"] * 3,
         loc='center',
         bbox=[0,0,1,1])
    '''
    buffer = BytesIO()
    #plt.savefig('table.png')
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    table = base64.b64encode(image_png)
    table = table.decode('utf-8')
    buffer.close()
    '''
    table = get_table()

    #table = ""

    return table

#
#   added on 17April2023 by Katsuji
#
def get_table():

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    table = base64.b64encode(image_png)
    table = table.decode('utf-8')
    buffer.close()
    return table


def map (request):
    return render(request, "map.html")