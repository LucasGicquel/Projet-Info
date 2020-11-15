import fonctions as fc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





document = pd.read_csv("post-32566-EIVP_KM.csv", index_col=0, sep=';')

temperature=document['temp']
hum=document['humidity']
carbone=document['co2']
date=document['sent_at']

"""print(document.head())""" #afficher les 5 premiers éléments de la table
"""print(document.tail()""" #afficher les 5 derniers éléments de la table
"""document.info()""" #Informations diverses sur les variables et la table
"""print(document.columns)"""# Noms des colonnes

"""document.rename(columns={'NOM_Column': 'NEW_NOM'}, inplace=True)""" #Renommer le nom d'une colonne
print(document.isnull().sum()) #Compte le nombre de valeurs nulles pour chaque variable de la table
print(document.corr()) #indice de corrélation entre les différentes variables de la table 
print(document)
"""print(document.loc[1, ['co2']])"""

#_______________________________________________________________________________


# Date :

def separer_date(date_elementaire):         # a-b-c ...
    l=[]
    a=''
    for x in date_elementaire:
        if x=='-':
            l.append(int(a))
            a=''
        if x==' ':
            l.append(int(a))
            break

        if x!='-':
            a+=x

    return l                                # [a,b,c]


def separer_date_liste(date_liste):
    L=[]
    n=len(date_liste)
    for k in range(n):
        l=separer_date(date_liste[k])
        L.append(l)
    return L


def separer_date_liste_num(date_liste):
    L=[]
    n=len(date_liste)
    for k in range(n):
        l=separer_date(date_liste[k])
        num=[k]
        num.append(l)
        L.append(num)
    return L


def trier_date(L):
    return L



#_______________________________________________________________________________


# Courbe :

def Afficher_carbone(start_date='2019-01-01',end_date='2019-02-01'):
    x = carbone
    y = date

    #plt.subplot(2, 1, 2)

    plt.plot(x, y, '.-')
    plt.xlabel('Carbone')
    plt.ylabel('Temps')
    plt.title('Evolution carbone')
    
    plt.show()


def Afficher_temperature(start_date='2019-01-01',end_date='2019-02-01'):
    x = temperature
    y = date

    #plt.subplot(2, 1, 2)

    plt.plot(x, y, '.-')
    plt.xlabel('Température')
    plt.ylabel('Temps')
    plt.title('Evolution température')
    
    plt.show()


def Afficher_humidite(start_date='2019-01-01',end_date='2019-02-01'):
    x = hum
    y = date

    #plt.subplot(2, 1, 2)

    plt.plot(x, y, '.-')
    plt.xlabel('humidité')
    plt.ylabel('Temps')
    plt.title('Evolution humidité')
    
    plt.show()
    

#Méthode basée sur l'interquartile

def interQ_detect(document, column):
       q1 = document[column].quantile(0.25)
       q3 = document[column].quantile(0.75)
       iqr = q3-q1 #distance Interquartile 
       limite_inf  = q1-1.5*iqr
       limite_sup = q3+1.5*iqr
       n=len(document.index)
       for i in range(n):
           if document.loc[document.index[i],column]<= limite_inf or document.loc[document.index[i],column]>= limite_sup :
               print(document.loc[i,[column]])
       document1 = document.loc[(document[column] > limite_inf) & (document[column] < limite_sup)] #document après suppression des anomalies 
       return (document1)


#Méthode basée sur l'écart-type

def Zscore(document,column,num_ligne):
    return((document.loc[document.index[num_ligne],column]-fc.moyenne(document[column]))/fc.ecart_type(document[column]))

def ET_detect(document,column,nb_EQ):
    n=len(document.index)
    for i in range(n):
        if abs(Zscore(document, column, i))>nb_EQ:
            print(document.loc[i,[column]])
    """document1 = document.loc[(abs(Zscore(document, column, i))<=nb_EQ)]
    return(document1)"""
