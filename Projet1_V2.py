import fonctions as fc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





donnee = pd.read_csv("post-32566-EIVP_KM.csv", index_col=0, sep=';')



"""print(donnee.head())""" #afficher les 5 premiers éléments de la table
"""print(donnee.tail()""" #afficher les 5 derniers éléments de la table
"""donnee.info()""" #Informations diverses sur les variables et la table
"""print(donnee.columns)"""# Noms des colonnes

"""donnee.rename(columns={'NOM_Column': 'NEW_NOM'}, inplace=True)""" #Renommer le nom d'une colonne
print(donnee.isnull().sum()) #Compte le nombre de valeurs nulles pour chaque variable de la table
print(donnee.corr()) #indice de corrélation entre les différentes variables de la table 
print(donnee)
"""print(donnee.loc[1, ['co2']])"""



temperature=donnee['temp']
humidite=donnee['humidity']
carbone=donnee['co2']
luminosite=donnee['lum']
bruit=donnee['noise']
date=donnee['sent_at']


# print(donnee)


# Date :

def separer_date_heure(date_elementaire):
    if len(date_elementaire)>=19:
        d=date_elementaire[0:10]
        h=date_elementaire[11:19]
    else:
        d=date_elementaire[0:10]
        h=[]
    return d,h


def separer_date(date_elementaire):         # a-b-c ...
    d,h=separer_date_heure(date_elementaire)
    a=''
    b=''
    for x in d:
        if x!='-':
            a+=x
    if h==[]:
        b=0
    else:
        for y in h:
            if y!=':':
                b+=y
    return int(a),int(b)                          # [abc]

def interQ_detect(document, column):
       q1 = document[column].quantile(0.25)
       q3 = document[column].quantile(0.75)
       iqr = q3-q1 #distance Interquartile 
       limite_inf  = q1-1.5*iqr
       limite_sup = q3+1.5*iqr
       n=len(document.index)
       l=[]
       L=[]
       index=[]
       for i in range(n):
           if document.loc[document.index[i],column]<= limite_inf or document.loc[document.index[i],column]>= limite_sup :
               """print(document.loc[i,[column]])"""
               index.append(i)
               l.append(document.loc[i,[column]])
               L.append(document.iloc[i,6])
       """document1 = document.loc[(document[column] > limite_inf) & (document[column] < limite_sup)] #document après suppression des anomalies 
       return (document1)"""
       return(index,l,L)


def Afficher_journee_avec_anomalie(document,col,column,start_date,end_date):
    Afficher_courbe(col,start_date,end_date)
    k,l,L=interQ_detect(document,column)
    M=[]
    n=len(L)
    """for j in range (n):
        M.append(trouver_first_date(L[j]))"""
    plt.scatter(l,L,color='red',label='anomalie')
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
    plt.title('Evolution col')
    plt.show()
    return None

def separer_date_liste(date_liste):
    L=[]
    H=[]
    n=len(date_liste)
    for k in range(n):
        l,h=separer_date(date_liste[k])
        L.append(l)
        H.append(h)
    return L,H




# Agrandi donnee

date2,heure2=separer_date_liste(date)
date2
donnee['date2']=date2
donnee['heure2']=heure2
donnee.sort_values(by=['date2','heure2'],ascending=[True,True])






def trouver_first_date(date_elementaire):
    d,h=separer_date(date_elementaire)
    index=0
    while d>date2[index]:
        index+=1
        if index==len(date2):
            break
    return index


def trouver_last_date(date_elementaire):
    d,h=separer_date(date_elementaire)
    index=0
    while d>=date2[index]:
        index+=1
        if index==len(date2):
            break
    return index



#_______________________________________________________________________________




# Courbe :

def Afficher_carbone(start_date,end_date):
    i=trouver_first_date(start_date)
    j=trouver_last_date(end_date)
    x = date.tolist()[i:j]
    y = carbone.tolist()[i:j]
    #plt.subplot(2, 1, 2)

    plt.plot(x, y, '.-')
    plt.xlabel('Temps')
    plt.ylabel('Carbone')
    
    plt.title('Evolution carbone')
    
    plt.show()
    return None


def Afficher_temperature(start_date,end_date):
    i=trouver_first_date(start_date)
    j=trouver_last_date(end_date)
    x = date.tolist()[i:j]
    y = temperature.tolist()[i:j]

    #plt.subplot(2, 1, 2)

    plt.plot(x, y, '.-')
    plt.xlabel('Temps')
    plt.ylabel('Température')
    plt.title('Evolution température')
    
    plt.show()
    return None


def Afficher_humidite(start_date,end_date):
    i=trouver_first_date(start_date)
    j=trouver_last_date(end_date)
    x = date.tolist()[i:j]
    y = humidite.tolist()[i:j]

    #plt.subplot(2, 1, 2)

    plt.plot(x, y, '.-')
    plt.xlabel('Temps')
    plt.ylabel('Humidité')
    plt.title('Evolution humidité')
    
    plt.show()
    return None


def Afficher_courbe(col,start_date,end_date):
    i=trouver_first_date(start_date)
    j=trouver_last_date(end_date)
    x = date.tolist()[i:j]
    y = col.tolist()[i:j]

    #plt.subplot(2, 1, 2)

    plt.plot(x, y, '.-')
    plt.xlabel('Temps')
    plt.ylabel('col')
    plt.title('Evolution col')
    
    """plt.show()"""
    return None


#____________________________________________________________


# Statistique : Moyenne, mediane, variance :

def moyenne(l):
    moyen=0
    n=len(l)
    for x in l:
        moyen+=x
    moyen=moyen/n
    return moyen

def mediane(l):
    l=quickSort(l)
    mediane=0
    n=len(l)
    if n%2==0:
        a=l[n//2]
        b=l[(n//2)+1]
        madiane=moyenne((a,b))
    else:
        mediane=l[n//2]
    return mediane
        

def variance(l):        # E(x**2)-E(x)**2
    Lcarree=[]
    var=0
    moyen=moyenne(l)
    for x in l:
        Lcarree.append(x**2)
    moyen_carree=moyenne(Lcarree)
    var=moyen_carree-moyen**2
    return var


def covariance(X,Y):
    n=len(X)
    m=len(Y)
    if n!=m:
        return None
    S=0
    x=moyenne(X)
    y=moyenne(Y)
    for i in range(n):
        S+=X[i]*Y[i]
    S=S/n
    covar=S-x*y
    return covar


def ecart_type(l):
    var=variance(l)
    return var**(1/2)



#Anomalie detection

    
# Anomalie :




# Dérive :

def derive(L,T):                # Il faut que len(T)=len(L)>0
    l=[]
    n=len(L)
    for k in range(n-1):
        x=(float(L[k+1])-float(L[k]))/(float(T[k+1])-float(T[k]))
        l.append(x)
    return l


def vitesse(L,T):
    return derive(L,T)
    
    
def acceleration(L,T):
    n=len(T)-1
    return vitesse(vitesse(L,T),T[0:n])


# Heure :

def Heure(H):
    heure=[]
    for h in H:
        h=h/10000
        hint=int(h)                 # heure entière
        h2=(h-hint)*100
        hmin=int(h2)                # minute
        h3=(h2-hmin)*100
        hs=int(h3)                  # seconde
        h=hint+(hmin/60)+(hs/3600)
        heure.append(h)
    return heure


heure3=Heure(donnee.heure2)
donnee['heure3']=heure3


# Définition de "e", la plus grande variatiion de l'acceleration possible (sans anomalie)

e=1000


def is_anomalie2(acc,id):          # acc=acceleration(col,date2)
    if abs(acc[id])>e:
        return True
    return False


def anomalie_list(col,T):
    index=[]
    value=[]
    acc=acceleration(col,T)
    for i in range(len(col)-2):
        if is_anomalie2(acc,i):
            index.append(i)
            value.append(col[i])
    return index,value


def anomalie_list_plusieurs_jour(col,T,date):
    index=[]
    value=[]
    D=[]
    acc=acceleration(col,T)
    for i in range(len(col)-2):
        if is_anomalie2(acc,i):
            index.append(i)
            value.append(col[i])
            D.append(date[i])
    return index,value,D


def anomalie_list_une_journee(col,T,heure):
    index=[]
    value=[]
    h=[]
    acc=acceleration(col,T)
    for i in range(len(col)-2):
        if is_anomalie2(acc,i):
            index.append(i)
            value.append(col[i])
            h.append(heure[i])
    return index,value,h

