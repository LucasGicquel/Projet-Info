import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import datetime


doc = pd.read_csv("post-32566-EIVP_KM.csv", index_col=0, sep=';')
doc["date"]=pd.to_datetime(doc["sent_at"])           #Transformation au format datetime
doc["day"] = doc["date"].dt.strftime("%Y-%m-%d")             #Création d'une colonne date du jour
doc["hour"] = doc["date"].dt.strftime("%H:%M:%S")            #Création d'une colonne heure du jour
doc["date"]=doc["day"]+' '+doc["hour"]
doc["date"]=pd.to_datetime(doc["date"])             #Suppression du fuseau horaire
doc=doc.sort_values(by=['date'], ascending=True)            #Table triée selon les dates (jour puis heure)


def display(variable,start_date,end_date):
    doc1 = doc.set_index(['date'])            #réindexation par la date

    doc1=doc1.loc[start_date : end_date]         #Création d'un nouveau document filtré par les dates de début et de fin choisies

#Choix de la variable
    
    if variable=="temperature":
        var=doc1["temp"]
    elif variable=="luminosite":
        var=doc1["lum"]
    elif variable=="bruit":
        var=doc1["noise"]
    elif variable=="humidite":
        var=doc1["hum"]
    elif variable=="carbone":
        var=doc1["co2"]
    elif variable=="humidex":
        return(humidex(start_date,end_date))
#Affichage

    temps=doc1["sent_at"]
    plt.plot(temps,var,'-',color='black', label=variable)
    plt.xlabel('Temps')
    plt.ylabel(variable)
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
    plt.title('Evolution '+variable+' en fonction du temps')
    plt.show()
    
    return(None)





# Tri :

def swap(a,i,b,j,L):
    L[i]=b
    L[j]=a


    # Tri rapide

def partitionner(T, first, last, pivot):
    n=len(T)
    
    swap(T[pivot],pivot,T[last],last,T)         # échange le pivot avec le dernier du tableau , le pivot devient le dernier du tableau
    j = first
    for i in range(first,last):                 # la boucle se termine quand i = (dernier-1).
        if T[i] <= T[last]:
            swap(T[i],i,T[j],j,T)
            j = j + 1
    swap(T[last],last,T[j],j,T)
    return j


def choix_pivot(T,first, last):
    return (first+last)//2


def tri_rapide(T,first,last):
    if first<last:
        pivot = choix_pivot(T, first, last)
        pivot = partitionner(T, first, last, pivot)
        tri_rapide(T, first, pivot-1)
        tri_rapide(T, pivot+1, last)


def quickSort(T):
    
    tri_rapide(T,0,len(T)-1)
    
    return T





#Statistiques

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
        mediane=moyenne((a,b))
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
    var=moyen_carree-moyen
    return var


def ecart_type(l):
    var=variance(l)
    return var**(1/2)


def Max(l):
    maxi = l[0]
    longueur=len(l)
    for i in range(longueur):
        if l[i] >= maxi:
            maxi = l[i]
            
    return (maxi)


def Min(l):
    mini = l[0]
    longueur=len(l)
    for i in range(longueur):
        if l[i] <= mini:
            mini = l[i]
    return (mini)









def displayStat(variable,start_date,end_date):
    doc1 = doc.set_index(['date'])            #réindexation par la date

    doc1=doc1.loc[start_date : end_date]         #Création d'un nouveau document filtré par les dates de début et de fin choisies

#Choix de la variable
    
    if variable=="temperature":
        var=doc1["temp"]
    elif variable=="luminosite":
        var=doc1["lum"]
    elif variable=="bruit":
        var=doc1["noise"]
    elif variable=="humidite":
        var=doc1["hum"]
    elif variable=="carbone":
        var=doc1["co2"]

    temps=doc1["sent_at"]
#Determination des valeurs statistiques

    mini=Min(var)
    maxi=Max(var)
    moy=moyenne(var)
    """med=mediane(var)"""
    et=ecart_type(var)
    v=variance(var)

#Affichage

    n=len(temps)
    Mini=n*[mini]
    Maxi=n*[maxi]
    Moy=n*[moy]
    """Med=n*[med]"""
    plt.plot(temps,var,'-',color='black', label=variable+"    Ecart-type= "+str(et))
    plt.xlabel('Temps')
    plt.ylabel(variable)
    plt.plot(temps,Mini,'-',color='blue', label="minimum"+"    Variance= "+str(v))
    plt.plot(temps,Maxi,'-',color='red', label="maximum")
    plt.plot(temps,Moy,'-',color='green', label="moyenne")
    """plt.plot(temps,Med,'-',color='yellow', label="mediane")"""
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
    plt.title('Evolution '+variable+' en fonction du temps')
    plt.show()
    
    return(None)




#Indice de covariance

def covariance(X,Y):
    n=len(X)
    m=len(Y)
    if n!=m:
        return None
    S=0
    x=moyenne(X)
    y=moyenne(Y)
    for i in range(n):
        S+=(X[i]-x)*(Y[i]-y)
    S=S/n
    covar=S/(ecart_type(X)*ecart_type(Y))
    return covar






def corrélation(variable1,variable2,start_date,end_date):

    var1=[]
    var2=[]
    v1=[]
    v2=[]
#Calcul indice de corrélation
    
    if variable1=="temperature":
        v1=doc["temp"]
    elif variable1=="luminosite":
        v1=doc["lum"]
    elif variable1=="bruit":
        v1=doc["noise"]
    elif variable1=="humidite":
        v1=doc["hum"]
    elif variable1=="carbone":
        v1=doc["co2"]

    
        
    if variable2=="temperature":
        v2=doc["temp"]
    elif variable2=="luminosite":
        v2=doc["lum"]
    elif variable2=="bruit":
        v2=doc["noise"]
    elif variable2=="humidite":
        v2=doc["hum"]
    elif variable2=="carbone":
        v2=doc["co2"]


    v1,v2=v1.values.tolist(),v2.values.tolist()
    icorr=covariance(v1,v2)



    doc1 = doc.set_index(['date'])            #réindexation par la date

    doc1=doc1.loc[start_date : end_date]         #Création d'un nouveau document filtré par les dates de début et de fin choisies

#Choix des variables

    #1
    
    if variable1=="temperature":
        var1=doc1["temp"]
    elif variable1=="luminosite":
        var1=doc1["lum"]
    elif variable1=="bruit":
        var1=doc1["noise"]
    elif variable1=="humidite":
        var1=doc1["hum"]
    elif variable1=="carbone":
        var1=doc1["co2"]

    #2
        
    if variable2=="temperature":
        var2=doc1["temp"]
    elif variable2=="luminosite":
        var2=doc1["lum"]
    elif variable2=="bruit":
        var2=doc1["noise"]
    elif variable2=="humidite":
        var2=doc1["hum"]
    elif variable2=="carbone":
        var2=doc1["co2"]



        
#Affichage

    temps=doc1["sent_at"]
    plt.plot(temps,var1,'-',color='black', label=variable1+" Indice de corrélation= "+str(icorr))
    plt.plot(temps,var2,'-',color='blue', label=variable2)
    plt.xlabel('Temps')
    plt.ylabel(variable1+" et "+variable2)
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
    plt.title('Evolution '+variable1+" et "+variable2+' en fonction du temps')
    plt.show()
    return(None)





# Fonctions de base


# Operation :

def exp_rapide(n,x=np.e):
    if n==0:
        return 1
    if n==1:
        return x
    if n%2==0:
        return exp_rapide(n//2,x)*exp_rapide(n//2,x)
    else:
        return exp_rapide(n//2,x)*exp_rapide((n//2)+1,x)
    

def exp(x):
    n=int(x)
    d=x-int(x)
    e_entier=exp_rapide(n)
    e_fract=np.e**d
    return e_entier*e_fract



# Indice humidex :



def humidex_unite(Tair,Trosee):
    e=exp(5417.7530*((1/273.16)-(1/(273.15+Trosee))))
    H=Tair+0.5555*(6.11*e-10)
    return H


def humidex(start_date,end_date):
    doc1 = doc.set_index(['date'])            #réindexation par la date

    doc1=doc1.loc[start_date : end_date]         #Création d'un nouveau document filtré par les dates de début et de fin choisies

    
    Trosee=20
    temperature=doc1["temp"]
    H=humidex_unite(temperature,Trosee)
    
    return H




#Detection d'anomalies


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
