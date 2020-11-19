import fonctions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


donnee = pd.read_csv("post-32566-EIVP_KM.csv", sep=';')

temperature=donnee['temp']
hum=donnee['humidity']
carbone=donnee['co2']

date=donnee['sent_at']


# print(donnee)



#_______________________________________________________________________________


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


def separer_date2(date_elementaire):         # a-b-c ...
    D=[]
    H=[]
    d,h=separer_date_heure(date_elementaire)
    a=''
    b=''
    for x in d:
        if x=='-':
            D.append(int(a))
            a=''
        if x!='-':
            a+=x
    D.append(int(a))
    for y in h:
        if y==':':
            H.append(int(b))
            b=''
        if y!=':':
            b+=y
    H.append(int(b))
    return D,H                           # [a,b,c]


def separer_date_liste(date_liste):
    L=[]
    H=[]
    n=len(date_liste)
    for k in range(n):
        l,h=separer_date(date_liste[k])
        L.append(l)
        H.append(h)
    return L,H


def separer_date_liste_num(date_liste):
    L=[]
    n=len(date_liste)
    for k in range(n):
        l=separer_date(date_liste[k])
        num=[k]
        num.append(l)
        L.append(num)
    return L


# Agrandi donnee

date2,heure2=separer_date_liste(date)
date2
donnee['date2']=date2
donnee['heure2']=heure2
donnee.sort_values(by=['date2','heure2'],ascending=[True,True])




def trouver_first_date(date_elementaire):       # Retourne la première index de la date 
    d,h=separer_date(date_elementaire)
    index=0
    while d>date2[index]:
        index+=1
        if index==len(date2):
            break
    return index


def trouver_last_date(date_elementaire):        # Retourne la dernière index de la date
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
    plt.plot(x, y, '.-',color='black', label="Carbone")
    plt.xlabel('Temps')
    plt.ylabel('Carbone')
    
    k,j,i=anomalie_list_plusieurs_jour(y,donnee.heure3,x)
    plt.scatter(i,j,color='red',label="anomalie")
    
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
    plt.title('Evolution carbone')
    
    plt.show()
    return None


def Afficher_temperature(start_date,end_date):
    i=trouver_first_date(start_date)
    j=trouver_last_date(end_date)
    x = date.tolist()[i:j]
    y = temperature.tolist()[i:j]
    plt.plot(x, y, '.-',color='blue', label="Température")
    plt.xlabel('Temps')
    plt.ylabel('Température')
    
    k,j,i=anomalie_list_plusieurs_jour(y,donnee.heure3,x)
    plt.scatter(i,j,color='red',label="anomalie")
    
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
    plt.title('Evolution température')
    
    plt.show()
    return None


def Afficher_luminosite(start_date,end_date):
    i=trouver_first_date(start_date)
    j=trouver_last_date(end_date)
    x = date.tolist()[i:j]
    y = donnee.lum.tolist()[i:j]
    plt.plot(x, y, '.-',color='yellow', label="Luminosité")
    plt.xlabel('Temps')
    plt.ylabel('Luminosité')
    
    k,j,i=anomalie_list_plusieurs_jour(y,donnee.heure3,x)
    plt.scatter(i,j,color='red',label="anomalie")
    
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
    plt.title('Evolution luminosité')
    
    plt.show()
    return None
    

def Afficher_bruit(start_date='2019-08-11',end_date='2019-08-25'):
    i=trouver_first_date(start_date)
    j=trouver_last_date(end_date)
    x = date.tolist()[i:j]
    y = donnee.noise.tolist()[i:j]
    plt.plot(x, y, '.-',color='green', label="Bruit")
    plt.xlabel('Temps')
    plt.ylabel('Bruit')
    
    k,j,i=anomalie_list_plusieurs_jour(y,donnee.heure3,x)
    plt.scatter(i,j,color='red',label="anomalie")
    
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
    plt.title('Evolution bruit')
    
    plt.show()
    return None


def Afficher_humidite(start_date='2019-08-11',end_date='2019-08-25'):
    i=trouver_first_date(start_date)
    j=trouver_last_date(end_date)
    x = date.tolist()[i:j]
    y = hum.tolist()[i:j]
    plt.plot(x, y, '.-',color='cyan', label="Humidité")
    plt.xlabel('Temps')
    plt.ylabel('Humidité')
    
    k,j,i=anomalie_list_plusieurs_jour(y,donnee.heure3,x)
    plt.scatter(i,j,color='red',label="anomalie")
    
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
    plt.title('Evolution humidité')
    
    plt.show()
    return None


def Afficher_courbe(start_date='2019-08-11',end_date='2019-08-25'):
    i=trouver_first_date(start_date)
    j=trouver_last_date(end_date)
    x = date.tolist()[i:j]
    y = carbone.tolist()[i:j]
    plt.plot(x, y, '.-',color='black', label="Carbone")
    y2 = temperature.tolist()[i:j]
    plt.plot(x, y2, '.-',color='blue', label="Température")
    y3 = donnee.lum.tolist()[i:j]
    plt.plot(x, y3, '.-',color='yellow', label="Luminosité")
    y4 = donnee.noise.tolist()[i:j]
    plt.plot(x, y4, '.-',color='green', label="Bruit")
    y5 = hum.tolist()[i:j]
    plt.plot(x, y5, '.-',color='cyan', label="Humidité")
    
    plt.xlabel('Temps')
    
    k,j,i=anomalie_list_plusieurs_jour(y,donnee.heure3,x)
    plt.scatter(i,j,color='red',label="anomalie")
    
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
    plt.title('Donnée entre le '+start_date+' et le '+end_date)
    
    plt.show()
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



#_______________________________________________________________________________

# Une journée :

def list_une_journee(L,jour):      #L est la liste de donnée
    donnee_une_journee=donnee[donnee.date2==jour]
    donnee_une_journee.sort_values(by='heure2')
    return donnee_une_journee


def separer_une_journee(jour):
    donnee_une_journee=list_une_journee(donnee,jour)
    
    carbone_journee=donnee_une_journee.co2.tolist()
    temperature_journee=donnee_une_journee.temp.tolist()
    hum_journee=donnee_une_journee.humidity.tolist()
    bruit_journee=donnee_une_journee.noise.tolist()
    lum_journee=donnee_une_journee.lum.tolist()
    date_journee=donnee_une_journee.date2
    heure_journee=donnee_une_journee.heure2.tolist()
    
    return carbone_journee,temperature_journee,hum_journee,bruit_journee,lum_journee,date_journee,heure_journee

    
def Afficher_une_journee(jour):
    donnee_une_journee=list_une_journee(donnee,jour)
    donnee_une_journee.sort_values(by='heure2')
    x = donnee_une_journee.heure2
    y = donnee_une_journee.co2
    y2 = donnee_une_journee.temp
    y3 = donnee_une_journee.noise
    y4 = donnee_une_journee.humidity
    y5 = donnee_une_journee.lum

    plt.plot(x, y,color='black')
    plt.plot(x, y2,color='red')
    plt.plot(x, y3,color='green')
    plt.plot(x, y4,color='cyan')
    plt.plot(x, y5,color='yellow')
    
    plt.xlabel('Temps')
    plt.title('Donnée sur une journée')
    
    plt.show()
    return None


def Afficher_journee(jour,col=None):
    if col==None:
        return Afficher_une_journee(jour)
    
    donnee_une_journee=list_une_journee(donnee,jour)
    donnee_une_journee.sort_values(by='heure2')
    x = donnee_une_journee.heure2
    y = donnee_une_journee.col

    plt.plot(x, y)
    
    plt.xlabel('Temps')
    plt.title('Donnée sur une journée')
    
    plt.show()
    return None


def Afficher_une_journee2(jour):
    if len(str(jour))>8:
        jour,h=separer_date(str(jour))
    jour=int(jour)
    carbone_journee,temperature_journee,hum_journee,bruit_journee,lum_journee,date_journee,heure_journee=separer_une_journee(jour)
    jour2=jour-20190800

    x = [x/10000 for x in heure_journee]
    y = carbone_journee
    y2 = temperature_journee
    y3 = bruit_journee
    y4 = hum_journee
    y5 = lum_journee
    
    plt.plot(x, y,color='black',label="carbone")
    plt.plot(x, y2,color='red',label="température")
    plt.plot(x, y3,color='green',label="bruit")
    plt.plot(x, y4,color='cyan',label="humidité")
    plt.plot(x, y5,color='yellow',label="luminosité")
    
    plt.xlabel('Temps')
    plt.title('Donnée de la journee du '+str(jour2)+' août 2019')
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    return None



#_______________________________________________________________________________

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


#_______________________________________________________________________________


def Afficher_un_jour_avec_anomalie(jour):
    if len(str(jour))>8:
        jour,h=separer_date(str(jour))
    jour=int(jour)
    carbone_journee,temperature_journee,hum_journee,bruit_journee,lum_journee,date_journee,heure_journee=separer_une_journee(jour)
    
    jour2=jour-20190800

    x = [x/10000 for x in heure_journee]
    y = carbone_journee
    y2 = temperature_journee
    y3 = bruit_journee
    y4 = hum_journee
    y5 = lum_journee
    
    plt.plot(x, y,color='black',label="carbone")
    plt.plot(x, y2,color='blue',label="température")
    plt.plot(x, y3,color='green',label="bruit")
    plt.plot(x, y4,color='cyan',label="humidité")
    plt.plot(x, y5,color='yellow',label="luminosité")
    
    k,j,i=anomalie_list_une_journee(lum_journee,donnee.heure3,heure_journee)
    i=[x/10000 for x in i]
    plt.scatter(i,j,color='red',label="anomalie")
    
    plt.xlabel('Temps')
    plt.title('Donnée de la journee du '+str(jour2)+' août 2019')
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    return None


def Afficher_un_jour_avec_anomalie_carbone(jour):
    if len(str(jour))>8:
        jour,h=separer_date(str(jour))
    jour=int(jour)

    carbone_journee,temperature_journee,hum_journee,bruit_journee,lum_journee,date_journee,heure_journee=separer_une_journee(jour)

    jour2=jour-20190800

    x = [x/10000 for x in heure_journee]
    y = carbone_journee
    plt.plot(x, y,color='black',label="carbone")
    
    k,j,i=anomalie_list_une_journee(carbone_journee,donnee.heure3,heure_journee)
    i=[x/10000 for x in i]
    plt.scatter(i,j,color='red',label="anomalie")
    
    plt.xlabel('Temps')
    plt.title('Donnée de la journee du '+str(jour2)+' août 2019')
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    return None
