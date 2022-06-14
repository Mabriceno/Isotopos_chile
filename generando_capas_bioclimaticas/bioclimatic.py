'''
Parametros Bioclimaticos de precipitacion y temperatura

'''

def ajuste_tri(i):
    if i==10:
        b,c=11,0
    elif i==11:
        b,c=0,1
    else:
        b,c=i+1,i+2
    return i,b,c

def tri_mas_humedo(var_anualxmes):
    tri_=var_anualxmes[0]+var_anualxmes[1]+var_anualxmes[2]
    meses=[0,1,2]
    for i in range(0,12):
        a,b,c = ajuste_tri(i)
        tri=var_anualxmes[a]+var_anualxmes[b]+var_anualxmes[c]
        if tri>tri_:
            tri_=tri
            meses=[a,b,c]

    return meses
def tri_mas_seco(var_anualxmes):
    tri_=var_anualxmes[0]+var_anualxmes[1]+var_anualxmes[2]
    meses=[0,1,2]
    for i in range(0,12):
        a,b,c = ajuste_tri(i)
        tri=var_anualxmes[a]+var_anualxmes[b]+var_anualxmes[c]
        if tri<tri_:
            tri_=tri
            meses=[a,b,c]
    return meses
def tri_mas_frio(var_anualxmes):
    tri_=(var_anualxmes[0]+var_anualxmes[1]+var_anualxmes[2])/3
    meses=[0,1,2]
    for i in range(0,12):
        a,b,c = ajuste_tri(i)
        tri=(var_anualxmes[a]+var_anualxmes[b]+var_anualxmes[c])/3
        if tri<tri_:
            tri_=tri
            meses=[a,b,c]
    return meses
def tri_mas_calido(var_anualxmes):
    tri_=(var_anualxmes[0]+var_anualxmes[1]+var_anualxmes[2])/3
    meses=[0,1,2]
    for i in range(0,12):
        a,b,c = ajuste_tri(i)
        tri=(var_anualxmes[a]+var_anualxmes[b]+var_anualxmes[c])/3
        if tri>tri_:
            tri_=tri
            meses=[a,b,c]
    return meses
def mes_mas_calido(var_anualxmes):
    val_=var_anualxmes[0]
    mes=0
    for i in range(0,12):
        if var_anualxmes[i]>val_:
            val_=var_anualxmes[i]
            mes=i
    return mes

def mes_mas_frio(var_anualxmes):
    val_=var_anualxmes[0]
    mes=0
    for i in range(0,12):
        if var_anualxmes[i]<val_:
            val_=var_anualxmes[i]
            mes=i
    return mes


import numpy as np 

class Param_bioclimaticos:

    def __init__(self, tm21, tmin1, tmax1, pr1,grid_lat1,grid_lon1,shape): # tm2,tmin,tmax,pr estructuras de 12xM, corresponden a 12 capas una por cada mes
        self.grid_lat=grid_lat1
        self.grid_lon=grid_lon1
        self.tm2=tm21
        self.tmin=tmin1
        self.tmax=tmax1
        self.pr=pr1
        self.bio1=np.nanmean(self.tm2,axis=0) #
        self.bio2=np.nanmean(self.tmax,axis=0)-np.nanmean(self.tmin,axis=0) #
        self.bio3=np.zeros(shape) #
        self.bio4=np.nanstd(self.tm2,axis=0)*100 #
        self.bio5=np.zeros(shape)
        self.bio6=np.zeros(shape)
        self.bio7=np.zeros(shape)
        self.bio8=np.zeros(shape)
        self.bio9=np.zeros(shape)
        self.bio10=np.zeros(shape)
        self.bio11=np.zeros(shape)
        self.bio12=np.nansum(self.pr,axis=0) #
        self.bio13=np.zeros(shape)
        self.bio14=np.zeros(shape)
        self.bio15=(np.nanstd(self.pr,axis=0)*100)/(1+np.nanmean(self.pr,axis=0)) #
        self.bio16=np.zeros(shape)
        self.bio17=np.zeros(shape)
        self.bio18=np.zeros(shape)
        self.bio19=np.zeros(shape)

    def I(self):
        for i in range(0,len(self.grid_lat)):
            for j in range(0,len(self.grid_lon)):
                
                var_tmax,var_tmin,var_tm2,var_pr=[],[],[],[]
                for m in range(0,12):
                    var_tmax.append(self.tmax[m][i][j])  ##OJO ACA
                    var_tmin.append(self.tmin[m][i][j])
                    var_tm2.append(self.tm2[m][i][j])
                    var_pr.append(self.pr[m][i][j])
                
                self.bio5[i][j]=self.tmax[mes_mas_calido(var_tmax)][i][j]
                self.bio6[i][j]=self.tmin[mes_mas_frio(var_tmin)][i][j]
                self.bio8[i][j]=(self.tm2[tri_mas_humedo(var_pr)[0]][i][j]+self.tm2[tri_mas_humedo(var_pr)[1]][i][j]+self.tm2[tri_mas_humedo(var_pr)[2]][i][j])/3
                self.bio9[i][j]=(self.tm2[tri_mas_seco(var_pr)[0]][i][j]+self.tm2[tri_mas_seco(var_pr)[1]][i][j]+self.tm2[tri_mas_seco(var_pr)[2]][i][j])/3
                self.bio10[i][j]=(self.tm2[tri_mas_calido(var_tm2)[0]][i][j]+self.tm2[tri_mas_calido(var_tm2)[1]][i][j]+self.tm2[tri_mas_calido(var_tm2)[2]][i][j])/3
                self.bio11[i][j]=(self.tm2[tri_mas_frio(var_tm2)[0]][i][j]+self.tm2[tri_mas_frio(var_tm2)[1]][i][j]+self.tm2[tri_mas_frio(var_tm2)[2]][i][j])/3
                self.bio13[i][j]=self.pr[mes_mas_calido(var_pr)][i][j]#mes mas lluvioso
                self.bio14[i][j]=self.pr[mes_mas_frio(var_pr)][i][j]#mes mas seco
                self.bio16[i][j]=(self.pr[tri_mas_humedo(var_pr)[0]][i][j]+self.pr[tri_mas_humedo(var_pr)[1]][i][j]+self.pr[tri_mas_humedo(var_pr)[2]][i][j])
                self.bio17[i][j]=(self.pr[tri_mas_seco(var_pr)[0]][i][j]+self.pr[tri_mas_seco(var_pr)[1]][i][j]+self.pr[tri_mas_seco(var_pr)[2]][i][j])
                self.bio18[i][j]=(self.pr[tri_mas_calido(var_tm2)[0]][i][j]+self.pr[tri_mas_calido(var_tm2)[1]][i][j]+self.pr[tri_mas_calido(var_tm2)[2]][i][j])
                self.bio19[i][j]=(self.pr[tri_mas_frio(var_tm2)[0]][i][j]+self.pr[tri_mas_frio(var_tm2)[1]][i][j]+self.pr[tri_mas_frio(var_tm2)[2]][i][j])
                #print(i,j)

    def II(self):
        self.bio7= self.bio5-self.bio6
        self.bio3=(self.bio2/self.bio7)*100
    
    def setup(self):
        self.I()
        self.II()

    def get_bio(self):

        return[self.bio1,
                self.bio2,
                self.bio3,
                self.bio4,
                self.bio5,
                self.bio6,
                self.bio7,
                self.bio8,
                self.bio9,
                self.bio10,
                self.bio11,
                self.bio12,
                self.bio13,
                self.bio14,
                self.bio15,
                self.bio16,
                self.bio17,
                self.bio18,
                self.bio19,
                ]
