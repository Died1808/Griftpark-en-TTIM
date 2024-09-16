from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
from shapely import geometry
import math
import numpy as np
import pandas as pd
from ttim import *
import matplotlib.pyplot as plt 
import fiona
import os
import seaborn as sns

def kleuren(n):
    color = ['darkblue','darkorange','grey','red','green']
    params = {'font.family': 'sans-serif',
              'font.sans-serif': 'arial',
              'axes.labelsize': 10,
              'axes.facecolor': '#ffffff', 
              'axes.labelcolor': 'black',
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'lines.linewidth': 1,
              'grid.color': 'grey',
              'grid.linestyle': 'dashed',
              'grid.linewidth': 0.5,
              'text.usetex': False,
              'font.style': 'normal',
              'font.variant':'normal',
              'figure.facecolor': 'white',
              'font.size':8,
              'figure.autolayout': True,
              'figure.figsize': (10,8),
              'figure.dpi': 100,
              }
    plt.rcParams.update(params)
    
#############################################################
pd.set_option('display.max_columns', None) 
pd.options.display.width=None
pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', True)
pd.options.display.float_format = '{:,.2f}'.format


class GEF:
    def __init__(self):
        self._data_seperator = ' '
        self._columns = {}
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.dz = []
        self.qc = []
        self.pw = []
        self.wg = []
        self.c  = []
        self.kb = []
        self.dist =[]
        self.npor=[]
        
    def readFile(self, filename):
        lines = open(filename, 'r').readlines()
        for line in lines:
            reading_header = True
        for line in lines:   
            if reading_header:
                self._parseHeaderLine(line)
            else:
                self._parseDataLine(line)
            if line.find('#EOH') > -1:
                if self._check_header():
                    reading_header = False
                else:
                    print(filename,'bestaat al')
                    return
            
    def _check_header(self):
        if not 1 in self._columns:
            return False
        if not 2 in self._columns:
            return False
        return True

    def _parseHeaderLine(self, line):
        for xe in ['#COMMENT', 'Peil=', 'uitvoerder', 'materieel','WATERSTAND',
                    'opmerkingen#MEASUREMENTTEXT','==','= NAP','antropogeen']:
            if xe in line:
                return          
        if len(line.split()) == 0:
            return
        
        keyword, argline = line.split('=')         
        keyword = keyword.strip()
        argline = argline.strip()
        args = argline.split(',')
      
        if '#XYID' in line:
            argline = argline.replace('.','')        
        args = argline.split(',')

        if 'Waterspanning'  in line:
            self.u = float(args[3])
        if 'Waterdruk'  in line:
            self.u = float(args[3]) 
        try:
            if 'waterspanning'  in line:
                self.u = int(args[3])
        except ValueError:
            return

        if keyword=='#XYID':
            if float(args[1]) < 1e5:
                args[1] = args[1]
            else:
                args[1]=args[1].replace('.','')
            self.x = float(args[1])
            args[2]=args[2].replace('.','')
            self.y = float(args[2])
            if (len(str(int(self.x))))>5:
                 self.x=int(self.x/pow(10,len(str(int(self.x)))-6))
            if (len(str(int(self.y))))>5:
                 self.y=int(self.y/pow(10,len(str(int(self.y)))-6))
            if self.x > 3e5:
                self.x=self.x/10

        elif keyword=='#ZID':
            self.z = round(float(args[1]),3)
           
        elif keyword=='#COLUMNINFO':
            column = int(args[0])
            dtype = int(args[-1])
            if dtype==11:
                dtype = 10
            self._columns[dtype] = column - 1    
       
    def _parseDataLine(self, line):
        line=line.strip()
        line = line.replace('|',' ')
        line = line.replace(';',' ')
        line = line.replace('!',' ')
        line = line.replace(':',' ')
        args=line.split()
        for n, i in enumerate(args):
            if i in ['9.9990e+003','-9999.9900','-1.0000e+007','-99999','-99999.0',
                  '-99999.00','-99999.000','-9.9990e+003','999.000', '-9999.99', '999.999',
                  '99','9.999','99.999', '999.9']:
                args[n] = '0.1'       
        if len(line.split()) == 0:
            return
 
        zz  = round(abs(float(args[self._columns[1]])),4)
        dz = round(self.z - zz,4)
        qc = round(float(args[self._columns[2]]),4)  

        slope    =  0.0104
        intercept=  0.0190
                         
        pw = float(args[self._columns[3]]) 
        if pw<-10:
            pw=0.1
        elif pw>5:
            pw=slope*qc+intercept             
                

        self.dz.append(dz)
        self.qc.append(qc)
        self.pw.append(pw)
        if qc<=0.001:
            qc=0.1
            self.wg.append(10.)
        else:
            wg = abs((pw / qc) * 100.)
        wg = abs((pw / qc) * 100.)

##############################################K-waarde        
        if wg>=0.0:
            if wg >5: wg=15
            ke=math.exp(wg)
        if ke <=0:  ke=1
        else:
            kb  = (qc / ke)*2
            self.kb.append(kb)
        if kb <=0.1:
            npor=0.05
        elif 0.1<kb<=1:
            npor=0.1
        elif 1<kb<=30:
            npor=0.35
        elif kb>30:
            npor=0.25
        self.npor.append(npor)
            
    def asNumpy(self):
        return np.transpose(np.array([self.dz, self.kb]))

    def asDataFrame(self):
        a = self.asNumpy()
        return pd.DataFrame(data=a, columns=['depth', 'k'])
        
    def plt(self, filename):
        df = self.asDataFrame()
        df = df.sort_values('depth', ascending=False)

        if df.empty:
            return df
        
        df = df.rolling(10).mean() 
        df = df.iloc[:: 10]
        df=df.dropna()
        df = df.reset_index(drop=True)
        df['k'][0] = 0.02
        a = df
        
#%%Data
        t       =  (150)        # Lengte pompproef in dagen  
        print('duur pompproef ' , round(t,1), ' dagen')
        Qm1     =  24*10        # Debiet dag
        Tf1     =  self.z-12    # Top filter in m NAP z
        Bf1     =  self.z-20    # Onderzijde filter in m NAP
        factor  =  1            # dikte laag [m]
        fT      =  self.z       # Onderzijde freatisch niveau (GLG?)), dit ligt verassend op (zelfs in dikke freatische pakketten) maar een paar meter onder de GWS)
        corAq   =  1
        Cf      =  0.2          # kleiner is lagere S 
        Anif    =  0.5          # hoger is hogere anisotropie
        TWVP    =  self.z
        OWVP    =  -55
        ff      =   10.0        # Fijnfactor kleien
        dzend   =  -62
        print(filename)
        print('Filterquote =  ', round(((abs(Bf1)-abs(Tf1))/(abs(OWVP)-abs(TWVP))),2))
        
#%%Maken Dataframe voor tabel en maken S-waarde
        a['Anif'] = Anif
        a.columns=['Top', 'k', 'Anif']
        a=a.dropna()

        a['k'][54] = 0.08 # ingebracht om laag op PP locatie mee te nemen 
        a['k'][59] = 0.01 # ingebracht om laag op PP locatie mee te nemen 
        
        a['k']    = np.where(a['k']<1, a['k']/ff, a['k']*corAq)
        a['kzkh'] = np.where(a.Top>TWVP,2/(33.653*np.exp(-0.066*(a['k']))), 2/a['Anif']/(33.653*np.exp(-0.066*(a['k']))))  
        a['kzkh'] = np.where(a['kzkh']> 1, 1, a[ 'kzkh'])
        a['S']    = np.where(a.Top>TWVP, (a['k']/(24*60*60))*a['kzkh'],(((a['k']/(24*60*60))*a['kzkh']))*Cf)
        a['S']    = np.where(a['Top'] >fT, 0.05, a['S'])
        a['S']    = a['S'].apply(lambda x: '%.1e' % x).values.tolist() 
        a['an']   = (np.round(1/a['kzkh'],2)).astype('float')
        a = a.reset_index(drop=True)

#%% Transmitivity & Storativity voor op de plot
        a['d']   = factor
        a['kd']  = a['d']*a['k']
        a['c']   = 1/a['k']*a['d']
        print(a)

#%%Modelparameters TTim
        Z= a['Top'].tolist()
        Z.append(dzend-1)
        Z.append(dzend-2)
        Kaq= a['k'].tolist()
        Kaq.append(1e-3)
        S=a['S'].tolist()
        S.append(1e-5)
        kzkh=a['kzkh'].tolist()
        kzkh.append(0.15)

        tmin=1e-6
        tmax=1e6
        
#%% Keuze pomplagen en filterlagen   en WVP dikte     
        b=a
        b=a
        d=a

        b=b.drop(b[(b.Top > Tf1)].index)
        b=b.drop(b[(Bf1 >= b.Top)].index)
      
        d =d.drop(d[(d.Top > TWVP)].index)
        d =d.drop(d[(OWVP >= d.Top)].index)
        Ld=list(d.index.values) #WVP_lagen

        d['S'] = pd.to_numeric(d['S'])
        St=d['S'].sum()*factor
        if St>0.35:
            St=0.35
        print('Bergingsfactor WVP = ' ,  "{:.2E}".format(St))
       
        Lb=list(b.index.values) #Pomplagen1
        print('Filterlagen1 = ', Lb)
        
        a['c'] = a['c'].astype('int')
        a.to_excel('invoer.xlsx')
        
        print(int(a.k.sum()), 'KD [m2/dag]')
        print(round(a.k.sum()/54,1), 'k [m/dag]')
       
       
#%%model TTim       
        ml=Model3D(kaq=Kaq, z=Z, Saq= S, kzoverkh=kzkh, phreatictop=False, 
                    tmin=tmin, tmax=tmax, M=20)
        
        B20 = Well(ml,   xw=137262,  yw=456945, rw=0.25, tsandQ=[(0,0),
                                                      (31+24,Qm1),
                                                      (31+31,0),
                                                      (31+38,Qm1),
                                                      (31+45,0),
                                                      (31+59,Qm1),
                                                      (31+73,0),
                                                      (31+83,0),
                                                      (31+98,0),
                                                      (31+99,Qm1),
                                                      (31+102,0),
                                                      (31+113,Qm1),
                                                      ], layers=Lb)
        
        Qm2 = 24*3.5
        B21 = Well(ml,   xw=137227, yw=457090, rw=0.25, tsandQ=[(0,Qm2),
                                                      (31+24,Qm2),
                                                      (31+31,Qm2),
                                                      (31+38,Qm2),
                                                      (31+45,Qm2),
                                                      (31+59,Qm2),
                                                      (31+73,Qm2),
                                                      (31+83,0),
                                                      (31+98,Qm2),
                                                      (31+99,Qm2),
                                                      (31+102,0),
                                                      (31+113,Qm2),
                                                      ], layers=Lb)
        Qm3 = 24*4
        B22 = Well(ml,   xw=137175,	yw=457056, rw=0.25, tsandQ=[(0,Qm3),
                                                      (31+24,Qm3),
                                                      (31+31,Qm3),
                                                      (31+38,Qm3),
                                                      (31+45,Qm3),
                                                      (31+59,Qm3),
                                                      (31+73,Qm3),
                                                      (31+83,0),
                                                      (31+98,Qm3),
                                                      (31+99,Qm3),
                                                      (31+102,0),
                                                      (31+113,Qm3),
                                                      ], layers=Lb)


        mat=pd.read_excel('Schermwand.xlsx', engine='openpyxl')  
        mat=mat.replace(',','.')
        damwand = []
        damwand   = mat[mat['Naam'].str.match ('dw')]  
        wandlaag=np.arange(0,56,1)  # wanden moeten per laag 

        xp = []
        yp = []
        for rnum in damwand.index:
            xdw = damwand['x'][rnum]
            ydw = damwand['y'][rnum]
            xp.append(xdw)
            yp.append(ydw)
        xpf = xp[0]
        ypf = yp[0]
        xp.append(xpf)
        yp.append(ypf)

        Schermwand = LeakyLineDoubletString(ml, xy=list(zip(xp, yp)), res=54, 
                                            layers=wandlaag)        

        aantal_cores = os.cpu_count()
        pool = ProcessPoolExecutor(max_workers=aantal_cores)
        
        list(pool.map(ml.solve()))
        kleuren(1)
        mat1=pd.read_excel('WP.xlsx', index_col=None, skiprows=0, engine='openpyxl')
        x1  = mat1['tijd']
        y1  = mat1['Bu1']*-1/100+0.18 #0.18 dit is de verlaging die 21+22 permanent veroorzaken)
        y2  = mat1['Bu3']*-1/100+0.05 #0.05 dit is de verlaging die 21+22 permanent veroorzaken)
        y3  = mat1['Bu4']*-1/100+0.05 #0.05 dit is de verlaging die 21+22 permanent veroorzaken)
        t=np.linspace(0,150,15000)
        plt.plot(x1, y1, 'b--',   markersize=2)
        plt.plot(x1, y2, 'g--',   markersize=2)
        plt.plot(x1, y3, 'r--',   markersize=2)

        s1 = ml.head(137231,456947, t)
        s2 = ml.head(137302,456941, t) #DV12_in
        s3 = ml.head(137319,456948, t) #DV11_uit
       
        plt.plot(t, s1[18]*-1, label='Bu1_-18', color = 'blue',  lw=2, alpha=1)
        plt.plot(t, s1[55]*-1, label='Bu3_-55', color = 'green', lw=2, alpha=1)
        plt.plot(t, s1[61]*-1, label='Bu4_-61', color = 'red',   lw=2, alpha=1)
        plt.plot(t, s2[31]*-1, label='DV12_-31',color = 'darkorange', lw=2, alpha=1)
        plt.plot(t, s3[31]*-1, label='DV11_-31',color = 'hotpink', lw=2, alpha=1)
        
        plt.grid(axis='both')
        plt.ylim(1,0)
        plt.xlim(50,150)
        plt.xticks(np.arange(50,150,5))
        plt.grid('both')
        plt.xlabel('tijd [dagen], dagnr 90 = 31 maart 2022')
        plt.ylabel('verlaging [m]')
        plt.title(filename + '  S_WVP = '+ '%.1e'% (St), fontsize=12, pad=10)
        plt.legend()
        plt.savefig('Tijd_dh_Songef.png', bbox_inches='tight')
        plt.show()
        plt.close()


#%% Contouren   
        x1 = 137262  
        y1 = 456945     
        cs=ml.contour(win=[x1-500,x1+500,y1-500,y1+500], ngr=100, labels=False, decimals=2,
                    layers=[18], levels=np.arange(-1, 0, 0.05), t=148, figsize=(12,12))
        plt.show()
        
#%%Maken Shapefiles                  
        OpenPolyList = []

        # Verkrijg contourpaden
        for level, path in zip(cs.levels, cs.get_paths()):
            z = round(level, 2)
            for ncp, cp in enumerate(path.to_polygons()):
                cp = np.asarray(cp)
                lons = cp[:, 0]
                lats = cp[:, 1]
                vorm = [(i[0], i[1]) for i in zip(lons, lats)]
                vorm.pop(-1)
                poly = geometry.LineString(vorm)
                OpenPolyList.append({'poly': poly, 'props': {'z': z}})
        
        # Schrijf naar ESRI Shapefile
        outfi = os.path.join('GriftparkFOUT'+'.shp')
        schema = {'geometry': 'LineString', 'properties': {'z': 'float'}}
        with fiona.collection(outfi, "w", "ESRI Shapefile", schema) as output:
            for p in OpenPolyList:
                output.write({'properties': p['props'],
                              'geometry': geometry.mapping(p['poly'])})
        
# %% Anisotropie    
        kleuren(1)    
        az= a[a.k>1.0]
        print('Berekende anisotropie zand: ',  round(az.an.median(),1))
        sns.kdeplot(data=az, x='an', color='orange', fill='orange', common_norm=False, legend=False)

        plt.xlabel('Anisotropie  (Kh/Kv)')
        plt.ylabel('Relatief voorkomen')
        plt.xlim(0,50)
        plt.grid()
        plt.xticks(np.arange(0,50,5))

        plt.savefig('Anisotropie.png', bbox_inches='tight')

        plt.show()
        plt.close()

# %% Tijd
        end = time.time()
        hours, rem = divmod(end-strt, 3600)
        minutes, seconds = divmod(rem, 60)
        print('Rekentijd: '+"{:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds)))  
        
strt=time.time()
       
for filename in os.listdir(os.getcwd()):
    if filename.endswith ('.GEF') or filename.endswith ('.gef'):
        if __name__=="__main__":
            g=GEF()
            g.readFile(filename)
            g.plt(filename)
            plt.show()
        plt.close()
