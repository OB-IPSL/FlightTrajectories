import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import argparse
import os.path
import warnings
import sys
import great_circle_calculator.great_circle_calculator as gcc

#--OB: gradient descent
#--EG: Zermelo method
#
#  example on how to call the python script
#  python optim_contrail_all.py --yr=2019 --level=200
#  level=-1 => the script uses the pressure level that is closest to the average IAGOS flight pressure
#  one can prescribe inputfile, otherwise route, otherwise mth
#
parser = argparse.ArgumentParser()
parser.add_argument('--yr', type=int, choices=[2018,2019,2020,2021,2022], default=2019, help='year')
parser.add_argument('--level', type=int, choices=[-1,150,175,200,225,250], default=-1, help='level (hPa)')
parser.add_argument('--maxiter', type=int, default=100, help='max number of iterations')
parser.add_argument('--method', type=str, default='SLSQP', choices=['SLSQP','BFGS','L-BFGS-B'], help='minimization method')
#
#--get arguments from command line
args = parser.parse_args()
#
#--copy arguments into variables
yr=args.yr
level=args.level
maxiter=args.maxiter
method=args.method
#
print('\n')
print('year=',yr)
print('level=',level)
print('maxiter=',maxiter)
print('method=',method,'\n')
#
#--path with graph outputs
pathgraph='../GRAPHS/'

#--define regions
regions=['europe','southamerica','northamerica','centralamerica','asia','centralasia',\
         'southeastasia','australasia','middleeast','africa','pacific']
label_regions=['Europe','South America','North America','Central America','Asia','Central Asia',\
         'South-East Asia','Australasia','Middle East','Africa','Pacific']
nb_regions=len(regions)
#
#--flag to show plots while running script
pltshow=True
pltshow=False
#
#--IAGOS flight IDs
flight_ids=['D-AIGT','D-AIKO','D-AIHE','N384HA','B-18316','B-18317','B-18806','F-GLZU','F-GZCO','B-HLR']
#
#--airlines
airlines=['Lufthansa','Hawaiian Airlines','China Airlines','Air France','Cathay Pacific']
#
operators={'D-AIGT':'Lufthansa','D-AIKO':'Lufthansa','D-AIHE':'Lufthansa',
          'N384HA':'Hawaiian Airlines',
          'B-18316':'China Airlines','B-18317':'China Airlines','B-18806':'China Airlines',
          'F-GLZU':'Air France','F-GZCO':'Air France',
          'B-HLR':'Cathay Pacific'}
#
def aircraft2airline(row):  
    return operators[row['flightid_iagos']]
#
#--define priorities and colors for IAGOS airlines
priorities={'D-AIGT':0,'D-AIKO':0,'D-AIHE':0,     #--Lufthansa
            'N384HA':1,                           #--Hawaiian Airlines
            'B-18316':2,'B-18317':2,'B-18806':2,  #--China Airlines
            'F-GLZU':3,'F-GZCO':3,                #--Air France
            'B-HLR':4}                            #--Cathay Pacific
colors={'D-AIGT':'blue','D-AIKO':'blue','D-AIHE':'blue','Lufthansa':'blue',              #--Lufthansa
        'N384HA':'red','Hawaiian Airlines':'red',                                        #--Hawaiian Airlines
        'B-18316':'green','B-18317':'green','B-18806':'green','China Airlines':'green',  #--China Airlines
        'F-GLZU':'purple','F-GZCO':'purple','Air France':'purple',                       #--Air France
        'B-HLR':'orange','Cathay Pacific':'orange'}                                      #--Cathay Pacific
#
#--path with output data
pathout='/projsu/cmip-work/oboucher/FR24/ROUTE/'+str(maxiter)+'/'+method+'/'
#
#--stop warnings from division by zero
#warnings.filterwarnings("ignore")
#
#--dictionary with results
nb_flights=np.zeros((nb_regions,nb_regions))
time_elapsed_OB=np.zeros((nb_regions,nb_regions))
time_elapsed_EG=np.zeros((nb_regions,nb_regions))
success_rate_EG=np.zeros((nb_regions,nb_regions))
time_iagos=np.zeros((nb_regions,nb_regions))
time_quick=np.zeros((nb_regions,nb_regions))
time_short=np.zeros((nb_regions,nb_regions))
time_nownd=np.zeros((nb_regions,nb_regions))
#
#--dataframe out
dfout=pd.DataFrame(data={'Itinerary':[],'From':[],'To':[],'Year':[],'Nb flights':[],'IAGOS/Quickest':[],'Quickest/Shortest':[],'Shortest/No wind':[],'IAGOS/No wind':[]})
dfout.set_index('Itinerary',inplace=True)
#
#---loop on regions
for i1,region1 in enumerate(regions):
   for i2,region2 in enumerate(regions):
      route=region1+'_to_'+region2
      file=pathout+route+'_'+str(yr)+'_lev'+str(level)+'.csv'
      print(file)
      if not os.path.isfile(file): 
          continue
      #--read dataframe
      df=pd.read_csv(file)
      df.set_index('flightname_iagos',inplace=True)
      #--add airline
      df['airline']=df.apply(lambda row: aircraft2airline(row), axis=1)
      #--remove frames for which EG has failed
      df.loc[df['time EG'] > 1.02*df['time OB'],'time EG']=np.nan
      #--compute percentage differences
      df['dt OB']=(df['time_iagos']-df['time OB'])/df['time OB']*100.
      df['dt EG']=(df['time_iagos']-df['time EG'])/df['time EG']*100.
      nbflights=len(df)
      #
      #--recomputing distance
      for index, row in df.iterrows(): 
         df['time shortest no wind']=df['dist_gcc']/df['airspeed']/3600.
      #--computing optimality
      #--filling arrays
      nb_flights[i1,i2]=len(df)
      time_elapsed_OB[i1,i2]=df['time elapsed OB'].mean()
      time_elapsed_EG[i1,i2]=df['time elapsed EG'].mean()
      success_rate_EG[i1,i2]=(1.-df['time EG'].isnull().sum()/nbflights)*100.
      time_iagos[i1,i2]=df['time_iagos'].sum()
      time_quick[i1,i2]=df['time OB'].sum()
      time_short[i1,i2]=df['time shortest'].sum()
      time_nownd[i1,i2]=df['time shortest no wind'].sum()
      #--filling dataframe
      dfout.loc[label_regions[i1]+' to '+label_regions[i2],'Year']=yr
      dfout.loc[label_regions[i1]+' to '+label_regions[i2],'From']=label_regions[i1]
      dfout.loc[label_regions[i1]+' to '+label_regions[i2],'To']=label_regions[i2]
      dfout.loc[label_regions[i1]+' to '+label_regions[i2],'Nb flights']=len(df)
      dfout.loc[label_regions[i1]+' to '+label_regions[i2],'IAGOS/Quickest']=df['time_iagos'].sum()/df['time OB'].sum()
      dfout.loc[label_regions[i1]+' to '+label_regions[i2],'Quickest/Shortest']=df['time OB'].sum()/df['time shortest'].sum()
      dfout.loc[label_regions[i1]+' to '+label_regions[i2],'Shortest/No wind']=df['time shortest'].sum()/df['time shortest no wind'].sum()
      dfout.loc[label_regions[i1]+' to '+label_regions[i2],'IAGOS/No wind']=df['time_iagos'].sum()/df['time shortest no wind'].sum()
      #--print
      print(route)
      print('number of flights=',nb_flights[i1,i2])
      print('time elapsed Gradient descent=',time_elapsed_OB[i1,i2])
      print('time elapsed Zermelo method=',time_elapsed_EG[i1,i2])
      print('Zermelo method success rate=',success_rate_EG[i1,i2])
      print('ratio IAGOS to quickest=',time_iagos[i1,i2]/time_quick[i1,i2])
      print('ratio quickest to shortest=',time_quick[i1,i2]/time_short[i1,i2])
      print('ratio shortest to shortest no wind=',time_short[i1,i2]/time_nownd[i1,i2])
      print('==> ratio IAGOS to shortest no wind=',time_iagos[i1,i2]/time_nownd[i1,i2])
      #--compare airlines (eg AF and LH)
      ratio_route={}
      for airline in airlines:
        if airline in list(df['airline']):
          ratio_route[airline]=df[df['airline']==airline]['time_iagos'].sum()/df[df['airline']==airline]['time OB'].sum()
        else: 
          ratio_route[airline]=np.nan
      print('Ratios IAGOS/quickest for',[airline for airline in airlines],'=',[ratio_route[airline] for airline in airlines],'\n')
      if not np.isnan(ratio_route['Air France']) and not np.isnan(ratio_route['Lufthansa']):
          print(route,'AF=',ratio_route['Air France'],' LH=',ratio_route['Lufthansa'])
      #
      #--optimal flight times
      plt.figure(figsize=(20,6))
      if route=='':
        plt.title('IAGOS flight routes from '+str(yr)+' '+str(mth).zfill(2),fontsize=24)
      else:
        plt.title('IAGOS flights from '+label_regions[i1]+' to '+label_regions[i2]+' - '+str(yr),fontsize=24)
      plt.plot(np.arange(nbflights),df['dt OB'],label='Gradient descent')
      plt.plot(np.arange(nbflights),df['dt EG'],label='Zermelo method',zorder=100)
      plt.plot([0,nbflights],[0,0],color='black',linewidth=0.3)
      plt.xticks(np.arange(nbflights),df.index.values,rotation=80)
      plt.yticks(fontsize=24)
      plt.ylabel('Flight time difference (%)',fontsize=24)
      plt.legend(fontsize=24)
      plt.tight_layout()
      if route=='':
        plt.savefig(pathgraph+'STATS_'+str(yr)+str(mth).zfill(2)+'.png')
      else:
        plt.savefig(pathgraph+'STATS_'+route+'_'+str(yr)+'.png')
      if pltshow: plt.show()
      plt.close()
      #
      #--computing times
      plt.figure(figsize=(20,6))
      if route=='':
        plt.title('IAGOS flight routes from '+str(yr)+' '+str(mth).zfill(2),fontsize=24)
      else:
        plt.title('IAGOS flights from '+label_regions[i1]+' to '+label_regions[i2]+' - '+str(yr),fontsize=24)
      plt.plot(np.arange(nbflights),df['time elapsed OB'],label='Gradient descent')
      plt.plot(np.arange(nbflights),df['time elapsed EG'],label='Zermelo descent')
      plt.xticks(np.arange(nbflights),df.index.values,rotation=80)
      plt.yticks(fontsize=24)
      plt.ylabel('Time elapsed (s)',fontsize=24)
      plt.yscale('log')
      plt.legend(fontsize=24)
      plt.tight_layout()
      if route=='':
        plt.savefig(pathgraph+'COMP_TIME_'+str(yr)+str(mth).zfill(2)+'.png')
      else:
        plt.savefig(pathgraph+'COMP_TIME_'+route+'_'+str(yr)+'.png')
      if pltshow: plt.show()
      plt.close()
#
#===========================================
#--scatter plot of all flight times by route
#===========================================
plt.figure(figsize=(10,10))
#---loop on regions
for i1,region1 in enumerate(regions):
   for i2,region2 in enumerate(regions):
      route=region1+'_to_'+region2
      file=pathout+route+'_'+str(yr)+'_lev'+str(level)+'.csv'
      if not os.path.isfile(file): 
          continue
      df=pd.read_csv(file)
      if len(df['time OB'])>0:
        plt.scatter(df['time OB'],df['time_iagos'],label=label_regions[i1]+' => '+label_regions[i2])
#--make plot nice
plt.legend(fontsize=9)
plt.xlim(2.5,12)
plt.ylim(2.5,12)
plt.xlabel('Optimised flight time (hours)',fontsize=20)
plt.ylabel('Actual IAGOS flight time (hours)',fontsize=20)
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot([0.1,15],[0.1,15],color='black',linestyle='solid',linewidth=1)
plt.plot([0.1,15],[0.1*1.01,15*1.01],color='black',linestyle='solid',linewidth=0.5)
plt.plot([0.1,15],[0.1*1.02,15*1.02],color='black',linestyle='solid',linewidth=0.5)
plt.plot([0.1,15],[0.1*1.05,15*1.05],color='black',linestyle='solid',linewidth=0.5)
plt.plot([0.1,15],[0.1*1.10,15*1.10],color='black',linestyle='solid',linewidth=0.5)
plt.plot([0.1,15],[0.1*1.20,15*1.20],color='black',linestyle='solid',linewidth=0.5)
plt.savefig(pathgraph+'SCATTER_PLOT_by_route_'+str(yr)+'.png')
if pltshow: plt.show()
plt.close()
#
#==============================================
#--scatter plot of all flight times by aircraft
#==============================================
plt.figure(figsize=(10,10))
flight_id_present={}
for flight_id in flight_ids:
   flight_id_present[flight_id]=False
#---loop on regions
for i1,region1 in enumerate(regions):
   for i2,region2 in enumerate(regions):
      route=region1+'_to_'+region2
      file=pathout+route+'_'+str(yr)+'_lev'+str(level)+'.csv'
      if not os.path.isfile(file): continue
      df=pd.read_csv(file)
      dt_iagos_ob=(df['time_iagos']/df['time OB']).values
      for flight_id in flight_ids: 
         if len(df[df['flightid_iagos']==flight_id]['time OB'])>0:
            flight_id_present[flight_id]=True
            plt.scatter(df[df['flightid_iagos']==flight_id]['time OB'],\
                        df[df['flightid_iagos']==flight_id]['time_iagos'],\
                        color=colors[flight_id],zorder=priorities[flight_id])
#--legend
for flight_id in flight_ids:
   if flight_id_present[flight_id]:
      plt.scatter([],[],color=colors[flight_id],label=flight_id)
#--execute plot
plt.legend(fontsize=20)
plt.xlim(2.5,12)
plt.ylim(2.5,12)
plt.xlabel('Optimised flight time (hours)',fontsize=20)
plt.ylabel('Actual IAGOS flight time (hours)',fontsize=20)
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot([0.1,15],[0.1,15],color='black',linestyle='solid',linewidth=1)
plt.plot([0.1,15],[0.1*1.01,15*1.01],color='black',linestyle='solid',linewidth=0.5)
plt.plot([0.1,15],[0.1*1.02,15*1.02],color='black',linestyle='solid',linewidth=0.5)
plt.plot([0.1,15],[0.1*1.05,15*1.05],color='black',linestyle='solid',linewidth=0.5)
plt.plot([0.1,15],[0.1*1.10,15*1.10],color='black',linestyle='solid',linewidth=0.5)
plt.plot([0.1,15],[0.1*1.20,15*1.20],color='black',linestyle='solid',linewidth=0.5)
plt.savefig(pathgraph+'SCATTER_PLOT_by_aircraft_'+str(yr)+'.png')
if pltshow: plt.show()
plt.close()
#
#========================
print('Total number of flights=',np.sum(nb_flights))
print('Average elapsed time Gradient descent=',np.sum(nb_flights*time_elapsed_OB)/np.sum(nb_flights))
print('Average elapsed time Zermelo method=',np.sum(nb_flights*time_elapsed_EG)/np.sum(nb_flights))
print('Average success rate Zermelo method=',np.sum(nb_flights*success_rate_EG)/np.sum(nb_flights))
ratio_iagos_quick=time_iagos/time_quick
ratio_quick_short=time_quick/time_short
ratio_short_nownd=time_short/time_nownd
ratio_iagos_nownd=time_iagos/time_nownd
print('Average Min Max ratio IAGOS to quickest=',np.sum(time_iagos)/np.sum(time_quick),np.nanmin(ratio_iagos_quick),np.nanmax(ratio_iagos_quick))
print('Average Min Max ratio quickest to shortest=',np.sum(time_quick)/np.sum(time_short),np.nanmin(ratio_quick_short),np.nanmax(ratio_quick_short))
print('Average Min Max ratio shortest to shortest no wind=',np.sum(time_short)/np.sum(time_nownd),np.nanmin(ratio_short_nownd),np.nanmax(ratio_short_nownd))
print('==> Average Min Max ratio IAGOS to shortest no wind=',np.sum(time_iagos)/np.sum(time_nownd),np.nanmin(ratio_iagos_nownd),np.nanmax(ratio_iagos_nownd),'\n')
#
#=============
#--latex table 
#=============
dfout = dfout.sort_values(by=['Year','Nb flights'],ascending=False)
dfout['Year'] = dfout['Year'].astype('int')
dfout['Nb flights'] = dfout['Nb flights'].astype('int')
dfout = dfout.style.format({'Year':'{:4d}','Nb flights':'{:4d}'},precision=3)
print(dfout.to_latex())
with open('mytable'+str(yr)+'.tex', 'w') as tf:
   tf.write(dfout.to_latex())
#========================
