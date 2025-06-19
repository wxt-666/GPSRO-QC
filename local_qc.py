import netCDF4 as nc
import numpy as np
import gc
import os
import math
import heapq
import glob
import scipy.signal
from eofs.standard import Eof
from netCDF4 import Dataset
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import scipy.stats as stats
import xarray as xr
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,FixedLocator
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from cartopy.util import add_cyclic_point


month=str('789')


# 3 months
# 7
# loc ih ba
# path_ba_ih1=(r'D:/DA/GPS/789/2023_07_atm_ba.nc')
path_ba_ih1=(r'D:/DA/GPS/789/cosmic2_202307_locate_ba_ih_pres_msl_time_trh.nc')
ba_ih1=Dataset(path_ba_ih1,format='netCDF4')
lat1=np.array(ba_ih1.variables['lat'][:])
lon1=np.array(ba_ih1.variables['lon'][:])
ih1=np.array(ba_ih1.variables['ih'][:])
ba1=np.array(ba_ih1.variables['ba'][:])
msl1=np.array(ba_ih1.variables['msl'][:])
ih1[ih1==-999.0]=np.NaN
msl1[msl1==-999.0]=np.NaN
ba1[ba1==-999.0]=np.NaN
ba1[ba1<0]=np.NaN
ba1[ba1>0.1]=np.NaN
# trh
# trh_wmo1=np.array(ba_ih1.variables['trh_wmo'][:])
# time
# day1=np.array(ba_ih1.variables['day'][:])
# hour1=np.array(ba_ih1.variables['hour'][:])
# minute1=np.array(ba_ih1.variables['minute'][:])
# time_utc1=hour1+minute1/60
# time_lst1=time_utc1+lon1/15
# day_lst1=day1
# for i in range(0,np.size(time_lst1)):
#  if time_lst1[i]>24:
#   time_lst1[i]=time_lst1[i]-24
#   day_lst1[i]=day_lst1[i]+1
#  if time_lst1[i]<0:
#   time_lst1[i]=time_lst1[i]+24
#   day_lst1[i]=day_lst1[i]-1
# del time_utc1
# omega1=15*(time_lst1-12)
# eptro1=2*np.pi*(day_lst1-1)/365
# gama1=np.ones((np.size(eptro1)))*(0.006918)-0.399912*np.cos(eptro1)+0.070257*np.sin(eptro1)-0.006758*np.cos(2*eptro1)+0.000907*np.sin(2*eptro1)-0.002697*np.cos(3*eptro1)+0.00148*np.sin(3*eptro1)
# soz1=np.ones((np.size(lat1)))*(90)-np.rad2deg(np.arcsin(np.sin(np.deg2rad(lat1))*np.sin(np.deg2rad(gama1))+np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(gama1))*np.cos(np.deg2rad(omega1))))
# del time_lst1
# del day_lst1
# del omega1
# del eptro1
# del gama1
# 8
# loc ih ba
# path_ba_ih2=(r'D:/DA/GPS/789/2023_08_atm_ba.nc')
path_ba_ih2=(r'D:/DA/GPS/789/cosmic2_202308_locate_ba_ih_pres_msl_time_trh.nc')
ba_ih2=Dataset(path_ba_ih2,format='netCDF4')
lat2=np.array(ba_ih2.variables['lat'][:])
lon2=np.array(ba_ih2.variables['lon'][:])
ih2=np.array(ba_ih2.variables['ih'][:])
ba2=np.array(ba_ih2.variables['ba'][:])
msl2=np.array(ba_ih2.variables['msl'][:])
ih2[ih2==-999.0]=np.NaN
msl2[msl2==-999.0]=np.NaN
ba2[ba2==-999.0]=np.NaN
ba2[ba2<0]=np.NaN
ba2[ba2>0.2]=np.NaN
# trh
# trh_wmo2=np.array(ba_ih2.variables['trh_wmo'][:])
# time
# day2=np.array(ba_ih2.variables['day'][:])
# hour2=np.array(ba_ih2.variables['hour'][:])
# minute2=np.array(ba_ih2.variables['minute'][:])
# time_utc2=hour2+minute2/60
# time_lst2=time_utc2+lon2/15
# day_lst2=day2
# for i in range(0,np.size(time_lst2)):
#  if time_lst2[i]>24:
#   time_lst2[i]=time_lst2[i]-24
#   day_lst2[i]=day_lst2[i]+1
#  if time_lst2[i]<0:
#   time_lst2[i]=time_lst2[i]+24
#   day_lst2[i]=day_lst2[i]-1
# del time_utc2
# omega2=15*(time_lst2-12)
# eptro2=2*np.pi*(day_lst2-1)/365
# gama2=np.ones((np.size(eptro2)))*(0.006918)-0.399912*np.cos(eptro2)+0.070257*np.sin(eptro2)-0.006758*np.cos(2*eptro2)+0.000907*np.sin(2*eptro2)-0.002697*np.cos(3*eptro2)+0.00148*np.sin(3*eptro2)
# soz2=np.ones((np.size(lat2)))*(90)-np.rad2deg(np.arcsin(np.sin(np.deg2rad(lat2))*np.sin(np.deg2rad(gama2))+np.cos(np.deg2rad(lat2))*np.cos(np.deg2rad(gama2))*np.cos(np.deg2rad(omega2))))
# del time_lst2
# del day_lst2
# del omega2
# del eptro2
# del gama2
# 9
# loc ih ba
# path_ba_ih3=(r'D:/DA/GPS/789/2023_09_atm_ba.nc')
path_ba_ih3=(r'D:/DA/GPS/789/cosmic2_202309_locate_ba_ih_pres_msl_time_trh.nc')
ba_ih3=Dataset(path_ba_ih3,format='netCDF4')
lat3=np.array(ba_ih3.variables['lat'][:])
lon3=np.array(ba_ih3.variables['lon'][:])
ih3=np.array(ba_ih3.variables['ih'][:])
ba3=np.array(ba_ih3.variables['ba'][:])
msl3=np.array(ba_ih3.variables['msl'][:])
ih3[ih3==-999.0]=np.NaN
msl3[msl3==-999.0]=np.NaN
ba3[ba3==-999.0]=np.NaN
ba3[ba3<0]=np.NaN
ba3[ba3>0.2]=np.NaN
# trh
# trh_wmo3=np.array(ba_ih3.variables['trh_wmo'][:])
# time
# day3=np.array(ba_ih3.variables['day'][:])
# hour3=np.array(ba_ih3.variables['hour'][:])
# minute3=np.array(ba_ih3.variables['minute'][:])
# time_utc3=hour3+minute3/60
# time_lst3=time_utc3+lon3/15
# day_lst3=day3
# for i in range(0,np.size(time_lst3)):
#  if time_lst3[i]>24:
#   time_lst3[i]=time_lst3[i]-24
#   day_lst3[i]=day_lst3[i]-1
#  if time_lst3[i]<0:
#   time_lst3[i]=time_lst3[i]+24
#   day_lst3[i]=day_lst3[i]-1
# del time_utc3 
# omega3=15*(time_lst3-12)
# eptro3=2*np.pi*(day_lst3-1)/365
# gama3=np.ones((np.size(eptro3)))*(0.006918)-0.399912*np.cos(eptro3)+0.070257*np.sin(eptro3)-0.006758*np.cos(2*eptro3)+0.000907*np.sin(2*eptro3)-0.002697*np.cos(3*eptro3)+0.00148*np.sin(3*eptro3)
# soz3=np.ones((np.size(lat3)))*(90)-np.rad2deg(np.arcsin(np.sin(np.deg2rad(lat3))*np.sin(np.deg2rad(gama3))+np.cos(np.deg2rad(lat3))*np.cos(np.deg2rad(gama3))*np.cos(np.deg2rad(omega3))))
# del time_lst3
# del day_lst3
# del omega3
# del eptro3
# del gama3
# total lon
lat=np.concatenate([lat1.flatten(),lat2.flatten(),lat3.flatten()])
lon=np.concatenate([lon1.flatten(),lon2.flatten(),lon3.flatten()])
# soz=np.concatenate([soz1.flatten(),soz2.flatten(),soz3.flatten()])
# del lat1
# del lat2
# del lat3
# del lon1
# del lon2
# del lon3
# del soz1
# del soz2
# del soz3
# total trh
# trh_wmo=np.concatenate([trh_wmo1.flatten(),trh_wmo2.flatten(),trh_wmo3.flatten()])
# del trh_wmo1
# del trh_wmo2
# del trh_wmo3
# ba
ba=np.ones((int(np.size(ba1,0)+np.size(ba2,0)+np.size(ba3,0)),int(np.max(np.array([np.size(ba1,1),np.size(ba2,1),np.size(ba3,1)])))))*(-999.0)
ba[0:np.size(ba1,0),0:np.size(ba1,1)]=np.array(ba1)
ba[np.size(ba1,0):int(np.size(ba1,0)+np.size(ba2,0)),0:np.size(ba2,1)]=np.array(ba2)
ba[int(np.size(ba1,0)+np.size(ba2,0)):,0:np.size(ba3,1)]=np.array(ba3)
del ba1
del ba2
del ba3
# ih
ih=np.ones((int(np.size(ih1,0)+np.size(ih2,0)+np.size(ih3,0)),int(np.max(np.array([np.size(ih1,1),np.size(ih2,1),np.size(ih3,1)])))))*(-999.0)
ih[0:np.size(ih1,0),0:np.size(ih1,1)]=np.array(ih1)
ih[np.size(ih1,0):int(np.size(ih1,0)+np.size(ih2,0)),0:np.size(ih2,1)]=np.array(ih2)
ih[int(np.size(ih1,0)+np.size(ih2,0)):,0:np.size(ih3,1)]=np.array(ih3)
del ih1
del ih2
del ih3
# msl
msl=np.ones((int(np.size(msl1,0)+np.size(msl2,0)+np.size(msl3,0)),int(np.max(np.array([np.size(msl1,1),np.size(msl2,1),np.size(msl3,1)])))))*(-999.0)
msl[0:np.size(msl1,0),0:np.size(msl1,1)]=np.array(msl1)
msl[np.size(msl1,0):int(np.size(msl1,0)+np.size(msl2,0)),0:np.size(msl2,1)]=np.array(msl2)
msl[int(np.size(msl1,0)+np.size(msl2,0)):,0:np.size(msl3,1)]=np.array(msl3)
del msl1
del msl2
del msl3

gc.collect()

# lat / ih levels
h_interval=0.1
lat_interval=2.5
lon_interval=2.5

# ih
ih_level=np.arange(2,20.01,h_interval)
ih_median=np.array([(ih_level[i]+ih_level[i+1])/2 for i in range(0,(np.size(ih_level)-1))])

# msl
# msl ih convert
msl_level=np.zeros((np.size(ih_level)))
for i in range(0,np.size(ih_level)):
 msl_total=msl[np.where(np.logical_and(ih>=(ih_level[i]-h_interval/2),ih<(ih_level[i]+h_interval/2)))]
 msl_level[i]=np.nanmean(msl_total)
msl_median=np.array([(msl_level[i]+msl_level[i+1])/2 for i in range(0,(np.size(msl_level)-1))])

# lat
lat_level=np.arange(-45,45.1,lat_interval)
lat_median=np.array([(lat_level[i]+lat_level[i+1])/2 for i in range(0,(np.size(lat_level)-1))])

# lon
lon_level=np.arange(-180,180.1,lon_interval)
lon_median=np.array([(lon_level[i]+lon_level[i+1])/2 for i in range(0,(np.size(lon_level)-1))])

# lon count
lon_count=np.array([np.size(lon[np.where(np.logical_and(lon>=lon_level[i],lon<lon_level[i+1]))]) for i in range(0,np.size(lon_median))])

# lat count
lat_count=np.array([np.size(lat[np.where(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]))]) for i in range(0,np.size(lat_median))])




# lat, lon 2.5 qc
lon_count_path=(str(str('D:/DA/GPS/789/cosmic2_')+month+str('_lat_lon_ba_bm_bsd.nc')))
lon_count_file=Dataset(lon_count_path,format='netCDF4')
lat_lon_count=np.array(lon_count_file.variables['count'][:])
lon_ba_bm=np.array(lon_count_file.variables['ba_bm'][:])
lon_ba_bsd=np.array(lon_count_file.variables['ba_bsd'][:])
lon_ba_bm[lon_ba_bm==-999.0]=np.NaN
lon_ba_bsd[lon_ba_bsd==-999.0]=np.NaN


z_threshold=2.5
# before
z_score_before=np.array(np.ones((np.size(ih_median),np.size(lat_median),np.size(lon_median)))*(-999.0),dtype='object')
z_mean_before=np.ones((np.size(ih_median),np.size(lat_median),np.size(lon_median)))*(-999.0)
z_std_before=np.ones((np.size(ih_median),np.size(lat_median),np.size(lon_median)))*(-999.0)
# z_mean_after=np.ones((np.size(ih_median),np.size(lat_median),np.size(lon_median)))*(-999.0)
# z_std_after=np.ones((np.size(ih_median),np.size(lat_median),np.size(lon_median)))*(-999.0)
z_above_count_before=np.zeros((np.size(ih_median),np.size(lat_median),np.size(lon_median)))
z_below_count_before=np.zeros((np.size(ih_median),np.size(lat_median),np.size(lon_median)))
z_each_prf=np.zeros((np.size(ba,0),np.size(ba,1)))
# after
# bm_after=np.ones((np.size(ih_median),np.size(lat_median),np.size(lon_median)))*(-999.0)
# bsd_after=np.ones((np.size(ih_median),np.size(lat_median),np.size(lon_median)))*(-999.0)
# z_score_after=np.array(np.ones((np.size(ih_median),np.size(lat_median),np.size(lon_median)))*(-999.0),dtype='object')
# z_mean_after=np.ones((np.size(ih_median),np.size(lat_median),np.size(lon_median)))*(-999.0)
# z_std_after=np.ones((np.size(ih_median),np.size(lat_median),np.size(lon_median)))*(-999.0)
# z_above_count_after=np.zeros((np.size(ih_median),np.size(lat_median),np.size(lon_median)))
# z_below_count_after=np.zeros((np.size(ih_median),np.size(lat_median),np.size(lon_median)))
# lon
lon_total_before=np.array(np.zeros((np.size(ih_median),np.size(lat_median),np.size(lon_median))),dtype='object')
lon_ih=np.zeros((np.size(lon),np.size(ih,1)))
for i in range(0,np.size(lon)):
 lon_ih[i,:]=np.ones((np.size(ih,1)))*(lon[i])
# lat
lat_total_before=np.array(np.zeros((np.size(ih_median),np.size(lat_median),np.size(lon_median))),dtype='object')
lat_ih=np.zeros((np.size(lat),np.size(ih,1)))
for i in range(0,np.size(lat)):
 lat_ih[i,:]=np.ones((np.size(ih,1)))*(lat[i])



for i in range(0,np.size(lat_median)):
 for j in range(0,np.size(lon_median)):
  loc_index=np.array(np.where(np.logical_and(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]),np.logical_and(lon>=lon_level[j],lon<lon_level[j+1])))[0])
  ih_loc=np.array(np.array(ih[np.where(np.logical_and(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]),np.logical_and(lon>=lon_level[j],lon<lon_level[j+1]))),:])[0,:,:])
  ba_loc=np.array(np.array(ba[np.where(np.logical_and(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]),np.logical_and(lon>=lon_level[j],lon<lon_level[j+1]))),:])[0,:,:])
  lon_loc=np.array(np.array(lon_ih[np.where(np.logical_and(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]),np.logical_and(lon>=lon_level[j],lon<lon_level[j+1]))),:])[0,:,:])
  lat_loc=np.array(np.array(lat_ih[np.where(np.logical_and(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]),np.logical_and(lon>=lon_level[j],lon<lon_level[j+1]))),:])[0,:,:])
  for k in range(0,np.size(ih_median)):
   # before
   h_loc_index=np.array(np.where(np.logical_and(ih_loc>=ih_level[k],ih_loc<ih_level[k+1]))[0])
   row_index=np.array([int(loc_index[int(h_loc_index[l])]) for l in range(0,np.size(h_loc_index))])
   del h_loc_index
   col_index=np.array(np.where(np.logical_and(ih_loc>=ih_level[k],ih_loc<ih_level[k+1]))[1])
   ba_loc_single_before=np.array(ba_loc[np.where(np.logical_and(ih_loc>=ih_level[k],ih_loc<ih_level[k+1]))])
   lon_loc_single_before=np.array(lon_loc[np.where(np.logical_and(ih_loc>=ih_level[k],ih_loc<ih_level[k+1]))])
   lat_loc_single_before=np.array(lat_loc[np.where(np.logical_and(ih_loc>=ih_level[k],ih_loc<ih_level[k+1]))])
   ba_loc_z_before=abs((ba_loc_single_before-lon_ba_bm[k,i,j])/lon_ba_bsd[k,i,j])
   m=np.nanmedian(ba_loc_single_before)
   mad=np.array(abs(ba_loc_single_before-m))
   # weight
   w=(ba_loc_single_before-m)/(7.5*mad)
   w[abs(w)>1]=1
   # biweight mean/std
   w_cal1=np.zeros((np.size(np.array(ba_loc_single_before))))
   for l in range(0,np.size(np.array(ba_loc_single_before))):
    w_cal1[l]=(ba_loc_single_before[l]-m)*((1-(w[l])**2)**2)
   w_cal2=((1-w)**2)**2
   w_cal3=w_cal1**2
   w_cal4=w_cal2*(1-5*(w**2))
   # mean
   ba_bm=m+(np.nansum(w_cal1)/np.nansum(w_cal2))
   # std
   ba_bsd=np.sqrt(np.size(np.array(ba_loc_single_before))*np.nansum(w_cal3))/abs(np.nansum(w_cal4))
   # lon
   lon_total_before[k,i,j]=lon_loc_single_before
   # lat
   lat_total_before[k,i,j]=lat_loc_single_before
   # z score
   z_score_before[k,i,j]=abs((ba_loc_single_before-ba_bm)/ba_bsd)
   # z_mean_before[k,i,j]=np.nanmean(np.array(z_score_before[k,i,j]))
   # z_std_before[k,i,j]=np.nanstd(np.array(z_score_before[k,i,j]))
   z_above_count_before[k,i,j]=np.size(np.array(np.array(z_score_before[k,i,j])[np.array(z_score_before[k,i,j])>z_threshold]))
   # z_below_count_before[k,i,j]=np.size(np.array(np.array(z_score_before[k,i,j])[np.array(z_score_before[k,i,j])<=z_threshold]))
   # z_mean_after[k,i,j]=np.nanmean(np.array(np.array(z_score_before[k,i,j])[np.array(z_score_before[k,i,j])<=z_threshold]))
   # z_std_after[k,i,j]=np.nanstd(np.array(np.array(z_score_before[k,i,j])[np.array(z_score_before[k,i,j])<=z_threshold]))
  #  for m in range(0,np.size(col_index)):
  #   z_each_prf[int(row_index[m]),int(col_index[m])]=ba_loc_z_before[m]
   # after
   # ba_loc_single_after=np.array(ba_loc_single_before[np.where(np.array(z_score_before[k,i,j])<=z_threshold)])
   # m_after=np.nanmedian(ba_loc_single_after)
   # mad_after=np.array(abs(ba_loc_single_after-m_after))
   # # weight
   # w_after=(ba_loc_single_after-m_after)/(7.5*mad_after)
   # w_after[abs(w_after)>1]=1
   # # biweight mean/std
   # w_cal1_after=np.zeros((np.size(np.array(ba_loc_single_after))))
   # for l in range(0,np.size(np.array(ba_loc_single_after))):
   #  w_cal1_after[l]=(ba_loc_single_after[l]-m)*((1-(w_after[l])**2)**2)
   # w_cal2_after=((1-w_after)**2)**2
   # w_cal3_after=w_cal1_after**2
   # w_cal4_after=w_cal2_after*(1-5*(w_after**2))
   # # mean
   # bm_after[k,i,j]=m_after+(np.nansum(w_cal1_after)/np.nansum(w_cal2_after))
   # # std
   # bsd_after[k,i,j]=np.sqrt(np.size(np.array(ba_loc_single_after))*np.nansum(w_cal3_after))/abs(np.nansum(w_cal4_after))
   # # z score
   # z_score_after[k,i,j]=abs((ba_loc_single_after-bm_after[k,i,j])/bsd_after[k,i,j])
   # z_mean_after[k,i,j]=np.nanmean(np.array(z_score_after[k,i,j]))
   # z_std_after[k,i,j]=np.nanstd(np.array(z_score_after[k,i,j]))
   # z_above_count_after[k,i,j]=np.size(np.array(np.array(z_score_after[k,i,j])[np.array(z_score_after[k,i,j])>z_threshold]))
   # z_below_count_after[k,i,j]=np.size(np.array(np.array(z_score_after[k,i,j])[np.array(z_score_after[k,i,j])<=z_threshold]))

# z_each_prf[np.where(np.isnan(ba))]=np.NaN

# z-score nc file
# save_path=(r'D:/DA/GPS/789/cosmic2_local_z_score2.nc')
# nc_file=Dataset(str(save_path),'w',format='NETCDF4')
# nc_file.createDimension('prf_number',np.size(ba,0))
# nc_file.createDimension('ih_level',np.size(ba,1))
# nc_file.createVariable('z_score',np.float64,('prf_number','ih_level'))
# nc_file.variables['z_score'][:]=z_each_prf
# nc_file.close()

ih_select=8
# count_index=int(np.where(abs(ih_level-ih_select)==np.min(abs(ih_level-ih_select)))[0])
count_index=int(np.array([i for i in range(0,np.size(ih_median)) if ih_median[i]<=ih_select and ih_median[i+1]>=ih_select]))
# count/location
count_select=np.array(lat_lon_count[count_index,:,:])
cyclic_count,cyclic_lon=add_cyclic_point(count_select,coord=lon_median)
cyclic_lon,cyclic_lat=np.meshgrid(cyclic_lon,lat_median)
# # before
lon_before_select=np.array(lon_total_before[count_index,:,:])
# lat_before_select=np.array(lat_total_before[count_index,:,:])
z_before_select=np.array(z_score_before[count_index,:,:])
# bm_before_select=np.array(lon_ba_bm[count_index,:,:])
# bsd_before_select=np.array(lon_ba_bsd[count_index,:,:])
# z_mean_before_select=np.array(z_mean_before[count_index,:,:])
# z_std_before_select=np.array(z_std_before[count_index,:,:])
# z_above_before_select=np.array(z_above_count_before[count_index,:,:])
# z_below_before_select=np.array(z_below_count_before[count_index,:,:])
# cyclic_bm_before,_=add_cyclic_point(bm_before_select,coord=lon_median)
# cyclic_bsd_before,_=add_cyclic_point(bsd_before_select,coord=lon_median)
# cyclic_zmean_before,_=add_cyclic_point(z_mean_before_select,coord=lon_median)
# cyclic_zstd_before,_=add_cyclic_point(z_std_before_select,coord=lon_median)
# cyclic_zabove_before,_=add_cyclic_point(z_above_before_select,coord=lon_median)
# cyclic_zbelow_before,_=add_cyclic_point(z_below_before_select,coord=lon_median)
# after
# bm_after_select=np.array(bm_after[count_index,:,:])
# bsd_after_select=np.array(bsd_after[count_index,:,:])
# z_mean_after_select=np.array(z_mean_after[count_index,:,:])
# z_std_after_select=np.array(z_std_after[count_index,:,:])
# z_above_after_select=np.array(z_above_count_after[count_index,:,:])
# z_below_after_select=np.array(z_below_count_after[count_index,:,:])
# cyclic_bm_after,_=add_cyclic_point(bm_after_select,coord=lon_median)
# cyclic_bsd_after,_=add_cyclic_point(bsd_after_select,coord=lon_median)
# cyclic_zmean_after,_=add_cyclic_point(z_mean_after_select,coord=lon_median)
# cyclic_zstd_after,_=add_cyclic_point(z_std_after_select,coord=lon_median)
# cyclic_zabove_after,_=add_cyclic_point(z_above_after_select,coord=lon_median)
# cyclic_zbelow_after,_=add_cyclic_point(z_below_after_select,coord=lon_median)


""" # single height z below count 360 before qc
col_num=11
below_level=np.linspace(100,1200,(col_num+1),endpoint=True)
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(below_level,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zbelow_before,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(25/5)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.0f',ticks=below_level)
cbar.set_ticks(below_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km_regional/')+str(month)+str('_')+str(ih_select)+str('_below_count.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z below count 360 after qc
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-35,35],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(30,470,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zbelow_after,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(25/5)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.0f',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/11.4/')+str('cosmic2_2023')+month+str('_single height_lat_zbelow_after_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z above count 360 before qc
z_threshold_diff=2.5
z_above_diff_before_select=np.zeros((np.size(lat_median),np.size(lon_median)))
for i in range(0,np.size(lat_median)):
 for j in range(0,np.size(lon_median)):
  z_above_diff_before_select[i,j]=np.size(np.array(np.array(z_before_select[i,j])[np.array(z_before_select[i,j])>=z_threshold_diff]))
cyclic_zabove_diff_before,_=add_cyclic_point(z_above_diff_before_select,coord=lon_median)
col_num=11
above_level=np.linspace(10,43,(col_num+1),endpoint=True)
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(above_level,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zabove_diff_before,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# quiver
# arrow=mpatches.ArrowStyle('-|>',head_length=0.5,head_width=0.2)
# ax1.streamplot(cyclic_lon_era,cyclic_lat_era,cyclic_u_hourly_mean,cyclic_v_hourly_mean,color='black',density=0.6,linewidth=0.5,arrowsize=0.6,arrowstyle=arrow,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(15/3)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.0f',ticks=above_level)
cbar.set_ticks(above_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/article2/')+str(ih_select)+str('km_regional/z_')+str(z_threshold_diff)+str('/')+str(month)+str('_')+str(ih_select)+str('_above_count.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z below count 360 after qc
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-35,35],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(3,14,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zabove_after,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(25/5)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.0f',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/11.4/')+str('cosmic2_2023')+month+str('_single height_lat_zabove_after_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height bm 360 before qc
col_num=11
bm_level=np.linspace(9,11.2,(col_num+1),endpoint=True)
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(bm_level,cbar_map.N,extend='both')
cyclic_bm_before_mask=np.nan_to_num(cyclic_bm_before,nan=np.nanmean(cyclic_bm_before))
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_bm_before_mask*(10**3),cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(25/5)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.1f',ticks=bm_level)
cbar.set_ticks(bm_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
# units
ax3=fig.add_axes([0.823,0.268,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-3}}$ rad)',family='Times New Roman',fontsize=8,color='black')
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km_regional/')+str(month)+str('_')+str(ih_select)+str('_bm.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height bsd 360 before qc
col_num=11
bsd_level=np.linspace(2,13,(col_num+1),endpoint=True)
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(bsd_level,cbar_map.N,extend='both')
cyclic_bsd_before_mask=np.nan_to_num(cyclic_bsd_before,nan=np.nanmean(cyclic_bm_before))
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_bsd_before_mask*(10**4),cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(25/5)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.0f',ticks=bsd_level)
cbar.set_ticks(bsd_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
# units
ax3=fig.add_axes([0.823,0.268,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-4}}$ rad)',family='Times New Roman',fontsize=8,color='black')
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km_regional/')+str(month)+str('_')+str(ih_select)+str('_bsd.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single lat z-score
lat_single_min=29
lat_single_max=32
lon360_level=np.arange(0,360.1,2.5)
lon360_median=np.array([(lon360_level[i]+lon360_level[i+1])/2 for i in range(0,int(np.size(lon360_level)-1))])
z_before_one_lat=z_before_select[np.where(np.logical_and(lat_median<lat_single_max,lat_median>lat_single_min))][0]
zmean_one_lat=np.array([np.nanmean(np.array(z_before_one_lat[i])) for i in range(0,np.size(z_before_one_lat))])
zstd_one_lat=np.array([np.nanstd(np.array(z_before_one_lat[i])) for i in range(0,np.size(z_before_one_lat))])
lon_before_one_lat=lon_before_select[np.where(np.logical_and(lat_median<lat_single_max,lat_median>lat_single_min))][0]
# single lat z-score scatter figure
fig=plt.figure(figsize=(6,3),dpi=600)
ax1=fig.add_subplot(1,1,1)
# z-level count
z_level=np.arange(0,12.01,0.1)
z_level_median=np.array([(z_level[i]+z_level[i+1])/2 for i in range(0,int(np.size(z_level)-1))])
z_level_count=np.zeros((np.size(z_level_median),np.size(lon360_median)))
lon_before_one_lat_singlez=[]
z_before_one_lat_singlez=[]
for i in range(0,np.size(z_level_median)):
 for j in range(0,np.size(lon360_median)):
  z_before_one_lat_lon=np.array(z_before_one_lat[j])
  lon_before_one_lat_lon=np.array(lon_before_one_lat[j])
  z_level_count[i,j]=np.size(z_before_one_lat_lon[np.where(np.logical_and(z_before_one_lat_lon>=z_level[i],z_before_one_lat_lon<z_level[i+1]))])
  if z_level_count[i,j]==1.0:
   lon_before_one_lat_singlez.append(lon_before_one_lat_lon[np.where(np.logical_and(z_before_one_lat_lon>=z_level[i],z_before_one_lat_lon<z_level[i+1]))])
   z_before_one_lat_singlez.append(z_before_one_lat_lon[np.where(np.logical_and(z_before_one_lat_lon>=z_level[i],z_before_one_lat_lon<z_level[i+1]))])
lon_before_one_lat_singlez=np.array(lon_before_one_lat_singlez)
lon_before_one_lat_singlez[np.where(lon_before_one_lat_singlez<0)]=lon_before_one_lat_singlez[np.where(lon_before_one_lat_singlez<0)]+360
z_before_one_lat_singlez=np.array(z_before_one_lat_singlez)
# color
col_num=11
# col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_list=["#EAE3C7","#C8E1B7","#AED7B7","#83CDAC","#58BEA6","#12B7A9","#00B7B6","#01B4CA","#2094c7","#697FBB","#8366A9","#92509A","#A93391"]
z_level_count_level=np.linspace(5,104,(col_num+1),endpoint=True)
z_level_count_level=np.concatenate([np.array([2]).flatten(),z_level_count_level.flatten()])
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mcolors.BoundaryNorm(z_level_count_level,cbar_map.N,extend='max')
z_count_mask=np.ma.masked_where((z_level_count<2),z_level_count)
# mesh
ax1.pcolormesh(lon360_median,z_level_median,z_count_mask,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0)
# scatter
# for i in range(0,np.size(z_before_one_lat)):
#  lon_before_one_lat_trans=lon_before_one_lat[i]
#  lon_before_one_lat_trans[lon_before_one_lat_trans<0]=lon_before_one_lat_trans[lon_before_one_lat_trans<0]+360
ax1.scatter(lon_before_one_lat_singlez[np.where(z_before_one_lat_singlez>np.percentile(z_before_one_lat_singlez,60))],z_before_one_lat_singlez[np.where(z_before_one_lat_singlez>np.percentile(z_before_one_lat_singlez,60))],marker='o',c='none',edgecolors='#000000',linewidths=0.4,s=6,zorder=0)
# mean std
ax1.plot(lon360_median,zmean_one_lat,color='#000000',linewidth=0.8,linestyle='-',zorder=1,label=str('\u03bc'+'$_{\mathregular{Z}}$'))
# ax1.plot(lon360_median,zmean_one_lat,color='#000000',linewidth=0.8,linestyle='-',zorder=1,label='$\overline{\mathregular{Z}}$')
# '\u03bc'+'$_{\mathregular{\u03B1}}$
ax1.plot(lon360_median,zstd_one_lat,color='#000000',linewidth=0.8,linestyle='--',zorder=1,label=str(str(chr(963))+str('$_{\mathregular{Z}}$')))
font_legend={'size':9,'family':'Times New Roman','weight':'light'}
ax1.legend(loc='upper right',frameon=False,facecolor='none',prop=font_legend,ncol=1,labelspacing=0.5,columnspacing=1,borderaxespad=0.2) 
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel(str('Z ('+str(chr(966))+', '+str(chr(955))+')'),fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([0,360])
ax1.set_ylim([0,12])
xticks1=np.arange(0,(360+60/10),60)
yticks1=np.arange(0,(12+2/10),2)
# 刻度
yticks1_label=[str(int(yticks1[t])) for t in range(0,np.size(yticks1))]
ax1.set_xticks(xticks1)
# plt.xticks(fontproperties='Times New Roman',fontsize=10)
# ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
xticks1_label=['0${\mathregular{\u00B0}}$E','60${\mathregular{\u00B0}}$E','120${\mathregular{\u00B0}}$E','180${\mathregular{\u00B0}}$','120${\mathregular{\u00B0}}$W','60${\mathregular{\u00B0}}$W','0${\mathregular{\u00B0}}$W']
labels_x=ax1.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_x]
ax1.set_xticklabels(xticks1_label)
ax1.tick_params(axis="x",direction="out",length=4,labelsize=10)
xminorlocator=MultipleLocator(60/3)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.set_yticks(yticks1)
labels_y=ax1.yaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_y]
ax1.set_yticklabels(yticks1_label)
ax1.tick_params(axis="y",direction="out",length=4,labelsize=10)
yminorlocator1=MultipleLocator(((0.5)))
ax1.yaxis.set_minor_locator(yminorlocator1)
ax1.tick_params(axis="x",direction="out",which="minor",length=2)
ax1.tick_params(axis="y",direction="out",which="minor",length=2)
# twin x
# ax2=ax1.twinx()
# ax2.set_ylim([0,2])
# ax2.plot(lon360_median,zmean_one_lat,color='#0000FF',linewidth=0.6,linestyle='-',zorder=1,label='$\overline{\mathregular{Z}}$')
# ax2.plot(lon360_median,zstd_one_lat,color='#FF0000',linewidth=0.6,linestyle='-',zorder=1,label=str(str(chr(963))+str('$_{\mathregular{Z}}$')))
# font_legend={'size':9,'family':'Times New Roman','weight':'light'}
# ax2.legend(loc='upper right',frameon=False,facecolor='none',prop=font_legend,ncol=1,labelspacing=0.5,columnspacing=1,borderaxespad=0.2) 
# yticks2=np.arange(0,(2+0.5/10),0.5)
# ax2.set_yticks(yticks2)
# labels_y2=ax2.yaxis.get_ticklabels()
# [label.set_fontname('Times New Roman') for label in labels_y2]
# yticks2_label=['{:.1f}'.format(yticks2[t]) for t in range(0,np.size(yticks2))]
# ax2.set_yticklabels(yticks2_label)
# ax2.tick_params(axis="y",direction="out",length=4,labelsize=10)
# yminorlocator2=MultipleLocator(((0.125)))
# ax2.yaxis.set_minor_locator(yminorlocator2)
# ax2.tick_params(axis="x",direction="out",which="minor",length=2)
# ax2.tick_params(axis="y",direction="out",which="minor",length=2)
# colorbar
# ax2=fig.add_axes([0.2,0.00,0.63,0.02])
# cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.0f',ticks=z_level_count_level)
# cbar.set_ticks(z_level_count_level)
# labels_cbar=cbar.ax.xaxis.get_ticklabels()
# [label.set_fontname('Times New Roman') for label in labels_cbar]
# cbar.ax.tick_params(labelsize=8)
# cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
# cbar.ax.tick_params(length=1.5)
ax2=fig.add_axes([0.91,0.15,0.01,0.7])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,aspect=60,format='%.0f',ticks=z_level_count_level,extend='max')
cbar.set_ticks(z_level_count_level)
labels_cbar=cbar.ax.yaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=True)
cbar.ax.tick_params(length=1.5)
# plt.savefig(str('D:/DA/article/cosmic2_latlon_30N_scatter.png'),dpi=600,bbox_inches='tight',pad_inches=0)
plt.savefig(str(str('D:/DA/publish/MWR/fig/fig8b.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # zmean after QC
z_mean_after_select=np.array(z_mean_after[count_index,:,:])
cyclic_zmean_after,_=add_cyclic_point(z_mean_after_select,coord=lon_median)
col_num=11
zmean_level=np.linspace(0.4,0.95,(col_num+1),endpoint=True)
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(zmean_level,cbar_map.N,extend='both')
cyclic_zmean_dis_mask=np.nan_to_num(cyclic_zmean_after,nan=0)
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zmean_dis_mask,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(25/5)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=zmean_level)
cbar.set_ticks(zmean_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km_regional/')+str(month)+str('_')+str(ih_select)+str('_zmean_after.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # zstd after QC
z_std_after_select=np.array(z_std_after[count_index,:,:])
cyclic_zstd_after,_=add_cyclic_point(z_std_after_select,coord=lon_median)
col_num=11
zstd_level_after=np.linspace(0.4,0.62,(col_num+1),endpoint=True)
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(zstd_level_after,cbar_map.N,extend='both')
cyclic_zstd_dis_mask=np.nan_to_num(cyclic_zstd_after,nan=0)
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zstd_dis_mask,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(25/5)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=zstd_level_after)
cbar.set_ticks(zstd_level_after)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km_regional/')+str(month)+str('_')+str(ih_select)+str('_zstd_after2.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # zmean before QC
col_num=11
zmean_level=np.linspace(0.6,0.93,(col_num+1),endpoint=True)
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(zmean_level,cbar_map.N,extend='both')
cyclic_zmean_dis_mask=np.nan_to_num(cyclic_zmean_before,nan=0)
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zmean_dis_mask,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(25/5)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=zmean_level)
cbar.set_ticks(zmean_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km_regional/')+str(month)+str('_')+str(ih_select)+str('_zmean_before.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # zstd before QC
zstd_level_after=np.linspace(0.6,0.93,(col_num+1),endpoint=True)
col_num=11
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(zstd_level_after,cbar_map.N,extend='both')
cyclic_zstd_dis_mask=np.nan_to_num(cyclic_zstd_before,nan=0)
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zstd_dis_mask,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(25/5)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=zstd_level_after)
cbar.set_ticks(zstd_level_after)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km_regional/')+str(month)+str('_')+str(ih_select)+str('_zstd_before.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""



# article figure

""" # z above lat count
z_above_count_lat=np.nansum(z_above_count_before,axis=2)
print(np.max(z_above_count_lat))
z_above_count_lat_height=np.nansum(z_above_count_lat,axis=1)
print(np.max(z_above_count_lat_height))

z_count_level=np.concatenate([np.array([100]).flatten(),np.linspace(200,3500,12,endpoint=True).flatten()])


fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(z_count_level,c_map.N,extend='both')
z_mask=z_above_count_lat
# bm_mask=np.ma.masked_where((ba_bm<(1.5*10**(-3))),ba_bm)
msl_fig=msl_median.copy()
msl_fig[0]=0
msl_fig[-1]=20
ax1.pcolormesh(lat_median,msl_fig,z_mask,cmap=c_map,norm=c_norm,shading='nearest',zorder=0)
# # era w
# p_level_s=p_level[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)]))]
# w_scale=w_select_mean[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)])),:][0,:,:]
# v_scale=v_select_mean[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)])),:][0,:,:]
# w_h=np.array([ih_median[np.where(np.array(abs(p_level_mean-p_level_s[i]))==np.min(np.array(abs(p_level_mean-p_level_s[i]))))] for i in range(0,np.size(p_level_s))]).flatten()
# w_f=ax1.contour(lat_era,w_h,w_scale,w_level,colors='black',linewidths=0.5,zorder=1)
# plt.clabel(w_f,w_level[::1],inline=True,fontsize=6)
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Height (km)',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-45,45])
ax1.set_ylim([0,20])
xticks1=np.arange(-45,(45+15/10),15)
yticks1=np.arange(0,(20+4/10),4)
yticks1_label=[str(int(yticks1[t])) for t in range(0,np.size(yticks1))]
ax1.set_xticks(xticks1)
ax1.xaxis.set_major_formatter(LatitudeFormatter())
plt.xticks(fontproperties='Times New Roman')
ax1.set_yticks(yticks1)
labels_y=ax1.yaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_y]
ax1.set_yticklabels(yticks1_label)
ax1.tick_params(axis="x",direction="out",length=4,labelsize=10)
ax1.tick_params(axis="y",direction="out",length=4,labelsize=10)
xminorlocator1=MultipleLocator(5)
yminorlocator1=MultipleLocator(((4/4)))
ax1.xaxis.set_minor_locator(xminorlocator1)
ax1.yaxis.set_minor_locator(yminorlocator1)
ax1.yaxis.set_minor_formatter(FormatStrFormatter(''))
ax1.tick_params(axis="x",direction="out",which="minor",length=2)
ax1.tick_params(axis="y",direction="out",which="minor",length=2)
# grid
# ax1.grid(which='minor',axis='x',linestyle='--',linewidth=0.4,color='#000000')
# ax1.grid(which='major',axis='x',linestyle='--',linewidth=0.4,color='#000000')
# ax1.grid(which='minor',axis='y',linestyle='--',linewidth=0.4,color='#000000')
# ax1.grid(which='major',axis='y',linestyle='--',linewidth=0.4,color='#000000')
# 双y轴
ax2=ax1.twiny()
# cumulation
ax2.plot(np.array(z_above_count_lat_height)/(10**4),msl_fig,linewidth=0.6,color='#000000',linestyle='-',zorder=2)
# 坐标轴标签
# ax2.set_xlabel(str(str('|Z|$_{\mathregular{\u03B1}}$ > ')+str(z_threshold)+str(' Count')),fontdict=font_label1)
ax2.set_xlabel(str('Outlier (10$^{\mathregular{4}}$)'),fontdict=font_label1)
# 刻度(主+次)
ax2.set_xlim([0,8])
xticks2=np.arange(0,8.1,2)
ax2.set_xticks(xticks2)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
ax2.tick_params(axis="x",direction="out",length=4)
xminorlocator2=MultipleLocator((0.5))
ax2.xaxis.set_minor_locator(xminorlocator2)
ax2.tick_params(axis="x",direction="out",which="minor",length=2)
# p
# ax3=ax2.twinx()
# ax3.set_ylabel(str('Pressure (hPa)'),fontdict=font_label1)
# ax3.set_ylim([2,20])
# yticks2=np.concatenate([np.array(ih_median[ih_median<=1000])[::30].flatten(),np.array(ih_median[-1]).flatten()])
# p_tick=np.concatenate([np.array(p_level_mean[ih_median<=1000])[::30].flatten(),np.array(p_level_mean[-1]).flatten()])
# yticks2_label=['{:.0f}'.format((p_tick[t])) for t in range(0,np.size(p_tick))]
# ax3.set_yticks(yticks2)
# labels_y=ax3.yaxis.get_ticklabels()
# [label.set_fontname('Times New Roman') for label in labels_y]
# ax3.set_yticklabels(yticks2_label)
# ax3.tick_params(axis="y",direction="out",length=4,labelsize=10)
# ax3.yaxis.set_minor_locator(FixedLocator(ih_median[::10]))
# ax3.tick_params(axis="y",direction="out",which="minor",length=2)
# colorbar
ax4=fig.add_axes([0.18,0.03,0.67,0.018])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=c_norm,cmap=c_map),cax=ax4,aspect=60,format='%.0f',ticks=(z_count_level),orientation='horizontal',extend='both')
cbar.set_ticks(z_count_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar_label=[str(int(z_count_level[i]/100)) for i in range(0,np.size(z_count_level))]
cbar.ax.set_xticklabels(cbar_label)
cbar.ax.tick_params(labelsize=7)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
ax3=fig.add_axes([0.85,-0.012,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{2}}$)',family='Times New Roman',fontsize=7,color='black')
plt.savefig(str(str('D:/DA/article2/')+month+str('_zabove.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # z below lat count
z_below_count_lat=np.nansum(z_below_count_before,axis=2)
print(np.max(z_below_count_lat))
z_below_count_lat_height=np.nansum(z_below_count_lat,axis=1)
print(np.max(z_below_count_lat_height))
z_count_level=np.concatenate([np.array([10000]).flatten(),np.array(np.linspace(20000,130000,12,endpoint=True)).flatten()])
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(z_count_level,c_map.N,extend='both')
z_mask=z_below_count_lat
# bm_mask=np.ma.masked_where((ba_bm<(1.5*10**(-3))),ba_bm)
msl_fig=msl_median.copy()
msl_fig[0]=0
msl_fig[-1]=20
ax1.pcolormesh(lat_median,msl_fig,z_mask,cmap=c_map,norm=c_norm,shading='nearest',zorder=0)
# # era w
# p_level_s=p_level[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)]))]
# w_scale=w_select_mean[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)])),:][0,:,:]
# v_scale=v_select_mean[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)])),:][0,:,:]
# w_h=np.array([ih_median[np.where(np.array(abs(p_level_mean-p_level_s[i]))==np.min(np.array(abs(p_level_mean-p_level_s[i]))))] for i in range(0,np.size(p_level_s))]).flatten()
# w_f=ax1.contour(lat_era,w_h,w_scale,w_level,colors='black',linewidths=0.5,zorder=1)
# plt.clabel(w_f,w_level[::1],inline=True,fontsize=6)
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Height (km)',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-45,45])
ax1.set_ylim([0,20])
xticks1=np.arange(-45,(45+15/10),15)
yticks1=np.arange(0,(20+4/10),4)
yticks1_label=[str(int(yticks1[t])) for t in range(0,np.size(yticks1))]
ax1.set_xticks(xticks1)
ax1.xaxis.set_major_formatter(LatitudeFormatter())
plt.xticks(fontproperties='Times New Roman')
ax1.set_yticks(yticks1)
labels_y=ax1.yaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_y]
ax1.set_yticklabels(yticks1_label)
ax1.tick_params(axis="x",direction="out",length=4,labelsize=10)
ax1.tick_params(axis="y",direction="out",length=4,labelsize=10)
xminorlocator1=MultipleLocator(5)
yminorlocator1=MultipleLocator(((4/4)))
ax1.xaxis.set_minor_locator(xminorlocator1)
ax1.yaxis.set_minor_locator(yminorlocator1)
ax1.yaxis.set_minor_formatter(FormatStrFormatter(''))
ax1.tick_params(axis="x",direction="out",which="minor",length=2)
ax1.tick_params(axis="y",direction="out",which="minor",length=2)
# grid
# ax1.grid(which='minor',axis='x',linestyle='--',linewidth=0.4,color='#000000')
# ax1.grid(which='major',axis='x',linestyle='--',linewidth=0.4,color='#000000')
# ax1.grid(which='minor',axis='y',linestyle='--',linewidth=0.4,color='#000000')
# ax1.grid(which='major',axis='y',linestyle='--',linewidth=0.4,color='#000000')
# 双y轴
ax2=ax1.twiny()
# cumulation
ax2.plot(z_below_count_lat_height/(10**5),msl_fig,linewidth=0.6,color='#000000',linestyle='-',zorder=2)
# 坐标轴标签
# ax2.set_xlabel(str(str('|Z|$_{\mathregular{\u03B1}}$ > ')+str(z_threshold)+str(' Count')),fontdict=font_label1)
ax2.set_xlabel(str('Data Count Passed QC (10$^{\mathregular{5}}$)'),fontdict=font_label1)
# 刻度(主+次)
ax2.set_xlim([0,30])
xticks2=np.arange(0,30.1,6)
ax2.set_xticks(xticks2)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
ax2.tick_params(axis="x",direction="out",length=4)
xminorlocator2=MultipleLocator((2))
ax2.xaxis.set_minor_locator(xminorlocator2)
ax2.tick_params(axis="x",direction="out",which="minor",length=2)
# p
# ax3=ax2.twinx()
# ax3.set_ylabel(str('Pressure (hPa)'),fontdict=font_label1)
# ax3.set_ylim([2,20])
# yticks2=np.concatenate([np.array(ih_median[ih_median<=1000])[::30].flatten(),np.array(ih_median[-1]).flatten()])
# p_tick=np.concatenate([np.array(p_level_mean[ih_median<=1000])[::30].flatten(),np.array(p_level_mean[-1]).flatten()])
# yticks2_label=['{:.0f}'.format((p_tick[t])) for t in range(0,np.size(p_tick))]
# ax3.set_yticks(yticks2)
# labels_y=ax3.yaxis.get_ticklabels()
# [label.set_fontname('Times New Roman') for label in labels_y]
# ax3.set_yticklabels(yticks2_label)
# ax3.tick_params(axis="y",direction="out",length=4,labelsize=10)
# ax3.yaxis.set_minor_locator(FixedLocator(ih_median[::10]))
# ax3.tick_params(axis="y",direction="out",which="minor",length=2)
# colorbar
ax4=fig.add_axes([0.18,0.03,0.67,0.018])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=c_norm,cmap=c_map),cax=ax4,aspect=60,format='%.0f',ticks=(z_count_level),orientation='horizontal',extend='both')
cbar.set_ticks(z_count_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar_label=[str(int(z_count_level[i]/10000)) for i in range(0,np.size(z_count_level))]
cbar.ax.set_xticklabels(cbar_label)
cbar.ax.tick_params(labelsize=7)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
ax3=fig.add_axes([0.85,-0.012,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{4}}$)',family='Times New Roman',fontsize=7,color='black')
plt.savefig(str(str('D:/DA/article2/')+month+str('_zbelow.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z scatter
z_lat_dis=np.array(np.zeros((np.size(lat_median))),dtype='object')
lat_lat_dis=np.array(np.zeros((np.size(lat_median))),dtype='object')
for i in range(0,np.size(lat_median)):
 z_before_select_lat=z_before_select[i,:]
 lat_before_select_lat=lat_before_select[i,:]
 z_lat=np.array([0]).flatten()
 lat_lat=np.array([0]).flatten()
 for j in range(0,np.size(lon_median)):
  z_lat=np.concatenate([z_lat,np.array(z_before_select_lat[j]).flatten()])
  lat_lat=np.concatenate([lat_lat,np.array(lat_before_select_lat[j]).flatten()])
 z_lat_dis[i]=np.array(z_lat[1:])
 lat_lat_dis[i]=np.array(lat_lat[1:])
# z level
z_level=np.arange(0,(12+0.1/10),0.1)
z_level_median=np.array([(z_level[i]+z_level[i+1])/2 for i in range(0,(np.size(z_level)-1))])
z_single_height_total=np.array(np.zeros((np.size(z_level_median),np.size(lat_median))),dtype='object')
lat_single_height_total=np.array(np.zeros((np.size(z_level_median),np.size(lat_median))),dtype='object')
z_single_height_count=np.zeros((np.size(z_level_median),np.size(lat_median)))
for i in range(0,np.size(z_level_median)):
 for j in range(0,np.size(lat_median)):
  z_lat_dis_single=np.array(z_lat_dis[j])
  lat_lat_dis_single=np.array(lat_lat_dis[j])
  z_single_height_total[i,j]=z_lat_dis_single[np.where(np.logical_and(z_lat_dis_single>=z_level[i],z_lat_dis_single<z_level[i+1]))]
  lat_single_height_total[i,j]=lat_lat_dis_single[np.where(np.logical_and(z_lat_dis_single>=z_level[i],z_lat_dis_single<z_level[i+1]))]
  z_single_height_count[i,j]=np.size(z_single_height_total[i,j])

col_num=11
z_lat_count_level=np.linspace(1000,8700,(col_num+1),endpoint=True)
z_lat_count_level=np.concatenate([np.array([5]).flatten(),z_lat_count_level.flatten()])
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(z_lat_count_level,c_map.N,extend='max')
z_count_mask=np.ma.masked_where((z_single_height_count<5),z_single_height_count)
# mesh
ax1.pcolormesh(lat_median,z_level_median,z_count_mask,cmap=c_map,norm=c_norm,shading='nearest',zorder=0)
# scatter
scatter_lat=lat_single_height_total[np.where(z_single_height_count<5)]
scatter_z=z_single_height_total[np.where(z_single_height_count<5)]
for i in range(0,np.size(scatter_lat)):
 ax1.scatter(scatter_lat[i],scatter_z[i],marker='o',c='none',edgecolors='#989898',linewidths=0.4,s=6,zorder=1)
# axis
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Z-score',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-45,45])
ax1.set_ylim([0,14])
xticks1=np.arange(-45,(45+15/10),15)
yticks1=np.arange(0,(14+2/10),2)
yticks1_label=[str(int(yticks1[t])) for t in range(0,np.size(yticks1))]
ax1.set_xticks(xticks1)
ax1.xaxis.set_major_formatter(LatitudeFormatter())
plt.xticks(fontproperties='Times New Roman')
ax1.set_yticks(yticks1)
labels_y=ax1.yaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_y]
ax1.set_yticklabels(yticks1_label)
ax1.tick_params(axis="x",direction="out",length=4,labelsize=10)
ax1.tick_params(axis="y",direction="out",length=4,labelsize=10)
xminorlocator1=MultipleLocator(5)
yminorlocator1=MultipleLocator(((2/4)))
ax1.xaxis.set_minor_locator(xminorlocator1)
ax1.yaxis.set_minor_locator(yminorlocator1)
ax1.yaxis.set_minor_formatter(FormatStrFormatter(''))
ax1.tick_params(axis="x",direction="out",which="minor",length=2)
ax1.tick_params(axis="y",direction="out",which="minor",length=2)
# colorbar
ax4=fig.add_axes([0.18,0.03,0.67,0.018])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=c_norm,cmap=c_map),cax=ax4,aspect=60,format='%.0f',ticks=z_lat_count_level,extend='max',orientation='horizontal')
cbar.set_ticks(z_lat_count_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar_label=[str(int(z_lat_count_level[i]/100)) for i in range(0,np.size(z_lat_count_level))]
cbar_label[0]=str('0.05')
cbar.ax.set_xticklabels(cbar_label)
cbar.ax.tick_params(labelsize=7)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
ax3=fig.add_axes([0.85,-0.012,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{2}}$)',family='Times New Roman',fontsize=7,color='black')
# plt.savefig(str(str('D:/DA/article2/')+month+str('_')+str(ih_select)+str('_z_lat_scatter.png')),dpi=600,bbox_inches='tight',pad_inches=0)
# plt.savefig(str(str('D:/DA/article2/')+month+str('_')+str(ih_select)+str('_z_lat_scatter.png')),dpi=600,bbox_inches='tight',pad_inches=0)
# plt.savefig(str(str('D:/DA/publish/MWR/fig/fig5c.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # 7-9 km
msl_min=6
msl_max=8
index_bottom=int(np.array([i for i in range(0,np.size(msl_median)) if msl_median[i]<=msl_min and msl_median[i+1]>=msl_min])[0])
index_top=int(int(np.array([i for i in range(0,np.size(msl_median)) if msl_median[i]<=msl_max and ih_median[i+1]>=msl_max])[0])+1)
zabove_scale_select=np.array(z_above_count_before[index_bottom:index_top,:,:])
zabove_scale_day_dis_total=np.nansum(zabove_scale_select,axis=0)
cyclic_zabove_scale_dis,_=add_cyclic_point(zabove_scale_day_dis_total,coord=lon_median)

above_level_diff=np.linspace(50,710,7,endpoint=True)
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# colormesh
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805",]#,"#F0F20D","#F8BB08","#F48B06","#F42103", "#BC012E","#FA02FA"
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(above_level_diff,cbar_map.N,extend='min')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zabove_scale_dis,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# ahi scale
# from matplotlib.patches import Rectangle
# region_ahi=Rectangle([80,-60],120,120,linewidth=0.6,linestyle='--',edgecolor='#000000',facecolor='none',transform=ccrs.PlateCarree(),zorder=10)
# ax1.add_patch(region_ahi)
# ax1.scatter(140,0,marker='x',c='#000000',linestyle='-',linewidth=0.6,zorder=10,transform=ccrs.PlateCarree())
# tick
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(15/3)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
ax2=fig.add_axes([0.35,0.3,0.32,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.0f',ticks=above_level_diff)
cbar.set_ticks(above_level_diff)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/article2/')+str('z_')+str(z_threshold)+str('_')+str(month)+str('_')+str(msl_min)+str('_')+str(msl_max)+str('_above_count.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # zmean msl 7-9
msl_min=6
msl_max=8
index_bottom=int(np.array([i for i in range(0,np.size(msl_median)) if msl_median[i]<=msl_min and msl_median[i+1]>=msl_min])[0])
index_top=int(int(np.array([i for i in range(0,np.size(msl_median)) if msl_median[i]<=msl_max and ih_median[i+1]>=msl_max])[0])+1)
z_scale_select=np.array(z_score_before[index_bottom:index_top,:,:])
zmean_scale_select=np.zeros((np.size(lat_median),np.size(lon_median)))
zstd_scale_select=np.zeros((np.size(lat_median),np.size(lon_median)))
for i in range(0,np.size(lat_median)):
 for j in range(0,np.size(lon_median)):
  z_scale_single=z_scale_select[:,i,j]
  z_total_single=np.array([0]).flatten()
  for k in range(0,np.size(z_scale_single)):
   z_total_single=np.concatenate([z_total_single,np.array(z_scale_single[k]).flatten()])
  z_total_single=z_total_single[1:]
  zmean_scale_select[i,j]=np.nanmean(z_total_single)
  zstd_scale_select[i,j]=np.nanstd(z_total_single)

cyclic_zmean_scale_dis,_=add_cyclic_point(zmean_scale_select,coord=lon_median)
cyclic_zstd_scale_dis,_=add_cyclic_point(zstd_scale_select,coord=lon_median)

col_num=11
zmean_level=np.linspace(0.6,0.93,(col_num+1),endpoint=True)
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(zmean_level,cbar_map.N,extend='both')
cyclic_zmean_dis_mask=np.nan_to_num(cyclic_zmean_scale_dis,nan=0)
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zmean_dis_mask,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(25/5)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=zmean_level)
cbar.set_ticks(zmean_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/article2/')+('regional_')+str(month)+str('_')+str(msl_min)+str('_')+str(msl_max)+str('km_zmean_before.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # zstd msl 7-9
zstd_level_after=np.linspace(0.6,0.93,(col_num+1),endpoint=True)
col_num=11
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(zstd_level_after,cbar_map.N,extend='both')
cyclic_zstd_dis_mask=np.nan_to_num(cyclic_zstd_scale_dis,nan=0)
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zstd_dis_mask,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(25/5)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=zstd_level_after)
cbar.set_ticks(zstd_level_after)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/article2/')+('regional_')+str(month)+str('_')+str(msl_min)+str('_')+str(msl_max)+str('km_zstd_before.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # bm msl 7-9
# msl_min=6
# msl_max=8
# index_bottom=int(np.array([i for i in range(0,np.size(msl_median)) if msl_median[i]<=msl_min and msl_median[i+1]>=msl_min])[0])
# index_top=int(int(np.array([i for i in range(0,np.size(msl_median)) if msl_median[i]<=msl_max and ih_median[i+1]>=msl_max])[0])+1)
# z_scale_select=np.array(z_score_before[index_bottom:index_top,:,:])
# zmean_scale_select=np.zeros((np.size(lat_median),np.size(lon_median)))
# zstd_scale_select=np.zeros((np.size(lat_median),np.size(lon_median)))
# for i in range(0,np.size(lat_median)):
#  for j in range(0,np.size(lon_median)):
#   z_scale_single=z_scale_select[:,i,j]
#   z_total_single=np.array([0]).flatten()
#   for k in range(0,np.size(z_scale_single)):
#    z_total_single=np.concatenate([z_total_single,np.array(z_scale_single[k]).flatten()])
#   z_total_single=z_total_single[1:]
#   zmean_scale_select[i,j]=np.nanmean(z_total_single)
#   zstd_scale_select[i,j]=np.nanstd(z_total_single)

# cyclic_zmean_scale_dis,_=add_cyclic_point(zmean_scale_select,coord=lon_median)
# cyclic_zstd_scale_dis,_=add_cyclic_point(zstd_scale_select,coord=lon_median)

# lat height count
count_path=(str(str('D:/DA/GPS/789/cosmic2_')+month+str('_lat_count_ba_bm_bsd2.nc')))
count_file=Dataset(count_path,format='netCDF4')
lat_count=np.array(count_file.variables['count'][:])
ba_bm=np.array(count_file.variables['ba_bm'][:])
ba_bsd=np.array(count_file.variables['ba_bsd'][:])
ba_bm_single=ba_bm[count_index,:]


bm_before_select=np.array(lon_ba_bm[count_index,:,:])
cyclic_bm_before,_=add_cyclic_point(bm_before_select,coord=lon_median)


col_num=11
zmean_level=np.linspace(8.8,11,(col_num+1),endpoint=True)
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
# col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_list=["#EAE3C7","#C8E1B7","#AED7B7","#83CDAC","#58BEA6","#12B7A9","#00B7B6","#01B4CA","#2094c7","#697FBB","#8366A9","#92509A","#A93391"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(zmean_level,cbar_map.N,extend='both')
cyclic_zmean_dis_mask=np.nan_to_num(cyclic_bm_before,nan=0)
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zmean_dis_mask*(10**3),cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=1,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(25/5)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# other axis
axin_width=1/3
x2min=8
x2max=11
x2_interval=1
axin=ax1.inset_axes([0,0,axin_width,1],alpha=1,zorder=0)
axin.set_xlim([x2min,x2max])
xticks2=np.arange(x2min,(x2max+x2_interval/10),x2_interval)
axin.set_xticks(xticks2)
axin.tick_params(axis="x",direction="out",top=True,length=4,labelsize=10)
xminorlocator2=MultipleLocator(x2_interval/2)
axin.xaxis.set_minor_locator(xminorlocator2)
axin.tick_params(axis="x",direction="out",top=True,which="minor",length=2)
axin.xaxis.tick_top()
xticks2_label=[str(int(xticks2[t])) for t in range(0,np.size(xticks2))]
labels_x2=axin.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_x2]
axin.set_xticklabels(xticks2_label)
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
axin.set_xlabel(str('\u03bc'+'$_{\mathregular{\u03B1}}$ (10$^{\mathregular{-3}}$ rad)'),fontdict=font_label1)
axin.xaxis.set_label_position('top')
axin.set_ylim([-45,45])
axin.set_yticks(np.array([]))
# bm plot
bm_trans=(ba_bm_single*(10**3)-x2min)/(x2max-x2min)*(360*axin_width)
ax1.plot(bm_trans,lat_median,linewidth=1.4,linestyle='--',color='#000000',transform=ccrs.PlateCarree(),zorder=2)
# ax1.scatter(bm_trans,lat_median,marker='o',s=2,linewidth=0.6,linestyle='-',c='none',edgecolors='#000000',transform=ccrs.PlateCarree(),zorder=3)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.1f',ticks=zmean_level)
cbar.set_ticks(zmean_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
# unit
ax3=fig.add_axes([0.823,0.268,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-3}}$ rad)',family='Times New Roman',fontsize=8,color='black')
# plt.savefig(str(str('D:/DA/article2/')+('regional_')+str(month)+str('_')+str(ih_select)+str('km_bm_before.png')),dpi=600,bbox_inches='tight',pad_inches=0)
plt.savefig(str(str('D:/DA/publish/MWR/fig/fig6a.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # bsd msl 7-9
count_path=(str(str('D:/DA/GPS/789/cosmic2_')+month+str('_lat_count_ba_bm_bsd.nc')))
count_file=Dataset(count_path,format='netCDF4')
lat_count=np.array(count_file.variables['count'][:])
ba_bm=np.array(count_file.variables['ba_bm'][:])
ba_bsd=np.array(count_file.variables['ba_bsd'][:])
ba_bsd_single=ba_bsd[count_index,:]


bsd_before_select=np.array(lon_ba_bsd[count_index,:,:])
cyclic_bsd_before,_=add_cyclic_point(bsd_before_select,coord=lon_median)

col_num=11
zmean_level=np.linspace(2,10.8,(col_num+1),endpoint=True)
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
# col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_list=["#EAE3C7","#C8E1B7","#AED7B7","#83CDAC","#58BEA6","#12B7A9","#00B7B6","#01B4CA","#2094c7","#697FBB","#8366A9","#92509A","#A93391"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(zmean_level,cbar_map.N,extend='both')
cyclic_zmean_dis_mask=np.nan_to_num(cyclic_bsd_before,nan=0)
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zmean_dis_mask*(10**4),cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=1,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(60/3)
yminorlocator=MultipleLocator(25/5)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# other axis
axin_width=1/3
x2min=2
x2max=12
x2_interval=2
axin=ax1.inset_axes([0,0,axin_width,1],alpha=1,zorder=0)
axin.set_xlim([x2min,x2max])
xticks2=np.arange(x2min,(x2max+x2_interval/10),x2_interval)
axin.set_xticks(xticks2)
axin.tick_params(axis="x",direction="out",top=True,length=4,labelsize=10)
xminorlocator2=MultipleLocator(x2_interval/2)
axin.xaxis.set_minor_locator(xminorlocator2)
axin.tick_params(axis="x",direction="out",top=True,which="minor",length=2)
axin.xaxis.tick_top()
xticks2_label=[str(int(xticks2[t])) for t in range(0,np.size(xticks2))]
labels_x2=axin.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_x2]
axin.set_xticklabels(xticks2_label)
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
axin.set_xlabel(str(str(chr(963))+'$_{\mathregular{\u03B1}}$ (10$^{\mathregular{-4}}$ rad)'),fontdict=font_label1)
axin.xaxis.set_label_position('top')
axin.set_ylim([-45,45])
axin.set_yticks(np.array([]))
# bm plot
bsd_trans=(ba_bsd_single*(10**4)-x2min)/(x2max-x2min)*(360*axin_width)
ax1.plot(bsd_trans,lat_median,linewidth=1.4,linestyle='--',color='#000000',transform=ccrs.PlateCarree(),zorder=2)
# ax1.scatter(bm_trans,lat_median,marker='o',s=2,linewidth=0.6,linestyle='-',c='none',edgecolors='#000000',transform=ccrs.PlateCarree(),zorder=3)
# colorbar
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.1f',ticks=zmean_level)
cbar.set_ticks(zmean_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
# unit
ax3=fig.add_axes([0.823,0.268,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-4}}$ rad)',family='Times New Roman',fontsize=8,color='black')
# plt.savefig(str(str('D:/DA/article2/')+('regional_')+str(month)+str('_')+str(ih_select)+str('km_bsd_before.png')),dpi=600,bbox_inches='tight',pad_inches=0)
plt.savefig(str(str('D:/DA/publish/MWR/fig/fig6b.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""









z_above_count_lat=np.nansum(z_above_count_before,axis=2)
z_above_count_lat_height=np.nansum(z_above_count_lat,axis=1)
z_below_count_lat=np.nansum(z_below_count_before,axis=2)
z_below_count_lat_height=np.nansum(z_below_count_lat,axis=1)


nc_file=Dataset(str('D:/DA/latlon_qc_z_lat_count.nc'),'w',format='NETCDF4')
nc_file.createDimension('row',np.size(ih_median))
nc_file.createDimension('col',np.size(lat_median))
nc_file.createVariable('msl_level',np.float64,('row'))
nc_file.variables['msl_level'][:]=msl_median
nc_file.createVariable('lat',np.float64,('col'))
nc_file.variables['lat'][:]=lat_median
nc_file.createVariable('z_above_count',np.float64,('row','col'))
nc_file.variables['z_above_count'][:]=z_above_count_lat
nc_file.createVariable('z_below_count',np.float64,('row','col'))
nc_file.variables['z_below_count'][:]=z_below_count_lat
nc_file.createVariable('z_above_h_count',np.float64,('row'))
nc_file.variables['z_above_h_count'][:]=z_above_count_lat_height
nc_file.createVariable('z_below_h_count',np.float64,('row'))
nc_file.variables['z_below_h_count'][:]=z_below_count_lat_height
nc_file.close()









