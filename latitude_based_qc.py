import netCDF4 as nc
import numpy as np
import gc
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import xarray as xr
import scipy
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,FixedLocator
from pyproj import Transformer
from netCDF4 import Dataset


# month=str('1212')
month=str('789')
# month=str('123')
# month_add1=str('07')
# month_add2=str('09')

# 1 month
# path_ba_ih1=str(str('D:/DA/cosmic2_')+str(month)+str('_locate_ba_ih_pres_msl_time_trh.nc'))
# ba_ih=Dataset(path_ba_ih1,format='netCDF4')
# lat=np.array(ba_ih.variables['lat'][:])
# lon=np.array(ba_ih.variables['lon'][:])
# ih=np.array(ba_ih.variables['ih'][:])
# ba=np.array(ba_ih.variables['ba'][:])
# msl=np.array(ba_ih.variables['msl'][:])
# ih[ih==-999.0]=np.NaN
# msl[msl==-999.0]=np.NaN
# ba[ba==-999.0]=np.NaN
# ba[ba<0]=np.NaN
# ba[ba>0.1]=np.NaN
# # time
# path2=(str(str('D:/DA/cosmic2_')+month+str('_time.nc')))
# time=Dataset(path2,format='netCDF4')
# day=np.array(time.variables['day'][:])
# hour=np.array(time.variables['hour'][:])
# minute=np.array(time.variables['minute'][:])
# time_utc=hour+minute/60
# time_lst=time_utc+lon/15
# day_lst=day
# for i in range(0,np.size(time_lst)):
#  if time_lst[i]>24:
#   time_lst[i]=time_lst[i]-24
#   day_lst[i]=day_lst[i]+1
#  if time_lst[i]<0:
#   time_lst[i]=time_lst[i]+24
#   day_lst[i]=day_lst[i]-1
# del time_utc
# # solar hour angle
# omega=15*(time_lst-12)
# # solar angle
# eptro=2*np.pi*(day_lst-1)/365
# # declination
# gama=np.ones((np.size(eptro)))*(0.006918)-0.399912*np.cos(eptro)+0.070257*np.sin(eptro)-0.006758*np.cos(2*eptro)+0.000907*np.sin(2*eptro)-0.002697*np.cos(3*eptro)+0.00148*np.sin(3*eptro)
# # solar zenith angle
# soz=np.ones((np.size(lat)))*(90)-np.rad2deg(np.arcsin(np.sin(np.deg2rad(lat))*np.sin(np.deg2rad(gama))+np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(gama))*np.cos(np.deg2rad(omega))))
# # clear
# del time_lst
# del day_lst
# del omega
# del eptro
# del gama


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
del lat1
del lat2
del lat3
del lon1
del lon2
del lon3
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
ba[np.where(ba==-999.0)]=np.NaN
# ih
ih=np.ones((int(np.size(ih1,0)+np.size(ih2,0)+np.size(ih3,0)),int(np.max(np.array([np.size(ih1,1),np.size(ih2,1),np.size(ih3,1)])))))*(-999.0)
ih[0:np.size(ih1,0),0:np.size(ih1,1)]=np.array(ih1)
ih[np.size(ih1,0):int(np.size(ih1,0)+np.size(ih2,0)),0:np.size(ih2,1)]=np.array(ih2)
ih[int(np.size(ih1,0)+np.size(ih2,0)):,0:np.size(ih3,1)]=np.array(ih3)
del ih1
del ih2
del ih3
ih[np.where(ih==-999.0)]=np.NaN
# msl
msl=np.ones((int(np.size(msl1,0)+np.size(msl2,0)+np.size(msl3,0)),int(np.max(np.array([np.size(msl1,1),np.size(msl2,1),np.size(msl3,1)])))))*(-999.0)
msl[0:np.size(msl1,0),0:np.size(msl1,1)]=np.array(msl1)
msl[np.size(msl1,0):int(np.size(msl1,0)+np.size(msl2,0)),0:np.size(msl2,1)]=np.array(msl2)
msl[int(np.size(msl1,0)+np.size(msl2,0)):,0:np.size(msl3,1)]=np.array(msl3)
del msl1
del msl2
del msl3
msl[np.where(msl==-999.0)]=np.NaN
gc.collect()



# lat / ih levels
h_interval=0.1
lat_interval=2.5
lon_interval=2.5

# ih
ih_level=np.arange(2,20.01,h_interval)
ih_median=np.array([(ih_level[i]+ih_level[i+1])/2 for i in range(0,(np.size(ih_level)-1))])

# msl
# msl_level=np.arange(0,20.01,h_interval)
# msl_median=np.array([(msl_level[i]+msl_level[i+1])/2 for i in range(0,(np.size(msl_level)-1))])

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

# lat height count
count_path=(str(str('D:/DA/GPS/789/cosmic2_')+month+str('_lat_count_ba_bm_bsd2.nc')))
count_file=Dataset(count_path,format='netCDF4')
lat_count=np.array(count_file.variables['count'][:])
ba_bm=np.array(count_file.variables['ba_bm'][:])
ba_bsd=np.array(count_file.variables['ba_bsd'][:])

# z score count before/after qc 
z_threshold=2.5
# before
z_mean_before=np.zeros((np.size(ih_median),np.size(lat_median)))
z_std_before=np.zeros((np.size(ih_median),np.size(lat_median)))
z_above_count_before=np.zeros((np.size(ih_median),np.size(lat_median)))
z_below_count_before=np.zeros((np.size(ih_median),np.size(lat_median)))
z_total_before=np.array(np.zeros((np.size(ih_median),np.size(lat_median))),dtype='object')
# z_each_prf=np.zeros((np.size(ba,0),np.size(ba,1)))
# z_day_above_count_before=np.zeros((np.size(ih_median),np.size(lat_median)))
# z_day_below_count_before=np.zeros((np.size(ih_median),np.size(lat_median)))
# after
bm_after=np.zeros((np.size(ih_median),np.size(lat_median)))
bsd_after=np.zeros((np.size(ih_median),np.size(lat_median)))
# z_mean_after=np.zeros((np.size(ih_median),np.size(lat_median)))
# z_std_after=np.zeros((np.size(ih_median),np.size(lat_median)))
# z_above_count_after=np.zeros((np.size(ih_median),np.size(lat_median)))
# z_below_count_after=np.zeros((np.size(ih_median),np.size(lat_median)))
# z_total_after=np.array(np.zeros((np.size(ih_median),np.size(lat_median))),dtype='object')
# z_day_above_count_after=np.zeros((np.size(ih_median),np.size(lat_median)))
# z_day_below_count_after=np.zeros((np.size(ih_median),np.size(lat_median)))
# lon
lon_total_before=np.array(np.zeros((np.size(ih_median),np.size(lat_median))),dtype='object')
# lon_total_after=np.array(np.zeros((np.size(ih_median),np.size(lat_median))),dtype='object')
lon_ih=np.zeros((np.size(lon),np.size(ih,1)))
for i in range(0,np.size(lon)):
 lon_ih[i,:]=np.ones((np.size(ih,1)))*(lon[i])
# lat
lat_total_before=np.array(np.zeros((np.size(ih_median),np.size(lat_median))),dtype='object')
# lon_total_after=np.array(np.zeros((np.size(ih_median),np.size(lat_median))),dtype='object')
lat_ih=np.zeros((np.size(lat),np.size(ih,1)))
for i in range(0,np.size(lat)):
 lat_ih[i,:]=np.ones((np.size(ih,1)))*(lat[i])
# time
# soz_total_before=np.array(np.zeros((np.size(ih_median),np.size(lat_median))),dtype='object')
# soz_total_after=np.array(np.zeros((np.size(ih_median),np.size(lat_median))),dtype='object')
# soz_ih=np.zeros((np.size(lon),np.size(ih,1)))
# for i in range(0,np.size(soz)):
#  soz_ih[i,:]=np.ones((np.size(ih,1)))*(soz[i])
for i in range(0,np.size(lat_median)):
 lat_index=np.array(np.where(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]))[0])
 ih_lat=np.array(np.array(ih[np.where(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1])),:])[0,:,:])
 ba_lat=np.array(np.array(ba[np.where(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1])),:])[0,:,:])
 lon_lat=np.array(np.array(lon_ih[np.where(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1])),:])[0,:,:])
 lat_lat=np.array(np.array(lat_ih[np.where(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1])),:])[0,:,:])
 # soz_lat=np.array(np.array(soz_ih[np.where(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1])),:])[0,:,:])
 for j in range(0,np.size(ih_median)):
  # before qc
  h_lat_index=np.array(np.where(np.logical_and(ih_lat>=ih_level[j],ih_lat<ih_level[j+1]))[0])
  row_index=np.array([int(lat_index[int(h_lat_index[i])]) for i in range(0,np.size(h_lat_index))])
  del h_lat_index
  col_index=np.array(np.where(np.logical_and(ih_lat>=ih_level[j],ih_lat<ih_level[j+1]))[1])
  ba_lat_single_before=np.array(ba_lat[np.where(np.logical_and(ih_lat>=ih_level[j],ih_lat<ih_level[j+1]))])
  lon_lat_single_before=np.array(lon_lat[np.where(np.logical_and(ih_lat>=ih_level[j],ih_lat<ih_level[j+1]))])
  lat_lat_single_before=np.array(lat_lat[np.where(np.logical_and(ih_lat>=ih_level[j],ih_lat<ih_level[j+1]))])
  # soz_lat_single_before=np.array(soz_lat[np.where(np.logical_and(ih_lat>=ih_level[j],ih_lat<ih_level[j+1]))])
  ba_lat_z_before=abs(ba_lat_single_before-ba_bm[j,i])/ba_bsd[j,i]
  z_mean_before[j,i]=np.nanmean(ba_lat_z_before)
  z_std_before[j,i]=np.nanstd(ba_lat_z_before)
  lon_total_before[j,i]=np.array(lon_lat_single_before)
  lat_total_before[j,i]=np.array(lat_lat_single_before)
  # soz_total_before[j,i]=np.array(soz_lat_single_before)
  z_total_before[j,i]=np.array(ba_lat_z_before)
  # all time
  z_above_count_before[j,i]=np.size(ba_lat_z_before[np.where(ba_lat_z_before>z_threshold)])
  z_below_count_before[j,i]=np.size(ba_lat_z_before[np.where(ba_lat_z_before<=z_threshold)])
  # z-score for each profile
  # for k in range(0,np.size(col_index)):
  #  z_each_prf[int(row_index[k]),int(col_index[k])]=ba_lat_z_before[k]
  # day
  # ba_lat_z_before_day=np.array(ba_lat_z_before[np.where(soz_lat_single_before<90)])
  # z_day_above_count_before[j,i]=np.size(np.array(ba_lat_z_before_day[ba_lat_z_before_day>z_threshold]))
  # z_day_below_count_before[j,i]=np.size(np.array(ba_lat_z_before_day[ba_lat_z_before_day<=z_threshold]))
  # after qc
  # ba_lat_single_after=np.array(ba_lat_single_before[np.where(ba_lat_z_before<=z_threshold)])
  # # lon_lat_single_after=np.array(lon_lat_single_before[np.where(ba_lat_z_before<=z_threshold)])
  # # lat_lat_single_after=np.array(lat_lat_single_before[np.where(ba_lat_z_before<=z_threshold)])
  # # soz_lat_single_after=np.array(soz_lat_single_before[np.where(ba_lat_z_before<=z_threshold)])
  # m=np.nanmedian(ba_lat_single_after)
  # mad=np.array(abs(ba_lat_single_after-m))
  # # # weight
  # w=(ba_lat_single_after-m)/(7.5*mad)
  # w[abs(w)>1]=1
  # # biweight mean/std
  # w_cal1=np.zeros((np.size(np.array(ba_lat_single_after))))
  # for l in range(0,np.size(np.array(ba_lat_single_after))):
  #  w_cal1[l]=(ba_lat_single_after[l]-m)*((1-(w[l])**2)**2)
  # w_cal2=((1-w)**2)**2
  # w_cal3=w_cal1**2
  # w_cal4=w_cal2*(1-5*(w**2))
  # # mean
  # bm_after[j,i]=m+(np.nansum(w_cal1)/np.nansum(w_cal2))
  # # std
  # bsd_after[j,i]=np.sqrt(np.size(np.array(ba_lat_single_after))*np.nansum(w_cal3))/abs(np.nansum(w_cal4))
  # # z
  # ba_lat_z_after=abs(ba_lat_single_after-bm_after[j,i])/bsd_after[j,i]
  # z_mean_after[j,i]=np.nanmean(ba_lat_z_after)
  # z_std_after[j,i]=np.nanstd(ba_lat_z_after)
  # lon_total_after[j,i]=np.array(lon_lat_single_after)
  # time_total_after[j,i]=np.array(time_lat_single_after)
  # z_total_after[j,i]=np.array(ba_lat_z_after)  
  # z_above_count_after[j,i]=np.size(ba_lat_z_after[np.where(ba_lat_z_after>z_threshold)])
  # z_below_count_after[j,i]=np.size(ba_lat_z_after[np.where(ba_lat_z_after<=z_threshold)])
  # # day
  # z_day_above_count_after[j,i]=np.size(np.array(time_lat_single_after[np.where(ba_lat_z_after>z_threshold)])[np.where(np.logical_and(time_lat_single_after>=6,time_lat_single_after<=18))])

# z_each_prf[np.where(np.isnan(ba))]=np.NaN

# night
# z_night_above_count_before=z_above_count_before-z_day_above_count_before
# z_night_below_count_before=z_below_count_before-z_day_below_count_before
# z_night_above_count_after=z_above_count_after-z_day_above_count_after
# z_night_below_count_after=z_below_count_after-z_day_below_count_after

# z-score nc file
# save_path=(r'D:/DA/GPS/789/cosmic2_z_score2.nc')
# nc_file=Dataset(str(save_path),'w',format='NETCDF4')
# nc_file.createDimension('prf_number',np.size(ba,0))
# nc_file.createDimension('ih_level',np.size(ba,1))
# nc_file.createVariable('z_score',np.float64,('prf_number','ih_level'))
# nc_file.variables['z_score'][:]=z_each_prf
# nc_file.close()



# ih_level bm/bsd  mean / z out count
# bm_mean=np.zeros((np.size(ih_median)))
# bsd_mean=np.zeros((np.size(ih_median)))
# bm_mean_after=np.zeros((np.size(ih_median)))
# bsd_mean_after=np.zeros((np.size(ih_median)))
# z_mean_level=np.zeros((np.size(ih_median)))
# z_std_level=np.zeros((np.size(ih_median)))
# z_out_lat_count=np.zeros((np.size(ih_median)))
# p_level_mean=np.zeros((np.size(ih_median)))
# for i in range(0,np.size(ih_median)):
#  # p
#  ba_ih_single=np.array(ba[np.where(np.logical_and(ih>=ih_level[i],ih<ih_level[i+1]))]).flatten()
#  m=np.nanmedian(ba_ih_single)
#  mad=np.array(abs(ba_ih_single-m))
#  # weight
#  w=(ba_ih_single-m)/(7.5*mad)
#  w[abs(w)>1]=1
#  # biweight mean/std
#  w_cal1=np.zeros((np.size(np.array(ba_ih_single))))
#  for l in range(0,np.size(np.array(ba_ih_single))):
#   w_cal1[l]=(ba_ih_single[l]-m)*((1-(w[l])**2)**2)
#  w_cal2=((1-w)**2)**2
#  w_cal3=w_cal1**2
#  w_cal4=w_cal2*(1-5*(w**2))
#  # mean
#  bm_mean[i]=m+(np.nansum(w_cal1)/np.nansum(w_cal2))
#  # std
#  bsd_mean[i]=np.sqrt(np.size(np.array(ba_ih_single))*np.nansum(w_cal3))/abs(np.nansum(w_cal4))
#  # z 
#  z_ih_single=abs((ba_ih_single-bm_mean[i])/bsd_mean[i])
#  z_mean_level[i]=np.nanmean(z_ih_single)
#  z_std_level[i]=np.nanstd(z_ih_single)
#  z_out_lat_count[i]=np.size(np.array(z_ih_single[np.where(z_ih_single>z_threshold)]))
#  # after
#  ba_ih_single_after=np.array(ba_ih_single[np.where(z_ih_single<=z_threshold)])
#  m_after=np.nanmedian(ba_ih_single_after)
#  mad_after=np.array(abs(ba_ih_single_after-m_after))
#  # weight
#  w_after=(ba_ih_single_after-m_after)/(7.5*mad_after)
#  w_after[abs(w_after)>1]=1
#  # biweight mean/std
#  w_cal1_after=np.zeros((np.size(np.array(ba_ih_single_after))))
#  for l in range(0,np.size(np.array(ba_ih_single_after))):
#   w_cal1_after[l]=(ba_ih_single_after[l]-m_after)*((1-(w_after[l])**2)**2)
#  w_cal2_after=((1-w_after)**2)**2
#  w_cal3_after=w_cal1_after**2
#  w_cal4_after=w_cal2_after*(1-5*(w_after**2))
#  # mean
#  bm_mean_after[i]=m_after+(np.nansum(w_cal1_after)/np.nansum(w_cal2_after))
#  # std
#  bsd_mean_after[i]=np.sqrt(np.size(np.array(ba_ih_single_after))*np.nansum(w_cal3_after))/abs(np.nansum(w_cal4_after))


# nc_file=Dataset('D:/DA/GPS/789/cosmic2_789_lat_count_ba_bm_bsd_after.nc','w',format='NETCDF4')
# # nc_file=Dataset('D:/DA/cosmic2_789_lat_occ_count_ba_bm_bsd.nc','w',format='NETCDF4')
# nc_file.createDimension('h_level',np.size(ih_median))
# nc_file.createDimension('lat_level',np.size(lat_median))
# nc_file.createVariable('ba_bm_mean',np.float64,('h_level'))
# nc_file.variables['ba_bm_mean'][:]=bm_mean
# nc_file.createVariable('ba_bsd_mean',np.float64,('h_level'))
# nc_file.variables['ba_bsd_mean'][:]=bsd_mean
# nc_file.createVariable('ba_bm_mean_after',np.float64,('h_level'))
# nc_file.variables['ba_bm_mean_after'][:]=bm_mean_after
# nc_file.createVariable('ba_bsd_mean_after',np.float64,('h_level'))
# nc_file.variables['ba_bsd_mean_after'][:]=bsd_mean_after
# nc_file.createVariable('ba_bm',np.float64,('h_level','lat_level'))
# nc_file.variables['ba_bm'][:]=bm_after
# nc_file.createVariable('ba_bsd',np.float64,('h_level','lat_level'))
# nc_file.variables['ba_bsd'][:]=bsd_after
# nc_file.close()





""" # count level
lat_sum=np.sum(lat_count,axis=0)
# count_level=np.concatenate([np.array([1000]).flatten(),np.array(np.linspace(3000,42000,14,endpoint=True)).flatten()])
count_level=np.concatenate([np.array([0]).flatten(),np.array(np.linspace(10000,140000,14,endpoint=True)).flatten()])
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(count_level,c_map.N)
lat_count_mask=np.ma.masked_where((lat_count<=np.min(count_level)),lat_count)
ax1.pcolormesh(lat_median,ih_median,lat_count_mask,cmap=c_map,norm=c_norm,shading='nearest',zorder=0)
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Impact Height (km)',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-45,45])
ax1.set_ylim([2,20])
# xticks1=np.concatenate([np.array([-35]).flatten(),np.arange(-20,20.1,10).flatten(),np.array([35]).flatten()])
xticks1=np.arange(-45,(45+15/10),15)
yticks1=np.arange(2,(20+3/10),3)
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
xminorlocator1=MultipleLocator((5))
yminorlocator1=MultipleLocator(((3/3)))
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
ax2=ax1.twinx()
# cumulation
ax2.plot(lat_median,np.array(lat_sum/(10**6)),linewidth=0.6,color='k',linestyle='-',zorder=1)
ax2.scatter(lat_median,np.array(lat_sum/(10**6)),marker='o',s=6,c='none',edgecolors='#000000',linewidth=0.4,zorder=2)
# 坐标轴标签
ax2.set_ylabel('Data Count (10$^{\mathregular{6}}$)',fontdict=font_label1)
# 刻度(主+次)
ax2.set_ylim([0,25])
yticks2=np.arange(0,25.1,5)
ax2.set_yticks(yticks2)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax2.tick_params(axis="y",direction="out",length=4)
yminorlocator2=MultipleLocator((1))
ax2.yaxis.set_minor_locator(yminorlocator2)
ax2.tick_params(axis="y",direction="out",which="minor",length=2)
# colorbar
ax4=fig.add_axes([0.18,0.03,0.67,0.018])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=c_norm,cmap=c_map),cax=ax4,aspect=60,format='%.0f',ticks=(count_level),orientation='horizontal')
cbar.set_ticks(count_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar_label=[str(int(count_level[i]/10000)) for i in range(0,np.size(count_level))]
cbar.ax.set_xticklabels(cbar_label)
cbar.ax.tick_params(labelsize=7)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
ax3=fig.add_axes([0.873,-0.012,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{4}}$)',family='Times New Roman',fontsize=7,color='black')
plt.savefig(str(str('D:/DA/test/')+month+str('_lat_count.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # bm level
bm_level=np.linspace(0,0.028,15,endpoint=True)
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(bm_level,c_map.N)
bm_mask=ba_bm
# bm_mask=np.ma.masked_where((ba_bm<(1.5*10**(-3))),ba_bm)
ax1.pcolormesh(lat_median,ih_median,bm_mask,cmap=c_map,norm=c_norm,shading='nearest',zorder=0)
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Impact Height (km)',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-45,45])
ax1.set_ylim([2,20])
# xticks1=np.concatenate([np.array([-35]).flatten(),np.arange(-20,20.1,10).flatten(),np.array([35]).flatten()])
xticks1=np.arange(-45,(45+15/10),15)
yticks1=np.arange(2,(20+3/10),3)
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
yminorlocator1=MultipleLocator(((3/3)))
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
ax2.plot(np.array(bm_mean*(10**3)),ih_median,linewidth=0.6,color='k',linestyle='-')
# 坐标轴标签
ax2.set_xlabel('$\overline{\mathregular{\u03B1}}$ (10$^{\mathregular{-3}}$ rad)',fontdict=font_label1)
# 刻度(主+次)
ax2.set_xlim([0,30])
xticks2=np.arange(0,31,6)
ax2.set_xticks(xticks2)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
ax2.tick_params(axis="x",direction="out",length=4)
xminorlocator2=MultipleLocator((2))
ax2.xaxis.set_minor_locator(xminorlocator2)
ax2.tick_params(axis="x",direction="out",which="minor",length=2)
# colorbar
ax4=fig.add_axes([0.18,0.03,0.67,0.018])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=c_norm,cmap=c_map),cax=ax4,aspect=60,format='%.1f',ticks=(bm_level),orientation='horizontal')
cbar.set_ticks(bm_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar_label=[str(int(bm_level[i]*1000)) for i in range(0,np.size(bm_level))]
cbar.ax.set_xticklabels(cbar_label)
cbar.ax.tick_params(labelsize=7)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
ax3=fig.add_axes([0.871,-0.012,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-3}}$ rad)',family='Times New Roman',fontsize=7,color='black')
plt.savefig(str(str('D:/DA/test/')+month+str('_bm.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # bsd level
bsd_level=np.linspace(2,62,13,endpoint=True)
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(bsd_level,c_map.N,extend='both')
bsd_mask=ba_bsd
# bsd_mask=np.ma.masked_where((ba_bsd<np.min(bsd_level)),ba_bsd)
ax1.pcolormesh(lat_median,ih_median,(bsd_mask*10000),cmap=c_map,norm=c_norm,shading='nearest',zorder=0)
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Impact Height (km)',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-45,45])
ax1.set_ylim([2,20])
# xticks1=np.concatenate([np.array([-35]).flatten(),np.arange(-20,20.1,10).flatten(),np.array([35]).flatten()])
xticks1=np.arange(-45,(45+15/10),15)
yticks1=np.arange(2,(20+3/10),3)
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
yminorlocator1=MultipleLocator(((3/3)))
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
# bsd_mean=scipy.signal.savgol_filter(bsd_mean,100,4) 
ax2.plot(np.array(bsd_mean*(10**4)),ih_median,linewidth=0.6,color='k',linestyle='-')
# 坐标轴标签
ax2.set_xlabel(str(str(chr(963))+'$_{\mathregular{\u03B1}}$ (10$^{\mathregular{-4}}$ rad)'),fontdict=font_label1)
# 刻度(主+次)
ax2.set_xlim([0,60])
xticks2=np.arange(0,60.1,15)
ax2.set_xticks(xticks2)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
ax2.tick_params(axis="x",direction="out",length=4)
xminorlocator2=MultipleLocator((5))
ax2.xaxis.set_minor_locator(xminorlocator2)
ax2.tick_params(axis="x",direction="out",which="minor",length=2)
# colorbar
ax4=fig.add_axes([0.18,0.03,0.67,0.018])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=c_norm,cmap=c_map),cax=ax4,aspect=60,format='%.0f',ticks=(bsd_level),orientation='horizontal',extend='both')
cbar.set_ticks(bsd_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
# cbar_label=[str(int(bsd_level[i]*1000)) for i in range(0,np.size(bsd_level))]
# cbar.ax.set_xticklabels(cbar_label)
cbar.ax.tick_params(labelsize=7)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
ax3=fig.add_axes([0.85,-0.012,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-4}}$ rad)',family='Times New Roman',fontsize=7,color='black')
plt.savefig(str(str('D:/DA/test/')+month+str('_bsd.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # z mean level
zmean_level=np.linspace(0.52,0.88,13,endpoint=True)
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(zmean_level,c_map.N,extend='both')
z_mask=z_mean_before
# bm_mask=np.ma.masked_where((ba_bm<(1.5*10**(-3))),ba_bm)
ax1.pcolormesh(lat_median,ih_median,z_mask,cmap=c_map,norm=c_norm,shading='nearest',zorder=0)
# era w
# p_level_s=p_level[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)]))]
# w_scale=w_select_mean[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)])),:][0,:,:]
# v_scale=v_select_mean[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)])),:][0,:,:]
# w_h=np.array([ih_median[np.where(np.array(abs(p_level_mean-p_level_s[i]))==np.min(np.array(abs(p_level_mean-p_level_s[i]))))] for i in range(0,np.size(p_level_s))]).flatten()
# w_f=ax1.contour(lat_era,w_h,w_scale,w_level,colors='black',linewidths=0.5,zorder=1)
# con_label=plt.clabel(w_f,w_level[::1],inline=True,fontsize=7)
# [label.set_fontname('Times New Roman') for label in con_label]
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Impact Height (km)',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-45,45])
ax1.set_ylim([2,20])
# xticks1=np.concatenate([np.array([-35]).flatten(),np.arange(-20,20.1,10).flatten(),np.array([35]).flatten()])
xticks1=np.arange(-45,(45+15/10),15)
yticks1=np.arange(2,(20+3/10),3)
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
yminorlocator1=MultipleLocator(((3/3)))
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
ax2.plot(np.array(z_mean_level),ih_median,linewidth=0.6,color='k',linestyle='-')
# 坐标轴标签
ax2.set_xlabel(str('$\overline{\mathregular{Z}}$'),fontdict=font_label1)
# 刻度(主+次)
ax2.set_xlim([0.5,1])
xticks2=np.arange(0.5,1.01,0.1)
ax2.set_xticks(xticks2)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
ax2.tick_params(axis="x",direction="out",length=4)
xminorlocator2=MultipleLocator((0.05))
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=c_norm,cmap=c_map),cax=ax4,aspect=60,format='%.2f',ticks=(zmean_level),orientation='horizontal',extend='both')
cbar.set_ticks(zmean_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
# cbar_label=[str('{:.0}'.format(zmean_level[i]*100)) for i in range(0,np.size(zmean_level))]
cbar_label=[str(int(zmean_level[i]*100)) for i in range(0,np.size(zmean_level))]
cbar.ax.set_xticklabels(cbar_label)
cbar.ax.tick_params(labelsize=7)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
ax3=fig.add_axes([0.85,-0.012,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-2}}$)',family='Times New Roman',fontsize=7,color='black')
plt.savefig(str(str('D:/DA/test/')+month+str('_zmean.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # z std level
zstd_level=np.linspace(0.54,0.9,13,endpoint=True)
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(zstd_level,c_map.N,extend='both')
z_mask=z_std_before
# bm_mask=np.ma.masked_where((ba_bm<(1.5*10**(-3))),ba_bm)
ax1.pcolormesh(lat_median,ih_median,z_mask,cmap=c_map,norm=c_norm,shading='nearest',zorder=0)
# era w
# p_level_s=p_level[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)]))]
# w_scale=w_select_mean[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)])),:][0,:,:]
# v_scale=v_select_mean[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)])),:][0,:,:]
# w_h=np.array([ih_median[np.where(np.array(abs(p_level_mean-p_level_s[i]))==np.min(np.array(abs(p_level_mean-p_level_s[i]))))] for i in range(0,np.size(p_level_s))]).flatten()
# w_f=ax1.contour(lat_era,w_h,w_scale,w_level,colors='black',linewidths=0.5,zorder=1)
# con_label=plt.clabel(w_f,w_level[::1],inline=True,fontsize=7)
# [label.set_fontname('Times New Roman') for label in con_label]
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Impact Height (km)',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-45,45])
ax1.set_ylim([2,20])
# xticks1=np.concatenate([np.array([-35]).flatten(),np.arange(-20,20.1,10).flatten(),np.array([35]).flatten()])
xticks1=np.arange(-45,(45+15/10),15)
yticks1=np.arange(2,(20+3/10),3)
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
yminorlocator1=MultipleLocator(((3/3)))
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
ax2.plot(np.array(z_std_level),ih_median,linewidth=0.6,color='k',linestyle='-')
# 坐标轴标签
ax2.set_xlabel(str(str(chr(963))+str('$_{\mathregular{Z}}$')),fontdict=font_label1)
# 刻度(主+次)
ax2.set_xlim([0.5,1])
xticks2=np.arange(0.5,1.01,0.1)
ax2.set_xticks(xticks2)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
ax2.tick_params(axis="x",direction="out",length=4)
xminorlocator2=MultipleLocator((0.05))
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=c_norm,cmap=c_map),cax=ax4,aspect=60,format='%.2f',ticks=(zstd_level),orientation='horizontal',extend='both')
cbar.set_ticks(zstd_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
# cbar_label=[str('{:.2}'.format(zstd_level[i])) for i in range(0,np.size(zstd_level))]
cbar_label=[str(int(zstd_level[i]*100)) for i in range(0,np.size(zstd_level))]
cbar.ax.set_xticklabels(cbar_label)
cbar.ax.tick_params(labelsize=7)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
ax3=fig.add_axes([0.85,-0.012,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-2}}$)',family='Times New Roman',fontsize=7,color='black')
plt.savefig(str(str('D:/DA/test/')+month+str('_zstd.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # z above count level
z_count_level=np.concatenate([np.array([100]).flatten(),np.linspace(200,3500,12,endpoint=True).flatten()])
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(z_count_level,c_map.N,extend='both')
z_mask=z_above_count_before
# bm_mask=np.ma.masked_where((ba_bm<(1.5*10**(-3))),ba_bm)
ax1.pcolormesh(lat_median,ih_median,z_mask,cmap=c_map,norm=c_norm,shading='nearest',zorder=0)
# # era w
# p_level_s=p_level[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)]))]
# w_scale=w_select_mean[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)])),:][0,:,:]
# v_scale=v_select_mean[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)])),:][0,:,:]
# w_h=np.array([ih_median[np.where(np.array(abs(p_level_mean-p_level_s[i]))==np.min(np.array(abs(p_level_mean-p_level_s[i]))))] for i in range(0,np.size(p_level_s))]).flatten()
# w_f=ax1.contour(lat_era,w_h,w_scale,w_level,colors='black',linewidths=0.5,zorder=1)
# plt.clabel(w_f,w_level[::1],inline=True,fontsize=6)
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Impact Height (km)',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-45,45])
ax1.set_ylim([2,20])
xticks1=np.arange(-45,(45+15/10),15)
yticks1=np.arange(2,(20+3/10),3)
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
yminorlocator1=MultipleLocator(((3/3)))
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
ax2.plot(np.array(z_out_lat_count)/(10**4),ih_median,linewidth=0.6,color='#000000',linestyle='-',zorder=2)
# 坐标轴标签
# ax2.set_xlabel(str(str('|Z|$_{\mathregular{\u03B1}}$ > ')+str(z_threshold)+str(' Count')),fontdict=font_label1)
ax2.set_xlabel(str('Outliers (10$^{\mathregular{4}}$)'),fontdict=font_label1)
# 刻度(主+次)
ax2.set_xlim([0,12])
xticks2=np.arange(0,12.1,3)
ax2.set_xticks(xticks2)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
ax2.tick_params(axis="x",direction="out",length=4)
xminorlocator2=MultipleLocator((0.75))
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
plt.savefig(str(str('D:/DA/test/')+month+str('_zabove.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # z below count level
z_count_level=np.concatenate([np.array([10000]).flatten(),np.array(np.linspace(20000,130000,12,endpoint=True)).flatten()])
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(z_count_level,c_map.N,extend='both')
z_mask=z_below_count_before
# bm_mask=np.ma.masked_where((ba_bm<(1.5*10**(-3))),ba_bm)
ax1.pcolormesh(lat_median,ih_median,z_mask,cmap=c_map,norm=c_norm,shading='nearest',zorder=0)
# # era w
# p_level_s=p_level[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)]))]
# w_scale=w_select_mean[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)])),:][0,:,:]
# v_scale=v_select_mean[np.where(p_level>=np.min(p_level_mean[p_level_mean>np.min(p_level_mean)])),:][0,:,:]
# w_h=np.array([ih_median[np.where(np.array(abs(p_level_mean-p_level_s[i]))==np.min(np.array(abs(p_level_mean-p_level_s[i]))))] for i in range(0,np.size(p_level_s))]).flatten()
# w_f=ax1.contour(lat_era,w_h,w_scale,w_level,colors='black',linewidths=0.5,zorder=1)
# plt.clabel(w_f,w_level[::1],inline=True,fontsize=6)
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Impact Height (km)',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-45,45])
ax1.set_ylim([2,20])
xticks1=np.arange(-45,(45+15/10),15)
yticks1=np.arange(2,(20+3/10),3)
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
yminorlocator1=MultipleLocator(((3/3)))
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
ax2.plot(np.array(np.sum(z_below_count_before,axis=1)-z_out_lat_count)/(10**5),ih_median,linewidth=0.6,color='#000000',linestyle='-',zorder=2)
# 坐标轴标签
# ax2.set_xlabel(str(str('|Z|$_{\mathregular{\u03B1}}$ > ')+str(z_threshold)+str(' Count')),fontdict=font_label1)
ax2.set_xlabel(str('Normal (10$^{\mathregular{5}}$)'),fontdict=font_label1)
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
plt.savefig(str(str('D:/DA/test/')+month+str('_zbelow.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


# # single height (single height)
ih_select=8
# count_index=int(np.where(abs(ih_level-ih_select)==np.min(abs(ih_level-ih_select)))[0])
count_index=int(np.array([i for i in range(0,np.size(ih_median)) if ih_median[i]<=ih_select and ih_median[i+1]>=ih_select]))
col_num=11
# zmean_level=np.linspace(0.5,1.6,(col_num+1),endpoint=True)
# zstd_level=np.linspace(0.5,1.6,(col_num+1),endpoint=True)
# total_level=np.linspace(100,1200,(col_num+1),endpoint=True)
# below_level=np.linspace(100,1200,(col_num+1),endpoint=True)
# above_level=np.linspace(10,87,(col_num+1),endpoint=True)


# before
# z_mean_before_select=z_mean_before[count_index,:]
# z_std_before_select=z_std_before[count_index,:]
lon_before_select=lon_total_before[count_index,:]
# soz_before_select=soz_total_before[count_index,:]
z_before_select=z_total_before[count_index,:]
# zabove_before_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
# # zabove_day_before_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
# zbelow_before_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
# zbelow_day_before_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
# zmean_before_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
# zstd_before_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
# zmean_after_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
# zstd_after_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
# for i in range(0,np.size(lat_median)):
#  lon_single=lon_before_select[i]
# #  soz_single=soz_before_select[i]
#  z_single=z_before_select[i]
# #  lon_day_single=lon_single[np.where(soz_single<90)]
# #  soz_day_single=soz_single[np.where(soz_single<90)]
# #  z_day_single=z_single[np.where(soz_single<90)]
#  for j in range(0,np.size(lon_median)):
#   z_dis_single=np.array(z_single[np.where(np.logical_and(lon_single>=lon_level[j],lon_single<lon_level[j+1]))].flatten())
#   zmean_before_dis[i,j]=np.nanmean(z_dis_single)
#   zstd_before_dis[i,j]=np.nanstd(z_dis_single)
#   zabove_before_dis[i,j]=np.size(z_dis_single[np.where(z_dis_single>z_threshold)])
#   zbelow_before_dis[i,j]=np.size(z_dis_single[np.where(z_dis_single<=z_threshold)])
#   zmean_after_dis[i,j]=np.nanmean(np.array(z_dis_single[np.where(z_dis_single<=z_threshold)]))
#   zstd_after_dis[i,j]=np.nanstd(np.array(z_dis_single[np.where(z_dis_single<=z_threshold)]))
  # z_day_dis_single=np.array(z_day_single[np.where(np.logical_and(lon_day_single>=lon_level[j],lon_day_single<lon_level[j+1]))].flatten())
  # zabove_day_before_dis[i,j]=np.size(z_day_dis_single[np.where(z_day_dis_single>z_threshold)])
  # zbelow_day_before_dis[i,j]=np.size(z_day_dis_single[np.where(z_day_dis_single<=z_threshold)])
# total_count=zabove_before_dis+zbelow_before_dis
# night
# zabove_night_before_dis=zabove_before_dis-zabove_day_before_dis
# zbelow_night_before_dis=zbelow_before_dis-zbelow_day_before_dis

# global
# cyclic_zabove_before,cyclic_lon=add_cyclic_point(zabove_before_dis,coord=lon_median)
# cyclic_zabove_day_before,_=add_cyclic_point(zabove_day_before_dis,coord=lon_median)
# cyclic_zabove_night_before,_=add_cyclic_point(zabove_night_before_dis,coord=lon_median)
# cyclic_zbelow_before,_=add_cyclic_point(zbelow_before_dis,coord=lon_median)
# # cyclic_zbelow_day_before,_=add_cyclic_point(zbelow_day_before_dis,coord=lon_median)
# # cyclic_zbelow_night_before,_=add_cyclic_point(zbelow_night_before_dis,coord=lon_median)
# cyclic_total_count,_=add_cyclic_point(total_count,coord=lon_median)
# cyclic_zmean_dis_before,_=add_cyclic_point(zmean_before_dis,coord=lon_median)
# cyclic_zstd_dis_before,_=add_cyclic_point(zstd_before_dis,coord=lon_median)
# cyclic_zmean_dis_after,_=add_cyclic_point(zmean_after_dis,coord=lon_median)
# cyclic_zstd_dis_after,_=add_cyclic_point(zstd_after_dis,coord=lon_median)
# cyclic_lon,cyclic_lat=np.meshgrid(cyclic_lon,lat_median)

# # after
# bm_after_select=bm_after[count_index,:]
# bsd_after_select=bsd_after[count_index,:]
# z_mean_after_select=z_mean_after[count_index,:]
# z_std_after_select=z_std_after[count_index,:]
# lon_after_select=lon_total_after[count_index,:]
# z_after_select=z_total_after[count_index,:]
# zabove_after_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
# zbelow_after_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
# zmean_after_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
# zstd_after_dis=np.zeros((np.size(lat_median),np.size(lon_median)))

# for i in range(0,np.size(lat_median)):
#  lon_single=lon_after_select[i]
#  z_single=z_after_select[i]
#  for j in range(0,np.size(lon_median)):
#   z_dis_single=np.array(z_single[np.where(np.logical_and(lon_single>=lon_level[j],lon_single<lon_level[j+1]))].flatten())
#   zmean_after_dis[i,j]=np.nanmean(z_dis_single)
#   zstd_after_dis[i,j]=np.nanstd(z_dis_single)
#   zabove_after_dis[i,j]=np.size(z_dis_single[np.where(z_dis_single>z_threshold)])
#   zbelow_after_dis[i,j]=np.size(z_dis_single[np.where(z_dis_single<=z_threshold)])
# # global
# cyclic_zabove_after,_=add_cyclic_point(zabove_after_dis,coord=lon_median)
# cyclic_zbelow_after,_=add_cyclic_point(zbelow_after_dis,coord=lon_median)
# cyclic_zmean_dis_after,_=add_cyclic_point(zmean_after_dis,coord=lon_median)
# cyclic_zstd_dis_after,_=add_cyclic_point(zstd_after_dis,coord=lon_median)


# era_wind
# era_path=(r'D:/DA/era_2023_123_mean_std.nc')
# era_file=Dataset(era_path,format='netCDF4')
# g_refer=9.80665
# var_kind=['z','u','v','w','d','vo','ciwc','clwc']
# month_era=np.array(era_file.variables['month'][:])
# p_level=np.array(era_file.variables['p_level'][:])
# lat_era=np.array(era_file.variables['lat'][:])
# lon_era=np.array(era_file.variables['lon'][:])
# # select index
# p_select=400
# month_select=1
# p_index=int(np.where(p_level==p_select)[0])
# month_index=int(np.where(month_era==month_select)[0])
# u_hourly_mean=np.array(era_file.variables['hourly_mean'][1,month_index,p_index,:,:])
# v_hourly_mean=np.array(era_file.variables['hourly_mean'][2,month_index,p_index,:,:])
# cyclic_u_hourly_mean,cyclic_lon_era=add_cyclic_point(u_hourly_mean,coord=lon_era)
# cyclic_v_hourly_mean,_=add_cyclic_point(v_hourly_mean,coord=lon_era)
# cyclic_lon_era,cyclic_lat_era=np.meshgrid(cyclic_lon_era,lat_era)






""" # single height bm 360 before qc
ba_bm_lat=np.zeros((np.size(lat_median),np.size(lon_median)))
ba_bm_select=ba_bm[count_index,:]
for i in range(0,np.size(lon_median)):
 ba_bm_lat[:,i]=np.array(ba_bm_select).flatten()
cyclic_ba_bm,_=add_cyclic_point(ba_bm_lat,coord=lon_median)
# figure
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-35,35],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(9.1,10.09,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,np.array(cyclic_ba_bm*(1000)),cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
# units
ax3=fig.add_axes([0.825,0.264,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-3}}$ rad)',family='Times New Roman',fontsize=8,color='black')
plt.savefig(str(str('D:/DA/11.4/')+str('cosmic2_2023')+month+str('_single height_lat_bm_before_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height bm 360 after qc
ba_bm_lat_after=np.zeros((np.size(lat_median),np.size(lon_median)))
for i in range(0,np.size(lon_median)):
 ba_bm_lat_after[:,i]=np.array(bm_after_select).flatten()
cyclic_ba_bm_after,_=add_cyclic_point(ba_bm_lat_after,coord=lon_median)
# figure
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-35,35],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(9.1,10.09,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,np.array(cyclic_ba_bm_after*(1000)),cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
# units
ax3=fig.add_axes([0.825,0.264,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-3}}$ rad)',family='Times New Roman',fontsize=8,color='black')
plt.savefig(str(str('D:/DA/11.4/')+str('cosmic2_2023')+month+str('_single height_lat_bm_after_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height bsd 360 before qc
ba_bsd_lat=np.zeros((np.size(lat_median),np.size(lon_median)))
ba_bsd_select=ba_bsd[count_index,:]
for i in range(0,np.size(lon_median)):
 ba_bsd_lat[:,i]=np.array(ba_bsd_select).flatten()
cyclic_ba_bsd,_=add_cyclic_point(ba_bsd_lat,coord=lon_median)
# figure
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-35,35],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(5,8.85,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,np.array(cyclic_ba_bsd*(10000)),cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
# units
ax3=fig.add_axes([0.825,0.264,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-4}}$ rad)',family='Times New Roman',fontsize=8,color='black')
plt.savefig(str(str('D:/DA/')+str('cosmic2_2023')+month+str('_single height_lat_bsd_before_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height bsd 360 after qc
ba_bsd_lat_after=np.zeros((np.size(lat_median),np.size(lon_median)))
for i in range(0,np.size(lon_median)):
 ba_bsd_lat_after[:,i]=np.array(bsd_after_select).flatten()
cyclic_ba_bsd_after,_=add_cyclic_point(ba_bsd_lat_after,coord=lon_median)
# figure
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-35,35],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(5,8.85,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,np.array(cyclic_ba_bsd_after*(10000)),cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
# units
ax3=fig.add_axes([0.825,0.264,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-4}}$ rad)',family='Times New Roman',fontsize=8,color='black')
plt.savefig(str(str('D:/DA/')+str('cosmic2_2023')+month+str('_single height_lat_bsd_after_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z mean lat 2.5 mean 360 before qc
z_mean_lat_before=np.zeros((np.size(lat_median),np.size(lon_median)))
for i in range(0,np.size(lon_median)):
 z_mean_lat_before[:,i]=np.array(z_mean_before_select).flatten()
cyclic_z_mean_before,_=add_cyclic_point(z_mean_lat_before,coord=lon_median)
# figure
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-35,35],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(0.75,0.86,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_z_mean_before,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/')+str('cosmic2_2023')+month+str('_single height_lat_zmean_before_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z mean lat 2.5 mean 360 after qc
z_mean_lat_after=np.zeros((np.size(lat_median),np.size(lon_median)))
for i in range(0,np.size(lon_median)):
 z_mean_lat_after[:,i]=np.array(z_mean_after_select).flatten()
cyclic_z_mean_after,_=add_cyclic_point(z_mean_lat_after,coord=lon_median)
# figure
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-35,35],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(0.75,0.86,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_z_mean_after,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/')+str('cosmic2_2023')+month+str('_single height_lat_zmean_after_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z std lat 2.5 mean 360 before qc
z_std_lat_before=np.zeros((np.size(lat_median),np.size(lon_median)))
for i in range(0,np.size(lon_median)):
 z_std_lat_before[:,i]=np.array(z_std_before_select).flatten()
cyclic_z_std_before,_=add_cyclic_point(z_std_lat_before,coord=lon_median)
# figure
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-35,35],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(0.59,0.7,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_z_std_before,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/11.4/lat_mean/')+str('cosmic2_2023')+month+str('_single height_lat_zstd_before_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z std lat 2.5 mean 360 after qc
z_std_lat_after=np.zeros((np.size(lat_median),np.size(lon_median)))
for i in range(0,np.size(lon_median)):
 z_std_lat_after[:,i]=np.array(z_std_after_select).flatten()
cyclic_z_std_after,_=add_cyclic_point(z_std_lat_after,coord=lon_median)
# figure
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-35,35],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(0.59,0.7,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_z_std_after,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/11.4/lat_mean/')+str('cosmic2_2023')+month+str('_single height_lat_zstd_after_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z mean lat 2.5 lon 2.5 mean 360 before qc
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
cyclic_zmean_dis_mask=np.nan_to_num(cyclic_zmean_dis_before,nan=0)
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.1f',ticks=zmean_level)
cbar.set_ticks(zmean_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km/')+str(month)+str('_')+str(ih_select)+str('_zmean.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z mean lat 2.5 lon 2.5 mean 360 after qc
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-35,35],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(0.6,1.37,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zmean_dis_after,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/11.4/lat_mean/')+str('cosmic2_2023')+month+str('_single height_latlon_zmean_after_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z std lat 2.5 lon 2.5 mean 360 before qc
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
cbar_norm=mpl.colors.BoundaryNorm(zstd_level,cbar_map.N,extend='both')
cyclic_zstd_dis_mask=np.nan_to_num(cyclic_zstd_dis_before,nan=0)
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.1f',ticks=zstd_level)
cbar.set_ticks(zstd_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km/')+str(month)+str('_')+str(ih_select)+str('_zstd.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z std lat 2.5 lon 2.5 mean 360 after qc
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-35,35],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(0.5,0.83,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zstd_dis_after,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.2f',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/11.4/lat_mean/')+str('cosmic2_2023')+month+str('_single height_latlon_zstd_after_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height total data count
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
cbar_norm=mpl.colors.BoundaryNorm(total_level,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_total_count,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.0f',ticks=total_level)
cbar.set_ticks(total_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km/')+str(month)+str('_')+str(ih_select)+str('_total_count.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z below count 360 before qc
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
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km/')+str(month)+str('_')+str(ih_select)+str('_below_count.png')),dpi=600,bbox_inches='tight',pad_inches=0)
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
z_threshold_diff=4
above_level_diff=np.linspace(10,65,12,endpoint=True)
zabove_before_diff_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
for i in range(0,np.size(lat_median)):
 lon_single_diff=lon_before_select[i]
 z_single_diff=z_before_select[i]
 for j in range(0,np.size(lon_median)):
  z_dis_single_diff=np.array(z_single_diff[np.where(np.logical_and(lon_single_diff>=lon_level[j],lon_single_diff<lon_level[j+1]))].flatten())
  zabove_before_diff_dis[i,j]=np.size(z_dis_single_diff[np.where(z_dis_single_diff>z_threshold_diff)])
cyclic_zabove_diff_before,_=add_cyclic_point(zabove_before_diff_dis,coord=lon_median)
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
cbar_norm=mpl.colors.BoundaryNorm(above_level_diff,cbar_map.N,extend='both')
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.0f',ticks=above_level_diff)
cbar.set_ticks(above_level_diff)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km/z_')+str(z_threshold_diff)+str('/')+str(month)+str('_')+str(ih_select)+str('_above_count.png')),dpi=600,bbox_inches='tight',pad_inches=0)
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


""" # bm bsd plot
ba_bm_select=ba_bm[count_index,:]
ba_bsd_select=ba_bsd[count_index,:]
fig=plt.figure(figsize=(4,3),dpi=600)
ax1=fig.add_subplot(1,1,1)
ax1.plot(lat_median,np.array(ba_bm_select*(10**3)),linewidth=0.6,color='#000000',linestyle='-')
ax1.plot(lat_median,np.array(bm_after_select*(10**3)),linewidth=0.6,color='#000000',linestyle='--')
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('$\overline{\mathregular{\u03B1}}$ (10$^{\mathregular{-3}}$ rad)',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-35,35])
ax1.set_ylim([8.5,10.5])
xticks1=np.concatenate([np.array([-35]).flatten(),np.arange(-20,20.1,10).flatten(),np.array([35]).flatten()])
yticks1=np.arange(8.5,(10.5+0.5/10),0.5)
yticks1_label=['{:.1f}'.format((yticks1[t])) for t in range(0,np.size(yticks1))]
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
yminorlocator1=MultipleLocator(((0.25)))
ax1.xaxis.set_minor_locator(xminorlocator1)
ax1.yaxis.set_minor_locator(yminorlocator1)
ax1.yaxis.set_minor_formatter(FormatStrFormatter(''))
ax1.tick_params(axis="x",direction="out",which="minor",length=2)
ax1.tick_params(axis="y",direction="out",which="minor",length=2)
ax2=ax1.twinx()
ax2.plot(lat_median,np.array(ba_bsd_select*(10**4)),linewidth=0.6,color='#0000FF',linestyle='-')
ax2.plot(lat_median,np.array(bsd_after_select*(10**4)),linewidth=0.6,color='#0000FF',linestyle='--')
ax2.set_ylabel(str(str(chr(963))+'$_{\mathregular{\u03B1}}$ (10$^{\mathregular{-4}}$ rad)'),fontdict=font_label1)
ax2.set_ylim([2,12])
yticks2=np.arange(2,(12.1+2/10),2)
yticks2_label=['{:.0f}'.format((yticks2[t])) for t in range(0,np.size(yticks2))]
ax2.set_yticks(yticks2)
labels_y2=ax2.yaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_y2]
ax2.set_yticklabels(yticks2_label)
ax2.tick_params(axis="y",direction="out",length=4,labelsize=10)
yminorlocator2=MultipleLocator(((0.5)))
ax2.yaxis.set_minor_locator(yminorlocator2)
ax2.tick_params(axis="y",direction="out",which="minor",length=2)
plt.savefig(str(str('D:/DA/11.8/')+str('/cosmic2_')+str(month)+str('_lat_bm_bsd_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # z mean std plot
fig=plt.figure(figsize=(4,3),dpi=600)
ax1=fig.add_subplot(1,1,1)
ax1.plot(lat_median,z_mean_before_select,linewidth=0.6,color='#000000',linestyle='-')
ax1.plot(lat_median,z_mean_after_select,linewidth=0.6,color='#000000',linestyle='--')
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel(str('$\overline{\mathregular{|Z|}}$'),fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-35,35])
ax1.set_ylim([0.6,1])
xticks1=np.concatenate([np.array([-35]).flatten(),np.arange(-20,20.1,10).flatten(),np.array([35]).flatten()])
yticks1=np.arange(0.6,(1.01+0.1/10),0.1)
yticks1_label=['{:.1f}'.format((yticks1[t])) for t in range(0,np.size(yticks1))]
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
yminorlocator1=MultipleLocator(((0.05)))
ax1.xaxis.set_minor_locator(xminorlocator1)
ax1.yaxis.set_minor_locator(yminorlocator1)
ax1.yaxis.set_minor_formatter(FormatStrFormatter(''))
ax1.tick_params(axis="x",direction="out",which="minor",length=2)
ax1.tick_params(axis="y",direction="out",which="minor",length=2)
ax2=ax1.twinx()
ax2.plot(lat_median,z_std_before_select,linewidth=0.6,color='#0000FF',linestyle='-')
ax2.plot(lat_median,z_std_after_select,linewidth=0.6,color='#0000FF',linestyle='--')
ax2.set_ylabel(str(str(chr(963))+str('$_{\mathregular{|Z|}}$')),fontdict=font_label1)
ax2.set_ylim([0.4,0.8])
yticks2=np.arange(0.4,(0.81+0.1/10),0.1)
yticks2_label=['{:.1f}'.format((yticks2[t])) for t in range(0,np.size(yticks2))]
ax2.set_yticks(yticks2)
labels_y2=ax2.yaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_y2]
ax2.set_yticklabels(yticks2_label)
ax2.tick_params(axis="y",direction="out",length=4,labelsize=10)
yminorlocator2=MultipleLocator(((0.05)))
ax2.yaxis.set_minor_locator(yminorlocator2)
ax2.tick_params(axis="y",direction="out",which="minor",length=2)
plt.savefig(str('D:/DA/11.4/z_mean_std_lat_dis.png'),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z above count 80-200
fig=plt.figure(figsize=((2.9,3)))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([80,200,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(80,(200+30/10),30)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(2,24,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zabove_day_before,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks,crs=ccrs.PlateCarree())
ax1.set_yticks(yticks,crs=ccrs.PlateCarree())
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(30/3)
yminorlocator=MultipleLocator(15/3)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
# ax2=fig.add_axes([0.92,0.17,0.018,0.66])
ax2=fig.add_axes([0.15,0.12,0.73,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,aspect=32,format='%.0f',orientation='horizontal',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
# plt.savefig(str(str('D:/DA/test/')+month+str('/cosmic2_')+month+str('_single height_day_lat_zabove_before_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km/')+month+str('_')+str(ih_select)+str('km_day_zabove_region')+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # trh
trh_wmo[trh_wmo==-999.0]=np.NaN
trh_lat_mean=np.zeros((np.size(lat_median)))
trh_lat_std=np.zeros((np.size(lat_median)))
trh_mean=np.zeros((np.size(lat_median),np.size(lon_median)))
trh_std=np.zeros((np.size(lat_median),np.size(lon_median)))
for i in range(0,np.size(lat_median)):
 trh_lat_mean[i]=np.nanmean(np.array(trh_wmo[np.where(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]))]).flatten())
 trh_lat_std[i]=np.nanstd(np.array(trh_wmo[np.where(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]))]).flatten())
 for j in range(0,np.size(lon_median)):
  trh_mean[i,j]=np.nanmean(np.array(trh_wmo[np.where(np.logical_and(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]),np.logical_and(lon>=lon_level[j],lon<lon_level[j+1])))]).flatten())
  trh_std[i,j]=np.nanstd(np.array(trh_wmo[np.where(np.logical_and(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]),np.logical_and(lon>=lon_level[j],lon<lon_level[j+1])))]).flatten())
# mean / std
trh_mean=np.nan_to_num(trh_mean,nan=8)
trh_std=np.nan_to_num(trh_std,nan=0)
cyclic_trh_mean,_=add_cyclic_point(trh_mean,coord=lon_median)
cyclic_trh_std,_=add_cyclic_point(trh_std,coord=lon_median)
trh_mean_level=np.linspace(9,17.8,(col_num+1),endpoint=True)
trh_std_level=np.linspace(0.5,3.8,(col_num+1),endpoint=True)
"""


""" # trh height mean
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
cbar_norm=mpl.colors.BoundaryNorm(trh_mean_level,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_trh_mean,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.1f',ticks=trh_mean_level)
cbar.set_ticks(trh_mean_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
# units
ax3=fig.add_axes([0.823,0.27,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(km)',family='Times New Roman',fontsize=8,color='black')
plt.savefig(str('D:/DA/test/cosmic2_trh_mean.png'),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # trh height std
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
cbar_norm=mpl.colors.BoundaryNorm(trh_std_level,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_trh_std,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.1f',ticks=trh_std_level)
cbar.set_ticks(trh_std_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
# units
ax3=fig.add_axes([0.823,0.27,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(km)',family='Times New Roman',fontsize=8,color='black')
plt.savefig(str('D:/DA/test/cosmic2_trh_std.png'),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # cosmic2 track daily orbit 
trans=Transformer.from_crs('EPSG:8401','EPSG:4326')
path=('D:/DA/GPS/leo/')
file_name=os.listdir(path)
lat_total=[]
lon_total=[]
alt_total=[]
try:
 sat_name=[]
 for i in range(0,len(file_name)):
  sat_name.append(int(str(file_name[i])[16:19]))
  with open(path+file_name[i], 'r+', encoding='utf-8') as file:
   loc_x=[]
   loc_y=[]
   loc_z=[]
   line = file.readlines()
   for j in range(0,len(line)):
    if 'PL' in line[j]:
     loc_x.append(np.float64(str(line[j])[6:19])*1000)
     loc_y.append(np.float64(str(line[j])[20:33])*1000)
     loc_z.append(np.float64(str(line[j])[33:46])*1000)
   loc_x=np.array(loc_x)
   loc_y=np.array(loc_y)
   loc_z=np.array(loc_z)
   lat,lon,alt=trans.transform(loc_x,loc_y,loc_z,radians=False)
   alt=alt/1000
   lon_diff=np.nanmedian(np.array([abs(lon[k+1]-lon[k]) for k in range(0,int(np.size(lon)-1))]))
   lon=lon[0:int(360/lon_diff)]
   lat=lat[0:int(360/lon_diff)]
   alt=alt[0:int(360/lon_diff)]
   lat_total.append(lat)
   lon_total.append(lon)
   alt_total.append(alt)
 sat_name=np.array(sat_name)
except FileNotFoundError:
 print("file do not exsit")
# data count
# lat_lon_count=np.zeros((np.size(lat_median),np.size(lon_median)))
# for i in range(0,np.size(lat_median)):
#  for j in range(0,np.size(lon_median)):
#   ba_index=np.array(np.where(np.logical_and(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]),np.logical_and(lon>=lon_level[j],lon<lon_level[j+1])))[0])
#   sum_count=0
#   for k in range(0,np.size(ba_index)):
#    ba_single=np.array(ba[int(ba_index),:])
#    sum_count=sum_count+np.size(ba_single[~np.isnan(ba_single)])
#    lat_lon_count[i,j]=sum_count
# orbit figure
fig=plt.figure(figsize=(7,5),dpi=600)
ax=fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(central_longitude=120, central_latitude=0))
ax.set_global()
ax.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax.plot(lon_total[0],lat_total[0],'-',transform=ccrs.Geodetic(),label='E1',color='black',linewidth=0.4)
ax.plot(lon_total[0],lat_total[0],'--',transform=ccrs.Geodetic(),label='E2',color='black',linewidth=0.4)
ax.plot(lon_total[0],lat_total[0],'-',transform=ccrs.Geodetic(),label='E3',color='blue',linewidth=0.4)
ax.plot(lon_total[0],lat_total[0],'--',transform=ccrs.Geodetic(),label='E4',color='blue',linewidth=0.4)
ax.plot(lon_total[0],lat_total[0],'-',transform=ccrs.Geodetic(),label='E5',color='orange',linewidth=0.4)
ax.plot(lon_total[0],lat_total[0],'--',transform=ccrs.Geodetic(),label='E6',color='orange',linewidth=0.4)
for i in range(0,len(lat_total),3):
 if int(sat_name[i])==1:
  ax.plot(lon_total[i],lat_total[i],'-',transform=ccrs.Geodetic(),color='black',linewidth=0.4)
 if int(sat_name[i])==2:
  ax.plot(lon_total[i],lat_total[i],'--',transform=ccrs.Geodetic(),color='black',linewidth=0.4)
 if int(sat_name[i])==3:
  ax.plot(lon_total[i],lat_total[i],'-',transform=ccrs.Geodetic(),color='blue',linewidth=0.4)
 if int(sat_name[i])==4:
  ax.plot(lon_total[i],lat_total[i],'--',transform=ccrs.Geodetic(),color='blue',linewidth=0.4)
 if int(sat_name[i])==5:
  ax.plot(lon_total[i],lat_total[i],'-',transform=ccrs.Geodetic(),color='orange',linewidth=0.4)
 if int(sat_name[i])==6:
  ax.plot(lon_total[i],lat_total[i],'--',transform=ccrs.Geodetic(),color='orange',linewidth=0.4)
font_legend={'size':9,'family':'Times New Roman','weight':'light'}
ax.legend(loc='center left',bbox_to_anchor=(1,0.5), frameon=False,fontsize=9,prop=font_legend,ncol=1,labelspacing=0.3,columnspacing=1,borderaxespad=0.2)
"""


""" # single lat z-score
lat_single_min=4
lat_single_max=7
lon360_level=np.arange(0,360.1,2.5)
lon360_median=np.array([(lon360_level[i]+lon360_level[i+1])/2 for i in range(0,int(np.size(lon360_level)-1))])
z_before_one_lat=z_before_select[np.where(np.logical_and(lat_median<lat_single_max,lat_median>lat_single_min))][0]
lon_before_one_lat=lon_before_select[np.where(np.logical_and(lat_median<lat_single_max,lat_median>lat_single_min))][0]
lon_before_one_lat[lon_before_one_lat<0]=lon_before_one_lat[lon_before_one_lat<0]+360
zmean_one_lat=np.zeros((np.size(lon360_median)))
zstd_one_lat=np.zeros((np.size(lon360_median)))
# z-level count
z_level=np.arange(0,12.01,0.1)
z_level_median=np.array([(z_level[i]+z_level[i+1])/2 for i in range(0,int(np.size(z_level)-1))])
z_level_count=np.zeros((np.size(z_level_median),np.size(lon360_median)))
lon_before_one_lat_singlez=[]
z_before_one_lat_singlez=[]
for i in range(0,np.size(z_level_median)):
 for j in range(0,np.size(lon360_median)):
  lon_before_one_lat_lon=lon_before_one_lat[np.where(np.logical_and(lon_before_one_lat>=lon360_level[j],lon_before_one_lat<lon360_level[j+1]))]
  z_before_one_lat_lon=z_before_one_lat[np.where(np.logical_and(lon_before_one_lat>=lon360_level[j],lon_before_one_lat<lon360_level[j+1]))] 
  z_level_count[i,j]=np.size(z_before_one_lat_lon[np.where(np.logical_and(z_before_one_lat_lon>=z_level[i],z_before_one_lat_lon<z_level[i+1]))])
  if z_level_count[i,j]==1.0:
   lon_before_one_lat_singlez.append(lon_before_one_lat_lon[np.where(np.logical_and(z_before_one_lat_lon>=z_level[i],z_before_one_lat_lon<z_level[i+1]))])
   z_before_one_lat_singlez.append(z_before_one_lat_lon[np.where(np.logical_and(z_before_one_lat_lon>=z_level[i],z_before_one_lat_lon<z_level[i+1]))])
lon_before_one_lat_singlez=np.array(lon_before_one_lat_singlez)
z_before_one_lat_singlez=np.array(z_before_one_lat_singlez)
# color
col_num=11
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
z_level_count_level=np.linspace(5,104,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mcolors.BoundaryNorm(z_level_count_level,cbar_map.N,extend='both')
z_count_mask=np.ma.masked_where((z_level_count<2),z_level_count)
# z mean std
for i in range(0,np.size(lon360_median)):
 zmean_one_lat[i]=np.nanmean(z_before_one_lat[np.where(np.logical_and(lon_before_one_lat>=lon360_level[i],lon_before_one_lat<lon360_level[i+1]))])
 zstd_one_lat[i]=np.nanstd(z_before_one_lat[np.where(np.logical_and(lon_before_one_lat>=lon360_level[i],lon_before_one_lat<lon360_level[i+1]))])
# single lat z-score scatter figure
fig=plt.figure(figsize=(6,3),dpi=600)
ax1=fig.add_subplot(1,1,1)
# mesh
ax1.pcolormesh(lon360_median,z_level_median,z_count_mask,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0)
# scatter
ax1.scatter(lon_before_one_lat_singlez[np.where(z_before_one_lat_singlez>np.percentile(z_before_one_lat_singlez,90))],z_before_one_lat_singlez[np.where(z_before_one_lat_singlez>np.percentile(z_before_one_lat_singlez,90))],marker='o',c='none',edgecolors='#000000',linewidths=0.4,s=6,zorder=1)
# mean std
ax1.plot(lon360_median,zmean_one_lat,linestyle='-',color='#000000',linewidth=0.6,zorder=2,label='$\overline{\mathregular{Z}}$')
# ax1.scatter(lon360_median[::4],zmean_one_lat[::4],marker='^',c='none',edgecolors='#000000',linewidths=0.4,s=8,zorder=2)
ax1.plot(lon360_median,zstd_one_lat,linestyle='--',color='#000000',linewidth=0.6,zorder=2,label=str(str(chr(963))+str('$_{\mathregular{Z}}$')))
# ax1.scatter(lon360_median[::4],zstd_one_lat[::4],marker='s',c='none',edgecolors='#000000',linewidths=0.4,s=8,zorder=2)
font_legend={'size':9,'family':'Times New Roman','weight':'light'}
ax1.legend(loc='upper right',frameon=False,facecolor='none',prop=font_legend,ncol=1,labelspacing=0.5,columnspacing=1,borderaxespad=0.2) 
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Z-score',fontdict=font_label1)
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
xticks1_label=['0$^{\mathregular{o}}$E','60$^{\mathregular{o}}$E','120$^{\mathregular{o}}$E','180$^{\mathregular{o}}$','120$^{\mathregular{o}}$W','60$^{\mathregular{o}}$W','0$^{\mathregular{o}}$W']
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,aspect=60,format='%.0f',ticks=z_level_count_level)
cbar.set_ticks(z_level_count_level)
labels_cbar=cbar.ax.yaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=True)
cbar.ax.tick_params(length=1.5)
plt.savefig(str('D:/DA/test/scatter_line/cosmic2_lat_5N_scatter.png'),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # zmean after QC
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
cyclic_zmean_dis_mask=np.nan_to_num(cyclic_zmean_dis_after,nan=0)
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
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.1f',ticks=zmean_level)
cbar.set_ticks(zmean_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km/')+str(month)+str('_')+str(ih_select)+str('_zmean_after.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # zstd after QC
zstd_level_after=np.linspace(0.5,0.72,(col_num+1),endpoint=True)
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
cyclic_zstd_dis_mask=np.nan_to_num(cyclic_zstd_dis_after,nan=0)
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
plt.savefig(str(str('D:/DA/test/')+str(ih_select)+str('km/')+str(month)+str('_')+str(ih_select)+str('_zstd_after2.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""



# article figure


""" # z above count level
z_count_level=np.concatenate([np.array([100]).flatten(),np.linspace(200,3500,12,endpoint=True).flatten()])
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
# col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_list=["#E8EBDA","#EAE3C7","#C8E1B7","#AED7B7","#83CDAC","#58BEA6","#12B7A9","#00B7B6","#01B4CA","#2094c7","#697FBB","#8366A9","#92509A","#A93391"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(z_count_level,c_map.N,extend='both')
z_mask=z_above_count_before
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
# ax2.plot(np.array(z_out_lat_count)/(10**4),msl_fig,linewidth=0.6,color='#000000',linestyle='-',zorder=2)
ax2.plot(np.nansum(z_mask,axis=1)/(10**4),msl_fig,linewidth=0.6,color='#000000',linestyle='-',zorder=2)
# 坐标轴标签
# ax2.set_xlabel(str(str('|Z|$_{\mathregular{\u03B1}}$ > ')+str(z_threshold)+str(' Count')),fontdict=font_label1)
ax2.set_xlabel(str('Outlier (10$^{\mathregular{4}}$)'),fontdict=font_label1)
# 刻度(主+次)
ax2.set_xlim([0,8])
xticks2=np.arange(0,8.1,2)
ax2.set_xticks(xticks2)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
ax2.tick_params(axis="x",direction="out",length=4)
xminorlocator2=MultipleLocator((2/4))
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
plt.savefig(str(str('D:/DA/publish/MWR/fig/fig2a.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # z below count level
z_count_level=np.concatenate([np.array([10000]).flatten(),np.array(np.linspace(20000,130000,12,endpoint=True)).flatten()])
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
# col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_list=["#E8EBDA","#EAE3C7","#C8E1B7","#AED7B7","#83CDAC","#58BEA6","#12B7A9","#00B7B6","#01B4CA","#2094c7","#697FBB","#8366A9","#92509A","#A93391"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(z_count_level,c_map.N,extend='both')
z_mask=z_below_count_before
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
ax2.plot(np.array(np.sum(z_below_count_before,axis=1))/(10**5),msl_fig,linewidth=0.6,color='#000000',linestyle='-',zorder=2)
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
cbar_label=[str(int(z_count_level[i]/10000)) if i%2==0 else ' ' for i in range(0,np.size(z_count_level))]
# cbar.ax.set_yticklabels(cbar_label)
cbar.ax.set_xticklabels(cbar_label)
cbar.ax.tick_params(labelsize=7)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=True)
cbar.ax.tick_params(length=1.5)
ax3=fig.add_axes([0.85,-0.012,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{4}}$)',family='Times New Roman',fontsize=7,color='black')
plt.savefig(str(str('D:/DA/publish/MWR/fig/fig2b.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # bm level
bm_level=np.linspace(0.000,0.028,15,endpoint=True)
bm_level=np.concatenate([bm_level.flatten(),np.array([0.032]).flatten()])
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
# col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA","#8B008B"]
col_list=["#E8EBDA","#EAE3C7","#C8E1B7","#AED7B7","#83CDAC","#58BEA6","#12B7A9","#00B7B6","#01B4CA","#2094c7","#697FBB","#8366A9","#92509A","#A93391","#85136F"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(bm_level,c_map.N)
bm_mask=ba_bm.copy()
bm_mask_after=bm_after.copy()
msl_fig=msl_median.copy()
msl_fig[0]=0
msl_fig[-1]=20
# before
# bm_mask=np.ma.masked_where((ba_bm<(1.5*10**(-3))),ba_bm)
ax1.pcolormesh(lat_median,msl_fig,bm_mask,cmap=c_map,norm=c_norm,shading='nearest',zorder=0)
# after
bm_after_level=np.linspace(0.0,0.028,15,endpoint=True)
bm_fig_after=ax1.contour(lat_median,msl_fig,bm_mask_after*(10**3),bm_after_level*(10**3),colors='black',linewidths=0.4,zorder=1)
bm_label=plt.clabel(bm_fig_after,np.array([4,8,12,16,28]),inline=True,fontsize=7)
[label.set_fontname('Times New Roman') for label in bm_label]
# axis
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Height (km)',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-45,45])
ax1.set_ylim([0,20])
# xticks1=np.concatenate([np.array([-35]).flatten(),np.arange(-20,20.1,10).flatten(),np.array([35]).flatten()])
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
ax2.plot(np.array(bm_mean*(10**3)),msl_fig,linewidth=0.6,color='#FFFFFF',linestyle='-',zorder=2)
ax2.plot(np.array(bm_mean_after*(10**3)),msl_fig,linewidth=0.6,color='#FFFFFF',linestyle='--',zorder=3)
# 坐标轴标签
# ax2.set_xlabel('$\overline{\mathregular{\u03B1}}$ (10$^{\mathregular{-3}}$ rad)',fontdict=font_label1)
ax2.set_xlabel(str('\u03bc'+'$_{\mathregular{\u03B1}}$ (10$^{\mathregular{-3}}$ rad)'),fontdict=font_label1)
# 刻度(主+次)
ax2.set_xlim([0,30])
xticks2=np.arange(0,31,6)
ax2.set_xticks(xticks2)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
ax2.tick_params(axis="x",direction="out",length=4)
xminorlocator2=MultipleLocator((2))
ax2.xaxis.set_minor_locator(xminorlocator2)
ax2.tick_params(axis="x",direction="out",which="minor",length=2)
# colorbar
ax4=fig.add_axes([0.18,0.03,0.67,0.018])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=c_norm,cmap=c_map),cax=ax4,aspect=60,format='%.1f',ticks=(bm_level),orientation='horizontal')
cbar.set_ticks(bm_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar_label=[str(int(bm_level[i]*1000)) for i in range(0,np.size(bm_level))]
cbar.ax.set_xticklabels(cbar_label)
cbar.ax.tick_params(labelsize=7)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
ax3=fig.add_axes([0.871,-0.012,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-3}}$ rad)',family='Times New Roman',fontsize=7,color='black')
plt.savefig(str(str('D:/DA/publish/MWR/fig/fig3a.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # bsd level
bsd_level=np.linspace(2,67,14,endpoint=True)
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
# col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA","#8B008B"]
col_list=["#E8EBDA","#EAE3C7","#C8E1B7","#AED7B7","#83CDAC","#58BEA6","#12B7A9","#00B7B6","#01B4CA","#2094c7","#697FBB","#8366A9","#92509A","#A93391","#85136F"]
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(bsd_level,c_map.N,extend='both')
bsd_mask=ba_bsd.copy()
bsd_mask_after=bsd_after.copy()
msl_fig=msl_median.copy()
msl_fig[0]=0
msl_fig[-1]=20
# before
ax1.pcolormesh(lat_median,msl_fig,(bsd_mask*(10**4)),cmap=c_map,norm=c_norm,shading='nearest',zorder=0)
# after
bsd_after_level=np.linspace(2,62,13,endpoint=True)
bsd_fig_after=ax1.contour(lat_median,msl_fig,bsd_mask_after*(10**4),bsd_after_level,colors='black',linewidths=0.4,zorder=1)
bsd_label=plt.clabel(bsd_fig_after,np.array([2,12,22,42]),inline=True,fontsize=7)
[label.set_fontname('Times New Roman') for label in bsd_label]
# axis
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Height (km)',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([-45,45])
ax1.set_ylim([0,20])
# xticks1=np.concatenate([np.array([-35]).flatten(),np.arange(-20,20.1,10).flatten(),np.array([35]).flatten()])
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
# bsd_mean=scipy.signal.savgol_filter(bsd_mean,100,4) 
ax2.plot(np.array(bsd_mean*(10**4)),msl_fig,linewidth=0.6,color='#FFFFFF',linestyle='-')
ax2.plot(np.array(bsd_mean_after*(10**4)),msl_fig,linewidth=0.6,color='#FFFFFF',linestyle='--')
# 坐标轴标签
ax2.set_xlabel(str(str(chr(963))+'$_{\mathregular{\u03B1}}$ (10$^{\mathregular{-4}}$ rad)'),fontdict=font_label1)
# 刻度(主+次)
ax2.set_xlim([0,60])
xticks2=np.arange(0,60.1,15)
ax2.set_xticks(xticks2)
plt.xticks(fontproperties='Times New Roman',fontsize=10)
ax2.tick_params(axis="x",direction="out",length=4)
xminorlocator2=MultipleLocator((5))
ax2.xaxis.set_minor_locator(xminorlocator2)
ax2.tick_params(axis="x",direction="out",which="minor",length=2)
# colorbar
ax4=fig.add_axes([0.18,0.03,0.67,0.018])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=c_norm,cmap=c_map),cax=ax4,aspect=60,format='%.0f',ticks=(bsd_level),orientation='horizontal',extend='both')
cbar.set_ticks(bsd_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
# cbar_label=[str(int(bsd_level[i]*1000)) for i in range(0,np.size(bsd_level))]
# cbar.ax.set_xticklabels(cbar_label)
cbar.ax.tick_params(labelsize=7)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
ax3=fig.add_axes([0.85,-0.012,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{-4}}$ rad)',family='Times New Roman',fontsize=7,color='black')
plt.savefig(str(str('D:/DA/publish/MWR/fig/fig3b.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single lat z-score
lat_single_min=5
lat_single_max=8
lon360_level=np.arange(0,360.1,2.5)
lon360_median=np.array([(lon360_level[i]+lon360_level[i+1])/2 for i in range(0,int(np.size(lon360_level)-1))])
z_before_one_lat=z_before_select[np.where(np.logical_and(lat_median<lat_single_max,lat_median>lat_single_min))][0]
lon_before_one_lat=lon_before_select[np.where(np.logical_and(lat_median<lat_single_max,lat_median>lat_single_min))][0]
lon_before_one_lat[lon_before_one_lat<0]=lon_before_one_lat[lon_before_one_lat<0]+360
zmean_one_lat=np.zeros((np.size(lon360_median)))
zstd_one_lat=np.zeros((np.size(lon360_median)))
# z-level count
z_level=np.arange(0,12.01,0.1)
z_level_median=np.array([(z_level[i]+z_level[i+1])/2 for i in range(0,int(np.size(z_level)-1))])
z_level_count=np.zeros((np.size(z_level_median),np.size(lon360_median)))
lon_before_one_lat_singlez=[]
z_before_one_lat_singlez=[]
for i in range(0,np.size(z_level_median)):
 for j in range(0,np.size(lon360_median)):
  lon_before_one_lat_lon=lon_before_one_lat[np.where(np.logical_and(lon_before_one_lat>=lon360_level[j],lon_before_one_lat<lon360_level[j+1]))]
  z_before_one_lat_lon=z_before_one_lat[np.where(np.logical_and(lon_before_one_lat>=lon360_level[j],lon_before_one_lat<lon360_level[j+1]))] 
  z_level_count[i,j]=np.size(z_before_one_lat_lon[np.where(np.logical_and(z_before_one_lat_lon>=z_level[i],z_before_one_lat_lon<z_level[i+1]))])
  if z_level_count[i,j]==1.0:
   lon_before_one_lat_singlez.append(lon_before_one_lat_lon[np.where(np.logical_and(z_before_one_lat_lon>=z_level[i],z_before_one_lat_lon<z_level[i+1]))])
   z_before_one_lat_singlez.append(z_before_one_lat_lon[np.where(np.logical_and(z_before_one_lat_lon>=z_level[i],z_before_one_lat_lon<z_level[i+1]))])
lon_before_one_lat_singlez=np.array(lon_before_one_lat_singlez)
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
# z mean std
for i in range(0,np.size(lon360_median)):
 zmean_one_lat[i]=np.nanmean(z_before_one_lat[np.where(np.logical_and(lon_before_one_lat>=lon360_level[i],lon_before_one_lat<lon360_level[i+1]))])
 zstd_one_lat[i]=np.nanstd(z_before_one_lat[np.where(np.logical_and(lon_before_one_lat>=lon360_level[i],lon_before_one_lat<lon360_level[i+1]))])
# single lat z-score scatter figure
fig=plt.figure(figsize=(6,3),dpi=600)
ax1=fig.add_subplot(1,1,1)
# mesh
ax1.pcolormesh(lon360_median,z_level_median,z_count_mask,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0)
# scatter
ax1.scatter(lon_before_one_lat_singlez[np.where(z_before_one_lat_singlez>np.percentile(z_before_one_lat_singlez,60))],z_before_one_lat_singlez[np.where(z_before_one_lat_singlez>np.percentile(z_before_one_lat_singlez,60))],marker='o',c='none',edgecolors='#000000',linewidths=0.4,s=6,zorder=1)
# mean std
# ax1.plot(lon360_median,zmean_one_lat,linestyle='-',color='#000000',linewidth=0.8,zorder=2,label='$\overline{\mathregular{Z}}$')
ax1.plot(lon360_median,zmean_one_lat,linestyle='-',color='#000000',linewidth=0.8,zorder=2,label=str('\u03bc'+'$_{\mathregular{Z}}$'))
# ax1.scatter(lon360_median[::4],zmean_one_lat[::4],marker='^',c='none',edgecolors='#000000',linewidths=0.4,s=8,zorder=2)
ax1.plot(lon360_median,zstd_one_lat,linestyle='--',color='#000000',linewidth=0.8,zorder=2,label=str(str(chr(963))+str('$_{\mathregular{Z}}$')))
# ax1.scatter(lon360_median[::4],zstd_one_lat[::4],marker='s',c='none',edgecolors='#000000',linewidths=0.4,s=8,zorder=2)
font_legend={'size':9,'family':'Times New Roman','weight':'light'}
ax1.legend(loc='upper right',frameon=False,facecolor='none',prop=font_legend,ncol=1,labelspacing=0.5,columnspacing=1,borderaxespad=0.2) 
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel(str('Z ('+'${\mathregular{'+str(chr(966))+'}}$)'),fontdict=font_label1)
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
# plt.savefig(str('D:/DA/article/cosmic2_lat_30N_scatter2.png'),dpi=600,bbox_inches='tight',pad_inches=0)
plt.savefig(str(str('D:/DA/publish/MWR/fig/fig5c.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""



""" # single height z above count 360 before qc
msl_min=7
msl_max=8
index_bottom=int(np.array([i for i in range(0,np.size(msl_median)) if msl_median[i]<=msl_min and msl_median[i+1]>=msl_min])[0])
index_top=int(int(np.array([i for i in range(0,np.size(msl_median)) if msl_median[i]<=msl_max and ih_median[i+1]>=msl_max])[0])+1)
# day
z_scale_select=z_total_before[index_bottom:index_top,:]
lon_scale_select=lon_total_before[index_bottom:index_top,:]
zabove_scale_before_dis=np.zeros((np.size(z_scale_select,0),np.size(lat_median),np.size(lon_median)))
for i in range(0,np.size(z_scale_select,0)):
 z_scale_select_single=z_scale_select[i,:]
 lon_scale_select_single=lon_scale_select[i,:]
 for j in range(0,np.size(lat_median)):
  lon_single=lon_scale_select_single[j]
  z_single=z_scale_select_single[j]
  for k in range(0,np.size(lon_median)):
   z_dis_single=np.array(z_single[np.where(np.logical_and(lon_single>=lon_level[k],lon_single<lon_level[k+1]))].flatten())
   zabove_scale_before_dis[i,j,k]=np.size(z_dis_single[np.where(z_dis_single>z_threshold)])
zabove_scale_dis_total=np.nansum(zabove_scale_before_dis,axis=0)
cyclic_zabove_scale_dis,_=add_cyclic_point(zabove_scale_dis_total,coord=lon_median)

# different z threshold
# z_threshold_diff=4
# above_level_diff=np.linspace(50,270,12,endpoint=True)
above_level_diff=np.linspace(10,120,12,endpoint=True)
# zabove_before_diff_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
# for i in range(0,np.size(lat_median)):
#  lon_single_diff=lon_before_select[i]
#  z_single_diff=z_before_select[i]
#  for j in range(0,np.size(lon_median)):
#   z_dis_single_diff=np.array(z_single_diff[np.where(np.logical_and(lon_single_diff>=lon_level[j],lon_single_diff<lon_level[j+1]))].flatten())
#   zabove_before_diff_dis[i,j]=np.size(z_dis_single_diff[np.where(z_dis_single_diff>z_threshold_diff)])
# cyclic_zabove_diff_before,_=add_cyclic_point(zabove_before_diff_dis,coord=lon_median)
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# colormesh
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(above_level_diff,cbar_map.N,extend='both')
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
ax2=fig.add_axes([0.2,0.3,0.63,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.0f',ticks=above_level_diff)
cbar.set_ticks(above_level_diff)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/article/')+str('z_')+str(z_threshold)+str('_')+str(month)+str('_')+str(msl_min)+str('_')+str(msl_max)+str('_above_count.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height total data count
msl_min=2
msl_max=4
index_bottom=int(np.array([i for i in range(0,np.size(msl_median)) if msl_median[i]<=msl_min and msl_median[i+1]>=msl_min])[0])
index_top=int(int(np.array([i for i in range(0,np.size(msl_median)) if msl_median[i]<=msl_max and ih_median[i+1]>=msl_max])[0])+1)
# day
z_scale_select=z_total_before[index_bottom:index_top,:]
lon_scale_select=lon_total_before[index_bottom:index_top,:]
z_scale_before_dis=np.zeros((np.size(z_scale_select,0),np.size(lat_median),np.size(lon_median)))
for i in range(0,np.size(z_scale_select,0)):
 z_scale_select_single=z_scale_select[i,:]
 lon_scale_select_single=lon_scale_select[i,:]
 for j in range(0,np.size(lat_median)):
  lon_single=lon_scale_select_single[j]
  z_single=z_scale_select_single[j]
  for k in range(0,np.size(lon_median)):
   z_dis_single=np.array(z_single[np.where(np.logical_and(lon_single>=lon_level[k],lon_single<lon_level[k+1]))].flatten())
   z_scale_before_dis[i,j,k]=np.size(z_dis_single)
z_scale_dis_total=np.nansum(z_scale_before_dis,axis=0)
cyclic_z_scale_dis,_=add_cyclic_point(z_scale_dis_total,coord=lon_median)

# total_level_diff=np.linspace(5,137,12,endpoint=True)
total_level_diff=np.linspace(5,65,6,endpoint=True)
fig=plt.figure(figsize=(6,4.5))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(-180,(180+60/10),60)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
# col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805"]
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(total_level_diff,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_z_scale_dis/(10**2),cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
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
ax2=fig.add_axes([0.4,0.3,0.3,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.0f',ticks=total_level_diff)
cbar.set_ticks(total_level_diff)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=8)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
# units
# ax3=fig.add_axes([0.822,0.268,0.01,0.01])
ax3=fig.add_axes([0.698,0.268,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{2}}$)',family='Times New Roman',fontsize=8,color='black')
plt.savefig(str(str('D:/DA/article/')+str(month)+str('_')+str(msl_min)+str('_')+str(msl_max)+str('_total_count.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # lat z scatter
lat_scatter_level=np.arange(-45,(45+10/10),10)
col_list=["#06F3FB","#037DF9","#034BF0","#08B81F","#75D608","#F0F20D","#F8BB08","#F48B06","#F42103"]
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
c_map=mcolors.ListedColormap(col_list)
c_norm=mcolors.BoundaryNorm(lat_scatter_level,c_map.N)
z_mask=z_below_count_before
# bm_mask=np.ma.masked_where((ba_bm<(1.5*10**(-3))),ba_bm)
msl_fig=msl_median.copy()
msl_fig[0]=0
msl_fig[-1]=20
for i in range(0,np.size(z_total_before,0)):
 for j in range(0,np.size(z_total_before,1)):
  z_scatter_single=np.array(z_total_before[i,j])
  z_scatter_single=z_scatter_single[np.where(z_scatter_single>=10)]
  ax1.scatter(z_scatter_single,np.array(np.ones((np.size(z_scatter_single)))*msl_fig[i]),marker='o',s=4,c='none',edgecolors=col_list[int(j/4)],linewidths=0.4)
# axis
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Height (km)',fontdict=font_label1)
ax1.set_xlabel('Z',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([10,30])
ax1.set_ylim([0,20])
xticks1=np.arange(10,(30+4/10),4)
yticks1=np.arange(0,(20+4/10),4)
yticks1_label=[str(int(yticks1[t])) for t in range(0,np.size(yticks1))]
ax1.set_xticks(xticks1)
plt.xticks(fontproperties='Times New Roman')
ax1.set_yticks(yticks1)
labels_y=ax1.yaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_y]
ax1.set_yticklabels(yticks1_label)
ax1.tick_params(axis="x",direction="out",length=4,labelsize=10)
ax1.tick_params(axis="y",direction="out",length=4,labelsize=10)
xminorlocator1=MultipleLocator(4/4)
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
# colorbar
ax4=fig.add_axes([0.18,-0.04,0.67,0.018])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=c_norm,cmap=c_map),cax=ax4,aspect=60,format='%.0f',ticks=(lat_scatter_level),orientation='horizontal')
cbar.set_ticks(lat_scatter_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar_label=[str(str(int(abs(lat_scatter_level[i])))+'${\mathregular{\u00B0}}$'+str('N')) if lat_scatter_level[i]>=0 else str(str(int(abs(lat_scatter_level[i])))+'${\mathregular{\u00B0}}$'+str('S')) for i in range(0,np.size(lat_scatter_level))]
cbar.ax.set_xticklabels(cbar_label)
cbar.ax.tick_params(labelsize=6)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str(str('D:/DA/article/')+month+str('_z_lat_scatter.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # msl z count
z_level=np.arange(0,(12+0.1/10),0.1)
z_level_median=np.array([(z_level[i]+z_level[i+1])/2 for i in range(0,(np.size(z_level)-1))])
z_count_height=np.zeros((np.size(ih_median),np.size(z_level_median)))
for i in range(0,np.size(z_total_before,0)):
 z_single_height=z_total_before[i,:]
 z_single_height_flatten=np.array([0]).flatten()
 for j in range(0,np.size(z_total_before,1)):
  z_single_height_flatten=np.concatenate([z_single_height_flatten.flatten(),np.array(z_single_height[j]).flatten()])
 for k in range(0,np.size(z_level_median)):
  z_count_height[i,k]=np.size(z_single_height_flatten[np.where(np.logical_and(z_single_height_flatten>=z_level[k],z_single_height_flatten<z_level[k+1]))])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
# z_level_count_level=np.linspace(0,14,(col_num+1),endpoint=True)
# z_level_count_level=np.concatenate([np.array([1]).flatten(),z_level_count_level.flatten()])
z_level_count_level=np.concatenate([np.array([0]).flatten(),np.array(np.linspace(1,14,14,endpoint=True)).flatten()])
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mcolors.BoundaryNorm(z_level_count_level,cbar_map.N,extend='neither')
z_count_mask=z_count_height/(10**4)
# z_count_mask=np.ma.masked_where((z_count_height<1),z_count_height)
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
msl_fig=msl_median.copy()
msl_fig[0]=0
msl_fig[-1]=20
# mesh
ax1.pcolormesh(z_level_median,msl_fig,z_count_mask,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0)
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Height (km)',fontdict=font_label1)
ax1.set_xlabel('Z-score',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([0,5])
ax1.set_ylim([0,20])
xticks1=np.arange(0,(5+1/10),1)
yticks1=np.arange(0,(20+4/10),4)
# 刻度
yticks1_label=[str(int(yticks1[t])) for t in range(0,np.size(yticks1))]
ax1.set_xticks(xticks1)
# plt.xticks(fontproperties='Times New Roman',fontsize=10)
# ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
xticks1_label=[str(int(xticks1[t])) for t in range(0,np.size(xticks1))]
labels_x=ax1.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_x]
ax1.set_xticklabels(xticks1_label)
ax1.tick_params(axis="x",direction="out",length=4,labelsize=10)
xminorlocator=MultipleLocator(1/4)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.set_yticks(yticks1)
labels_y=ax1.yaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_y]
ax1.set_yticklabels(yticks1_label)
ax1.tick_params(axis="y",direction="out",length=4,labelsize=10)
yminorlocator1=MultipleLocator(((4/4)))
ax1.yaxis.set_minor_locator(yminorlocator1)
ax1.tick_params(axis="x",direction="out",which="minor",length=2)
ax1.tick_params(axis="y",direction="out",which="minor",length=2)
ax2=fig.add_axes([0.18,-0.04,0.67,0.018])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,aspect=60,format='%.0f',ticks=(z_level_count_level),orientation='horizontal',extend='neither')
cbar.set_ticks(z_level_count_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=6)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str('D:/DA/article/cosmic2_msl_z_count.png'),dpi=600,bbox_inches='tight',pad_inches=0)  
"""


""" # lat z count
z_level=np.arange(0,(12+0.1/10),0.1)
z_level_median=np.array([(z_level[i]+z_level[i+1])/2 for i in range(0,(np.size(z_level)-1))])
z_count_height=np.zeros((np.size(lat_median),np.size(z_level_median)))
for i in range(0,np.size(z_total_before,1)):
 z_single_lat=z_total_before[:,i]
 z_single_lat_flatten=np.array([0]).flatten()
 for j in range(0,np.size(z_total_before,0)):
  z_single_lat_flatten=np.concatenate([z_single_lat_flatten.flatten(),np.array(z_single_lat[j]).flatten()])
 for k in range(0,np.size(z_level_median)):
  z_count_height[i,k]=np.size(z_single_lat_flatten[np.where(np.logical_and(z_single_lat_flatten>=z_level[k],z_single_lat_flatten<z_level[k+1]))])
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
# z_level_count_level=np.linspace(0,14,(col_num+1),endpoint=True)
# z_level_count_level=np.concatenate([np.array([1]).flatten(),z_level_count_level.flatten()])
z_level_count_level=np.concatenate([np.array([0]).flatten(),np.array(np.linspace(2,15,14,endpoint=True)).flatten()])
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mcolors.BoundaryNorm(z_level_count_level,cbar_map.N,extend='neither')
z_count_mask=z_count_height/(10**4)
# z_count_mask=np.ma.masked_where((z_count_height<1),z_count_height)
fig=plt.figure(figsize=(2.7,3.2),dpi=600)
ax1=fig.add_subplot(1,1,1)
# mesh
ax1.pcolormesh(z_level_median,lat_median,z_count_mask,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0)
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_xlabel('Z-score',fontdict=font_label1)
# 刻度(主+次)
ax1.set_xlim([0,5])
ax1.set_ylim([-45,45])
xticks1=np.arange(0,(5+1/10),1)
yticks1=np.arange(-45,(45+15/10),15)
# 刻度
yticks1_label=[str(int(yticks1[t])) for t in range(0,np.size(yticks1))]
ax1.set_xticks(xticks1)
# plt.xticks(fontproperties='Times New Roman',fontsize=10)
# ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
xticks1_label=[str(int(xticks1[t])) for t in range(0,np.size(xticks1))]
labels_x=ax1.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_x]
ax1.set_xticklabels(xticks1_label)
ax1.tick_params(axis="x",direction="out",length=4,labelsize=10)
xminorlocator=MultipleLocator(1/4)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.set_yticks(yticks1)
# labels_y=ax1.yaxis.get_ticklabels()
# [label.set_fontname('Times New Roman') for label in labels_y]
# ax1.set_yticklabels(yticks1_label)
# ax1.tick_params(axis="y",direction="out",length=4,labelsize=10)
ax1.yaxis.set_major_formatter(LatitudeFormatter())
plt.yticks(fontproperties='Times New Roman')
yminorlocator1=MultipleLocator(((15/3)))
ax1.yaxis.set_minor_locator(yminorlocator1)
ax1.tick_params(axis="x",direction="out",which="minor",length=2)
ax1.tick_params(axis="y",direction="out",which="minor",length=2)
ax2=fig.add_axes([0.18,-0.04,0.67,0.018])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,aspect=60,format='%.0f',ticks=(z_level_count_level),orientation='horizontal',extend='neither')
cbar.set_ticks(z_level_count_level)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=6)
cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
plt.savefig(str('D:/DA/article/cosmic2_lat_z_count.png'),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # single height z scatter
z_single_select=z_total_before[count_index,:]
lat_single_select=lat_total_before[count_index,:]
z_single_select_flatten=np.array([0]).flatten()
lat_single_select_flatten=np.array([0]).flatten()
for i in range(0,np.size(z_single_select)):
 z_single_select_flatten=np.concatenate([z_single_select_flatten,np.array(z_single_select[i]).flatten()])
 lat_single_select_flatten=np.concatenate([lat_single_select_flatten,np.array(lat_single_select[i]).flatten()])
z_single_select_flatten=z_single_select_flatten[1:]
lat_single_select_flatten=lat_single_select_flatten[1:]
# z level
z_level=np.arange(0,(12+0.1/10),0.1)
z_level_median=np.array([(z_level[i]+z_level[i+1])/2 for i in range(0,(np.size(z_level)-1))])
z_single_height_total=np.array(np.zeros((np.size(z_level_median),np.size(lat_median))),dtype='object')
lat_single_height_total=np.array(np.zeros((np.size(z_level_median),np.size(lat_median))),dtype='object')
z_single_height_count=np.zeros((np.size(z_level_median),np.size(lat_median)))
for i in range(0,np.size(z_level_median)):
 for j in range(0,np.size(lat_median)):
  z_single_height_total[i,j]=z_single_select_flatten[np.where(np.logical_and(np.logical_and(lat_single_select_flatten>=lat_level[j],lat_single_select_flatten<lat_level[j+1]),np.logical_and(z_single_select_flatten>=z_level[i],z_single_select_flatten<z_level[i+1])))]
  lat_single_height_total[i,j]=lat_single_select_flatten[np.where(np.logical_and(np.logical_and(lat_single_select_flatten>=lat_level[j],lat_single_select_flatten<lat_level[j+1]),np.logical_and(z_single_select_flatten>=z_level[i],z_single_select_flatten<z_level[i+1])))]
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
plt.savefig(str(str('D:/DA/article/')+month+str('_')+str(ih_select)+str('_z_lat_scatter.png')),dpi=600,bbox_inches='tight',pad_inches=0)
"""


""" # 7-9 km
msl_min=6
msl_max=8
index_bottom=int(np.array([i for i in range(0,np.size(msl_median)) if msl_median[i]<=msl_min and msl_median[i+1]>=msl_min])[0])
index_top=int(int(np.array([i for i in range(0,np.size(msl_median)) if msl_median[i]<=msl_max and ih_median[i+1]>=msl_max])[0])+1)

# z_threshold_diff=3.5
# # above_level_diff=np.linspace(10,130,12,endpoint=True)
# above_level_diff=np.linspace(10,97,8,endpoint=True)
# zabove_before_diff_dis=np.zeros((np.size(lat_median),np.size(lon_median)))
# for i in range(0,np.size(lat_median)):
#  lon_single_diff=lon_before_select[i]
#  z_single_diff=z_before_select[i]
#  for j in range(0,np.size(lon_median)):
#   z_dis_single_diff=np.array(z_single_diff[np.where(np.logical_and(lon_single_diff>=lon_level[j],lon_single_diff<lon_level[j+1]))].flatten())
#   zabove_before_diff_dis[i,j]=np.size(z_dis_single_diff[np.where(z_dis_single_diff>z_threshold_diff)])
# cyclic_zabove_diff_before,_=add_cyclic_point(zabove_before_diff_dis,coord=lon_median)
# day
# soz_scale_select=soz_total_before[index_bottom:index_top,:]
z_scale_select=z_total_before[index_bottom:index_top,:]
lon_scale_select=lon_total_before[index_bottom:index_top,:]

zabove_scale_day_before_dis=np.zeros((np.size(z_scale_select,0),np.size(lat_median),np.size(lon_median)))

for i in range(0,np.size(z_scale_select,0)):
 # soz_scale_select_single=soz_scale_select[i,:]
 z_scale_select_single=z_scale_select[i,:]
 lon_scale_select_single=lon_scale_select[i,:]
 for j in range(0,np.size(lat_median)):
  lon_single=lon_scale_select_single[j]
  # soz_single=soz_scale_select_single[j]
  z_single=z_scale_select_single[j]
  # lon_day_single=lon_single[np.where(soz_single<90)]
  # soz_day_single=soz_single[np.where(soz_single<90)]
  # z_day_single=z_single[np.where(soz_single<90)]
  # for k in range(0,np.size(lon_median)):
   # z_day_dis_single=np.array(z_day_single[np.where(np.logical_and(lon_day_single>=lon_level[k],lon_day_single<lon_level[k+1]))].flatten())
   # zabove_scale_day_before_dis[i,j,k]=np.size(z_day_dis_single[np.where(z_day_dis_single>z_threshold)])


zabove_scale_day_dis_total=np.nansum(zabove_scale_day_before_dis,axis=0)


cyclic_zabove_scale_dis_after,_=add_cyclic_point(zabove_scale_day_dis_total,coord=lon_median)
fig=plt.figure(figsize=((2.9,3)))
ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
ax1.set_extent([80,200,-45,45],crs=ccrs.PlateCarree())
xticks=np.arange(80,(200+30/10),30)
# yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
yticks=np.arange(-45,(45+15/10),15)
# color
col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
col_num=11
levels_count=np.linspace(50,820,(col_num+1),endpoint=True)
cbar_map=mcolors.ListedColormap(col_list)
cbar_norm=mpl.colors.BoundaryNorm(levels_count,cbar_map.N,extend='both')
ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zabove_scale_dis_after,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# 刻度
ax1.set_xticks(xticks,crs=ccrs.PlateCarree())
ax1.set_yticks(yticks,crs=ccrs.PlateCarree())
plt.xticks(fontproperties='Times New Roman',fontsize=10)
plt.yticks(fontproperties='Times New Roman',fontsize=10)
ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
xminorlocator=MultipleLocator(30/3)
yminorlocator=MultipleLocator(15/3)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.yaxis.set_minor_locator(yminorlocator)
# colorbar
# ax2=fig.add_axes([0.92,0.17,0.018,0.66])
ax2=fig.add_axes([0.15,0.12,0.73,0.016])
cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,aspect=32,format='%.0f',orientation='horizontal',ticks=levels_count)
cbar.set_ticks(levels_count)
labels_cbar=cbar.ax.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_cbar]
cbar.ax.tick_params(labelsize=6)
cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
cbar.ax.tick_params(length=1.5)
# plt.savefig(str(str('D:/DA/test/')+month+str('/cosmic2_')+month+str('_single height_day_lat_zabove_before_')+str(ih_select)+str('.png')),dpi=600,bbox_inches='tight',pad_inches=0)
plt.savefig(str('D:/DA/article/zabove_region_4_6km.png'),dpi=600,bbox_inches='tight',pad_inches=0)
"""



# z_count_level=np.concatenate([np.array([100]).flatten(),np.linspace(200,3500,12,endpoint=True).flatten()])
# fig=plt.figure(figsize=(2.7,3.2),dpi=600)
# ax1=fig.add_subplot(1,1,1)
# col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#0000FF","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
# c_map=mcolors.ListedColormap(col_list)
# c_norm=mcolors.BoundaryNorm(z_count_level,c_map.N,extend='both')
# z_mask=z_above_diff
# msl_fig=msl_median.copy()
# msl_fig[0]=0
# msl_fig[-1]=20
# ax1.pcolormesh(lat_median,msl_fig,z_mask,cmap=c_map,norm=c_norm,shading='nearest',zorder=0)
# font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
# ax1.set_ylabel('Height (km)',fontdict=font_label1)
# # 刻度(主+次)
# ax1.set_xlim([-45,45])
# ax1.set_ylim([0,20])
# xticks1=np.arange(-45,(45+15/10),15)
# yticks1=np.arange(0,(20+4/10),4)
# yticks1_label=[str(int(yticks1[t])) for t in range(0,np.size(yticks1))]
# ax1.set_xticks(xticks1)
# ax1.xaxis.set_major_formatter(LatitudeFormatter())
# plt.xticks(fontproperties='Times New Roman')
# ax1.set_yticks(yticks1)
# labels_y=ax1.yaxis.get_ticklabels()
# [label.set_fontname('Times New Roman') for label in labels_y]
# ax1.set_yticklabels(yticks1_label)
# ax1.tick_params(axis="x",direction="out",length=4,labelsize=10)
# ax1.tick_params(axis="y",direction="out",length=4,labelsize=10)
# xminorlocator1=MultipleLocator(5)
# yminorlocator1=MultipleLocator(((4/4)))
# ax1.xaxis.set_minor_locator(xminorlocator1)
# ax1.yaxis.set_minor_locator(yminorlocator1)
# ax1.yaxis.set_minor_formatter(FormatStrFormatter(''))
# ax1.tick_params(axis="x",direction="out",which="minor",length=2)
# ax1.tick_params(axis="y",direction="out",which="minor",length=2)
# # twin y
# ax2=ax1.twiny()
# # cumulation
# ax2.plot(z_above_h_diff/(10**4),msl_fig,linewidth=0.6,color='#000000',linestyle='-',zorder=2)
# ax2.set_xlabel(str('Outlier (10$^{\mathregular{4}}$)'),fontdict=font_label1)
# # 刻度(主+次)
# ax2.set_xlim([0,8])
# xticks2=np.arange(0,8.1,2)
# ax2.set_xticks(xticks2)
# plt.xticks(fontproperties='Times New Roman',fontsize=10)
# ax2.tick_params(axis="x",direction="out",length=4)
# xminorlocator2=MultipleLocator((2/4))
# ax2.xaxis.set_minor_locator(xminorlocator2)
# ax2.tick_params(axis="x",direction="out",which="minor",length=2)
# # colorbar
# ax4=fig.add_axes([0.18,0.03,0.67,0.018])
# cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=c_norm,cmap=c_map),cax=ax4,aspect=60,format='%.0f',ticks=(z_count_level),orientation='horizontal',extend='both')
# cbar.set_ticks(z_count_level)
# labels_cbar=cbar.ax.xaxis.get_ticklabels()
# [label.set_fontname('Times New Roman') for label in labels_cbar]
# cbar_label=[str(int(z_count_level[i]/100)) for i in range(0,np.size(z_count_level))]
# cbar.ax.set_xticklabels(cbar_label)
# cbar.ax.tick_params(labelsize=7)
# cbar.ax.tick_params(bottom=False,top=False,left=False,right=False)
# cbar.ax.tick_params(length=1.5)
# ax3=fig.add_axes([0.85,-0.012,0.01,0.01])
# ax3.axis('off')
# ax3.text(0,0,'(10$^{\mathregular{2}}$)',family='Times New Roman',fontsize=7,color='black')
# plt.savefig(str(str('D:/DA/article/')+month+str('_zabove_diff.png')),dpi=600,bbox_inches='tight',pad_inches=0)


# nc_file=Dataset(str('D:/DA/lat_qc_z_lat_count.nc'),'w',format='NETCDF4')
# nc_file.createDimension('row',np.size(ih_median))
# nc_file.createDimension('col',np.size(lat_median))
# nc_file.createVariable('msl_level',np.float64,('row'))
# nc_file.variables['msl_level'][:]=msl_median
# nc_file.createVariable('ih_level',np.float64,('row'))
# nc_file.variables['ih_level'][:]=ih_median
# nc_file.createVariable('lat',np.float64,('col'))
# nc_file.variables['lat'][:]=lat_median
# nc_file.createVariable('z_above_count',np.float64,('row','col'))
# nc_file.variables['z_above_count'][:]=z_above_count_before
# nc_file.createVariable('z_below_count',np.float64,('row','col'))
# nc_file.variables['z_below_count'][:]=z_below_count_before
# nc_file.createVariable('z_above_h_count',np.float64,('row'))
# nc_file.variables['z_above_h_count'][:]=np.nansum(z_above_count_before,axis=1)
# nc_file.createVariable('z_below_h_count',np.float64,('row'))
# nc_file.variables['z_below_h_count'][:]=np.nansum(z_below_count_before,axis=1)
# nc_file.close()



""" # percentage
path_latlon=(r'D:/DA/latlon_qc_z_lat_count.nc')
latlon_z=Dataset(path_latlon,format='netCDF4')
z_above_latlon=np.array(latlon_z.variables['z_above_count'][:])
z_below_latlon=np.array(latlon_z.variables['z_below_count'][:])
# z_above_h_latlon=np.array(latlon_z.variables['z_above_h_count'][:])
# z_below_h_latlon=np.array(latlon_z.variables['z_below_h_count'][:])



lat_select_min=-15
lat_select_max=15
lat_select_count=np.array(np.array(z_above_count_before[:,np.where(np.logical_and(lat_median>=lat_select_min,lat_median<=lat_select_max))])[:,0,:])
lat_select_total_count=np.array(np.array(z_above_count_before[:,np.where(np.logical_and(lat_median>=lat_select_min,lat_median<=lat_select_max))])[:,0,:])+np.array(np.array(z_below_count_before[:,np.where(np.logical_and(lat_median>=lat_select_min,lat_median<=lat_select_max))])[:,0,:])
lat_select_per=lat_select_count/lat_select_total_count*(100)
latlon_select_count=np.array(np.array(z_above_latlon[:,np.where(np.logical_and(lat_median>=lat_select_min,lat_median<=lat_select_max))])[:,0,:])
latlon_select_total_count=np.array(np.array(z_above_latlon[:,np.where(np.logical_and(lat_median>=lat_select_min,lat_median<=lat_select_max))])[:,0,:])+np.array(np.array(z_below_latlon[:,np.where(np.logical_and(lat_median>=lat_select_min,lat_median<=lat_select_max))])[:,0,:])
latlon_select_per=latlon_select_count/latlon_select_total_count*(100)


fig=plt.figure(figsize=(1.2,1.9),dpi=600)
ax1=fig.add_subplot(1,1,1)
msl_fig=msl_median.copy()
msl_fig[0]=0
msl_fig[-1]=20
# smooth
lat_fig_index=11
msl_index=int(np.max(np.array(np.where(msl_fig<=3.5)[0])))
lat_select_per2=np.array(lat_select_per[:,int(lat_fig_index)]).copy()
lat_select_per2[0:msl_index]=lat_select_per2[0:msl_index]*(1.5)
latlon_select_per2=np.array(latlon_select_per[:,int(lat_fig_index)]).copy()
latlon_select_per2[0:msl_index]=latlon_select_per2[0:msl_index]*(1.5)
import scipy
lat_select_smooth=scipy.signal.savgol_filter(lat_select_per2,21,3)
latlon_select_smooth=scipy.signal.savgol_filter(latlon_select_per2,21,3)
# mesh
curve_index=np.array(lat_select_total_count[:,int(lat_fig_index)])/(10**4)
ax1.plot(lat_select_smooth[np.where(curve_index>=1/2*np.nanmax(curve_index))],msl_fig[np.where(curve_index>=1/2*np.nanmax(curve_index))],linestyle='--',linewidth=0.6,color='#000000',zorder=2)
ax1.plot(latlon_select_smooth[np.where(curve_index>=1/2*np.nanmax(curve_index))],msl_fig[np.where(curve_index>=1/2*np.nanmax(curve_index))],linestyle='-',linewidth=0.6,color='#000000',zorder=2)
# ax1.plot(lat_select_per[:,11],msl_fig,linestyle='--',linewidth=0.6,color='#000000',zorder=0)
# ax1.plot(latlon_select_per[:,11],msl_fig,linestyle='-',linewidth=0.6,color='#000000',zorder=0)
# ax1.plot(lat_select_count[:,1],msl_fig,linestyle='-',linewidth=0.6,color='#0000FF',zorder=0)
# ax1.plot(latlon_select_count[:,1],msl_fig,linestyle='--',linewidth=0.6,color='#0000FF',zorder=0)
# ax1.plot(lat_select_count[:,2],msl_fig,linestyle='-',linewidth=0.6,color='#FF0000',zorder=0)
# ax1.plot(latlon_select_count[:,2],msl_fig,linestyle='--',linewidth=0.6,color='#FF0000',zorder=0)
# ax1.plot(lat_select_count[:,3],msl_fig,linestyle='-',linewidth=0.6,color='#00FF00',zorder=0)
# ax1.plot(latlon_select_count[:,3],msl_fig,linestyle='--',linewidth=0.6,color='#00FF00',zorder=0)
# ax1.plot(lat_select_count[:,4],msl_fig,linestyle='-',linewidth=0.6,color='#F48B06',zorder=0)
# ax1.plot(latlon_select_count[:,4],msl_fig,linestyle='--',linewidth=0.6,color='#F48B06',zorder=0)
# ax1.plot(lat_select_count[:,5],msl_fig,linestyle='-',linewidth=0.6,color='#FA02FA',zorder=0)
# ax1.plot(latlon_select_count[:,5],msl_fig,linestyle='--',linewidth=0.6,color='#FA02FA',zorder=0)
font_label1={'fontsize':11.5,'color':'k','family':'Times New Roman','weight':'light'}
ax1.set_ylabel('Height (km)',fontdict=font_label1,labelpad=1)
ax1.set_xlabel('Outlier (%)',fontdict=font_label1,labelpad=1)
# 刻度(主+次)
ax1.set_xlim([0,5])
ax1.set_ylim([0,20])
xticks1=np.arange(0,(5+1/10),1)
yticks1=np.arange(0,(20+4/10),4)
# 刻度
yticks1_label=[str(int(yticks1[t])) for t in range(0,np.size(yticks1))]
ax1.set_xticks(xticks1)
# plt.xticks(fontproperties='Times New Roman',fontsize=10)
# ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
xticks1_label=[str(int(xticks1[t])) for t in range(0,np.size(xticks1))]
labels_x=ax1.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_x]
ax1.set_xticklabels(xticks1_label)
ax1.tick_params(axis="x",direction="out",length=4,labelsize=10)
xminorlocator=MultipleLocator(0.25)
ax1.xaxis.set_minor_locator(xminorlocator)
ax1.set_yticks(yticks1)
labels_y=ax1.yaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_y]
ax1.set_yticklabels(yticks1_label)
ax1.tick_params(axis="y",direction="out",length=4,labelsize=10)
yminorlocator1=MultipleLocator(((4/4)))
ax1.yaxis.set_minor_locator(yminorlocator1)
ax1.tick_params(axis="x",direction="out",which="minor",length=2)
ax1.tick_params(axis="y",direction="out",which="minor",length=2)
ax2=ax1.twiny()
ax2.fill_betweenx(msl_fig,np.zeros((np.size(msl_fig))),np.array(lat_select_total_count[:,int(lat_fig_index)])/(10**4),color='#9C9C9C',edgecolor='none',alpha=0.3,zorder=0)
# ax2.plot(np.array(lat_select_total_count[:,int(lat_fig_index)])/(10**4),msl_fig,linewidth=0.6,color='#989898',alpha=0.3,linestyle='-')
ax2.set_xlabel('Data Count       ',fontdict=font_label1)
ax2.set_xlim([0,18])
xticks2=np.arange(0,(18.1+6/10),6)
xticks2_label=['{:.0f}'.format((xticks2[t])) for t in range(0,np.size(xticks2))]
ax2.set_xticks(xticks2)
labels_x2=ax2.xaxis.get_ticklabels()
[label.set_fontname('Times New Roman') for label in labels_x2]
ax2.set_xticklabels(xticks2_label)
ax2.tick_params(axis="x",direction="out",length=4,labelsize=10)
xminorlocator2=MultipleLocator(((2)))
ax2.xaxis.set_minor_locator(xminorlocator2)
ax2.tick_params(axis="x",direction="out",which="minor",length=2)
ax3=fig.add_axes([0.73,1.036,0.01,0.01])
ax3.axis('off')
ax3.text(0,0,'(10$^{\mathregular{4}}$)',family='Times New Roman',fontsize=9,color='black')
plt.savefig(str('D:/DA/article/north/125N_15N.png'),dpi=600,bbox_inches='tight',pad_inches=0)
"""



# # data count
# z_thre=2.5
# z_thre_qc3=2.8
# ih_min=6
# ih_max=8
# z_above_count=np.zeros((np.size(lat_median),np.size(lon_median)))
# # z_above_count_local_qc2=np.zeros((np.size(lat_median),np.size(lon_median)))
# # z_above_count_qc3=np.zeros((np.size(lat_median),np.size(lon_median)))
# # z_above_count_local_qc3=np.zeros((np.size(lat_median),np.size(lon_median)))
# lat_ih_total=np.zeros((np.size(z_each_prf,0),np.size(z_each_prf,1)))
# lon_ih_total=np.zeros((np.size(z_each_prf,0),np.size(z_each_prf,1)))
# for i in range(0,np.size(z_each_prf,0)):
#  lat_ih_total[i,:]=np.ones((np.size(z_each_prf,1)))*lat[i]
#  lon_ih_total[i,:]=np.ones((np.size(z_each_prf,1)))*lon[i]
# for i in range(0,np.size(lat_median)):
#  for j in range(0,np.size(lon_median)):
#   ih_lat_lon=np.array(np.array(ih[np.where(np.logical_and(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]),np.logical_and(lon>=lon_level[j],lon<lon_level[j+1]))),:])[0,:,:])
#   z_lat_lon=np.array(np.array(z_each_prf[np.where(np.logical_and(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]),np.logical_and(lon>=lon_level[j],lon<lon_level[j+1]))),:])[0,:,:])
#   # z_qc2_local_lat_lon=np.array(np.array(ba_z_local[np.where(np.logical_and(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]),np.logical_and(lon>=lon_level[j],lon<lon_level[j+1]))),:])[0,:,:])
#   # z_qc3_lat_lon=np.array(np.array(z_use[np.where(np.logical_and(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]),np.logical_and(lon>=lon_level[j],lon<lon_level[j+1]))),:])[0,:,:])
#   # z_local_qc3_lat_lon=np.array(np.array(z_local[np.where(np.logical_and(np.logical_and(lat>=lat_level[i],lat<lat_level[i+1]),np.logical_and(lon>=lon_level[j],lon<lon_level[j+1]))),:])[0,:,:])
#   # ih scale
#   z_lat_lon=z_lat_lon[np.where(np.logical_and(ih_lat_lon>=ih_min,ih_lat_lon<ih_max))]
#   z_above_count[i,j]=np.size(np.array(z_lat_lon[np.where(abs(z_lat_lon)>z_thre)]))
#   # z_qc2_local_lat_lon=z_qc2_local_lat_lon[np.where(np.logical_and(ih_lat_lon>=ih_min,ih_lat_lon<ih_max))]
#   # z_above_count_local_qc2[i,j]=np.size(np.array(z_qc2_local_lat_lon[np.where(abs(z_qc2_local_lat_lon)>z_thre_qc3)]))
#   # z_qc3_lat_lon=z_qc3_lat_lon[np.where(np.logical_and(ih_lat_lon>=ih_min,ih_lat_lon<ih_max))]
#   # z_above_count_qc3[i,j]=np.size(np.array(z_qc3_lat_lon[np.where(abs(z_qc3_lat_lon)>z_thre_qc3)]))
#   # z_local_qc3_lat_lon=z_local_qc3_lat_lon[np.where(np.logical_and(ih_lat_lon>=ih_min,ih_lat_lon<ih_max))]
#   # z_above_count_local_qc3[i,j]=np.size(np.array(z_local_qc3_lat_lon[np.where(abs(z_local_qc3_lat_lon)>z_thre_qc3)]))


# # lat qc2
# # above_level_diff=np.linspace(50,1260,12,endpoint=True)
# # above_level_diff=np.linspace(50,710,7,endpoint=True)
# above_level_diff=np.linspace(100,1260,12,endpoint=True)
# # above_level_diff=np.linspace(100,940,8,endpoint=True)
# cyclic_zabove,cyclic_lon=add_cyclic_point(z_above_count,coord=lon_median)
# cyclic_lon,cyclic_lat=np.meshgrid(cyclic_lon,lat_median)
# fig=plt.figure(figsize=(6,4.5))
# ax1=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
# ax1.add_feature(cfeature.LAND,edgecolor='#000000',facecolor='never',alpha=1,lw=0.4,zorder=2) # 陆地
# ax1.set_extent([-180,180,-45,45],crs=ccrs.PlateCarree())
# xticks=np.arange(-180,(180+60/10),60)
# # yticks=np.concatenate([np.array([-35]).flatten(),np.array(np.arange(-20,(20+10/10),10)).flatten(),np.array([35]).flatten()])
# yticks=np.arange(-45,(45+15/10),15)
# # color
# col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08","#F48B06","#F42103","#BC012E","#FA02FA"]
# # col_list=["#EAE3C7","#C8E1B7","#AED7B7","#83CDAC","#58BEA6","#12B7A9","#00B7B6","#01B4CA","#2094c7","#697FBB","#8366A9","#92509A","#A93391"]
# # col_list=["#EAE3C7","#C8E1B7","#AED7B7","#83CDAC","#58BEA6","#12B7A9","#00B7B6"]
# # col_list=["#EAE3C7","#C8E1B7","#AED7B7","#83CDAC","#58BEA6","#12B7A9","#00B7B6","#01B4CA","#2094c7","#697FBB","#8366A9","#92509A"]
# # col_list=["#06F3FB","#07B7FA","#037DF9","#034BF0","#08B81F","#75D608","#B5E805","#F0F20D","#F8BB08"]
# cbar_map=mcolors.ListedColormap(col_list)
# cbar_norm=mpl.colors.BoundaryNorm(above_level_diff,cbar_map.N,extend='both')
# ax1.pcolormesh(cyclic_lon,cyclic_lat,cyclic_zabove,cmap=cbar_map,norm=cbar_norm,shading='nearest',zorder=0,transform=ccrs.PlateCarree())
# # quiver
# # arrow=mpatches.ArrowStyle('-|>',head_length=0.5,head_width=0.2)
# # ax1.streamplot(cyclic_lon_era,cyclic_lat_era,cyclic_u_hourly_mean,cyclic_v_hourly_mean,color='black',density=0.6,linewidth=0.5,arrowsize=0.6,arrowstyle=arrow,transform=ccrs.PlateCarree())
# # 刻度
# ax1.set_xticks(xticks)
# ax1.set_yticks(yticks)
# plt.xticks(fontproperties='Times New Roman',fontsize=10)
# plt.yticks(fontproperties='Times New Roman',fontsize=10)
# ax1.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
# ax1.yaxis.set_major_formatter(LatitudeFormatter())
# xminorlocator=MultipleLocator(60/3)
# yminorlocator=MultipleLocator(15/3)
# ax1.xaxis.set_minor_locator(xminorlocator)
# ax1.yaxis.set_minor_locator(yminorlocator)
# # polygon
# # cloud_coords=[(140.0,-10),(180.0,-10),(200.0,-20),(160.0,-20)]
# # cloud_region=Polygon(cloud_coords)
# # cloud_feature=ShapelyFeature([cloud_region],crs=ccrs.PlateCarree(),edgecolor='k',facecolor='none',linewidth=0.6,linestyle='--')
# # ax1.add_feature(cloud_feature)
# # clear_coords=[(-160.0,-10),(-120.0,-10),(-100.0,-20),(-140.0,-20)]
# # clear_region=Polygon(clear_coords)
# # clear_feature=ShapelyFeature([clear_region],crs=ccrs.PlateCarree(),edgecolor='k',facecolor='none',linewidth=0.6,linestyle='--')
# # ax1.add_feature(clear_feature)
# # text
# # ax1.text(171,-7.5,str('A'),color='black',fontsize=10,family='Times New Roman',transform=ccrs.PlateCarree())
# # ax1.text(-128,-7.5,str('B'),color='black',fontsize=10,family='Times New Roman',transform=ccrs.PlateCarree())
# # colorbar
# ax2=fig.add_axes([0.2,0.3,0.63,0.016])
# # ax2=fig.add_axes([0.3,0.3,0.43,0.016])
# cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm,cmap=cbar_map),cax=ax2,orientation='horizontal',aspect=32,format='%.0f',ticks=above_level_diff)
# cbar.set_ticks(above_level_diff)
# labels_cbar=cbar.ax.xaxis.get_ticklabels()
# [label.set_fontname('Times New Roman') for label in labels_cbar]
# cbar.ax.tick_params(labelsize=8)
# cbar.ax.tick_params(bottom=True,top=False,left=False,right=False)
# cbar.ax.tick_params(length=1.5)
# # plt.savefig(str('D:/DA/publish/MWR/fig/')+str('fig10a.png'),dpi=600,bbox_inches='tight',pad_inches=0)


