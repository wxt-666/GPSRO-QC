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








