import netCDF4 as nc
import numpy as np
import os
import csv
import glob
import math
import chardet
import scipy
import ropp_abel
import matplotlib as mpl
import scipy.integrate as integ
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from eofs.standard import Eof
from datetime import datetime
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,FixedLocator
from netCDF4 import Dataset
from scipy import interpolate
from geopy.distance import geodesic
from scipy.optimize import minimize

# get gpsro vars
class GPSRO:
 
 def __init__(self,path,var_h,var_fh,h_interpolate):
  # obs data path
  self.path=path
  # obs
  self.var_h=var_h
  self.var_fh=var_fh
  # interpolate levels
  self.h_interpolate=h_interpolate

 @staticmethod
 def is_valid(path):
  return os.path.exists(path)

 @staticmethod  
 def get_attrs(path):
  file_folders=os.listdir(path)
  attr_keys=[]
  attrs=[]
  attr_types=[]
  # keys
  for file_path in file_folders:
   try:
    nc_file=nc.Dataset(path+file_path,'r')
    attr_key=nc_file.ncattrs()
    attr_keys.append(attr_key)
   except Exception as e:
    print(f"error occurs when reading file: {e}")  
  # total keys  
  for i in range(0,len(attr_keys)):
   attr_keys_total=[]
   attr_keys_total=set(attr_keys_total)|set(attr_keys[i])
   attr_keys_total=list(attr_keys_total)
  # attrs
  for file_path in file_folders:
   attr=[]
   attr_type=[]
   try:
    nc_file=nc.Dataset(path+file_path,'r')
    for i in range(0,len(attr_keys_total)):
     if str(attr_keys_total[i]) in nc_file.ncattrs():
      attr.append(nc_file.getncattr(attr_keys_total[i]))
      type_single=str(type(nc_file.getncattr(attr_keys_total[i])))
      attr_type.append(type_single)
     else:
      attr.append(np.NaN)
      attr_type.append('none')
    attrs.append(attr)
    attr_types.append(attr_type)
   except Exception as e:
    print(f"error occurs when reading file: {e}")
  attr_types=([attr_types[i] for i in range(0,len(attr_types)) if 'none' not in attr_types[i]])[0]
  return attr_keys_total,attr_types,attrs
 
 @staticmethod
 def get_single_attr(path,attr_name):
  file_folders=os.listdir(path)
  attr=[]
  # keys
  for file_path in file_folders:
   try:
    nc_file=nc.Dataset(path+file_path,'r')
    attr_key=nc_file.ncattrs()
    if str(attr_name) in list(attr_key):
     attr.append(nc_file.getncattr(str(attr_name)))
    else:
     attr.append(np.NaN)
   except Exception as e:
    print(f"error occurs when reading file: {e}")  
  return attr

 @staticmethod
 def get_variables(path,byte_order='littleendian'):
  file_folders=os.listdir(path)
  var_keys=[]
  vars=[]
  # var keys
  for file_path in file_folders:
   try:
    nc_file=nc.Dataset(path+file_path,'r')
    var_key=nc_file.variables.keys()
    var_keys.append(list(var_key))
   except Exception as e:
    print(f"error occurs when reading file: {e}")
  # total var keys  
  for i in range(0,len(var_keys)):
   var_keys_total=[]
   var_keys_total=set(var_keys_total)|set(var_keys[i])
   var_keys_total=list(var_keys_total)
  for file_path in file_folders:   
   var=[]
   try:
    nc_file=nc.Dataset(path+file_path,'r')
    for i in range(0,len(var_keys_total)):
     if var_keys_total[i] in list(nc_file.variables.keys()):
      var.append(np.array(nc_file.variables[str(var_keys_total[i])][:]))
      if byte_order.lower()=='bigendian':
       var.append((nc_file.variables[str(list(var_key)[i])][:]).byteswap().newbyteorder())
     else:
      var.append(np.array([-999.0]))
    vars.append(var)
   except Exception as e:
    print(f"error occurs when reading file: {e}")
  return var_keys_total,vars  

 @staticmethod
 def get_single_variable(path,var_name,byte_order='littleendian'):
  file_folders=os.listdir(path)
  var=[]
  order_index=0
  # var keys
  for file_path in file_folders:
   try:
    nc_file=nc.Dataset(path+file_path,'r')
    var_key=nc_file.variables.keys()
    if var_name in list(var_key):
     var.append(np.array(nc_file.variables[str(var_name)][:]))
     if byte_order.lower()=='bigendian':
       var.append((nc_file.variables[str(var_name)][:]).byteswap().newbyteorder())
    else:
     var.append(np.array([-999.0]))
   except Exception as e:
    print(f"error occurs when reading file: {e}")
  try:
   max_var=np.max(np.array([np.size(np.array(var[i])) for i in range(0,len(var))]))
   var_array=np.ones((int(len(var)),int(max_var)))*(-999.0)
   for i in range(0,len(var)):
    var_array[i,0:int(np.size(np.array(var[i])))]=np.array(var[i])
  except Exception as e:
   print(f"error occurs when forming array: {e}")  
  return var_array

 @staticmethod
 def profile_merge(path):
  # all vars
  attr_keys_total,attr_types,attrs=GPSRO.get_attrs(path)
  var_keys_total,vars=GPSRO.get_variables(path,byte_order='littleendian')
  # merge profiles
  # attrs
  attr_all_profile=np.array(attrs,dtype='object')
  # variables
  var_number=np.array([[np.size(np.array((vars[i])[j])) for j in range(0,len(vars[i]))] for i in range(0,len(vars))],dtype='int')
  var_all_profile=np.ones((len(var_keys_total),len(vars),np.max(var_number)))*(-999)
  for i in range(0,len(var_keys_total)):
   for j in range(0,len(vars)):
    var_all_profile[i,j,0:(var_number[j,i])]=np.array(np.array(vars[j],dtype='object')[i],dtype='object')
  return var_number,attr_keys_total,attr_types,attr_all_profile,var_keys_total,var_all_profile

 @staticmethod
 def profile_interpolate(var_h,h,fh,h_interpolate):
  # data
  fh_interpolate=np.ones((np.size(h,0),np.size(h_interpolate)))*(-999.0)
  empty_index=[]
  for i in range(0,np.size(h,0)):
   h_single=h[i,:]
   fh_single=fh[i,:]
   # null
   index=np.where(np.logical_and(np.logical_and(h_single!=-999,h_single>0),fh_single!=-999))
   h_single=h_single[index]
   fh_single=fh_single[index]
   # NaN
   index_nan=np.where(np.logical_and(~np.isnan(h_single),~np.isnan(fh_single)))
   h_single=h_single[index_nan]
   fh_single=fh_single[index_nan]
   # delete duplicate
   h_du=[h for h in list(h_single) if list(h_single).count(h)>1]
   if len(h_du)!=0:
    index_h_du=[]
    for i in range(0,len(h_du)):
     index_h_du.append([j for j in range(0,np.size(np.array(h_single))) if h_single[j]==h_du[i]])
    index_delete=[]
    for i in range(0,len(index_h_du)):
     index_delete.append(list(list(index_h_du[i])[1:]))
    index_delete_total=[]
    for i in range(0,len(index_delete)):
     index_delete_total=index_delete_total+index_delete[i]
    index_delete_total=np.array(index_delete_total)
    h_single=np.array([h_single[i] for i in range(0,np.size(np.array(h_single))) if i not in index_delete_total])
    fh_single=np.array([fh_single[i] for i in range(0,np.size(np.array(fh_single))) if i not in index_delete_total])
   if np.size(h_single)<=2 or np.size(fh_single)<=2:
    empty_index.append(i)
    fh_interpolate[i,:]=np.array([np.NaN for k in range(0,np.size(h_interpolate))])
   else:
    h_interpolate_single=h_interpolate[np.where(np.logical_and(h_interpolate>=np.nanmin(h_single),h_interpolate<=np.nanmax(h_single)))]
    if np.size(h_interpolate_single)<=2:
     empty_index.append(i)
     fh_interpolate[i,:]=np.array([np.NaN for k in range(0,np.size(h_interpolate))])
    else:
     min_index=int(np.nanmin(np.array([int(np.where(h_interpolate==np.nanmin(h_interpolate_single))[0]),int(np.where(h_interpolate==np.nanmax(h_interpolate_single))[0])])))
     max_index=int(np.nanmax(np.array([int(np.where(h_interpolate==np.nanmin(h_interpolate_single))[0]),int(np.where(h_interpolate==np.nanmax(h_interpolate_single))[0])]))+1)
     if var_h=='Pres':
      inter_f=interpolate.interp1d(np.array(np.log(h_single)),np.array(fh_single),kind='linear',fill_value='extrapolate')
      fh_interpolate[i,min_index:max_index]=inter_f(np.log(h_interpolate_single))
     else:
      inter_f=interpolate.interp1d(np.array(h_single),np.array(fh_single),kind='linear',fill_value='extrapolate')
      fh_interpolate[i,min_index:max_index]=inter_f(h_interpolate_single)
  fh_interpolate[fh_interpolate==-999.0]=np.NaN
  return fh_interpolate,empty_index

 # get interpolated array
 def get_file_array(self):
  _,_,_,_,var_keys_total,var_all_profile=GPSRO.profile_merge(self.path)
  h=var_all_profile[int(var_keys_total.index(str(self.var_h))),:,:]
  fh=var_all_profile[int(var_keys_total.index(str(self.var_fh))),:,:]
  fh_interpolate,_=GPSRO.profile_interpolate(self.var_h,h,fh,self.h_interpolate)
  return fh_interpolate

 
# quality control
class QC(GPSRO):

 def __init__(self,path,var_ref,var_h,var_fh,h_interpolate,add_path,var_ref_model,var_h_model,var_fh_model,h_min,h_max,z_standard_qc2,z_standard_qc3):
  # obs data path
  super().__init__(path,var_h,var_fh,h_interpolate)
  # model data path
  self.add_path=add_path
  # model
  self.var_ref_model=var_ref_model
  self.var_h_model=var_h_model
  self.var_fh_model=var_fh_model
  # qc1 ref/hmin/hmax
  self.var_ref=var_ref
  self.h_min=h_min
  self.h_max=h_max
  # z-score range
  self.z_standard_qc2=z_standard_qc2
  self.z_standard_qc3=z_standard_qc3

 @staticmethod 
 def qc1(ref,**kwds):
  # Range Check
  h_check=kwds.pop('h_check',False)
  h=kwds.pop('h',np.zeros((1,1)))
  h_min=kwds.pop('h_min',50)
  h_max=kwds.pop('h_max',800)
  try:
   index_keep_ref=[]
   for i in range(0,np.size(ref,0)):
    ref_single=ref[i,:]
    ref_single[ref_single==-999.0]=np.NaN
    ref_single=ref_single[~np.isnan(ref_single)]
    if np.size(ref_single)!=0:
     if np.min(ref_single)>=0.0:
      index_keep_ref.append(i)
  except Exception as e:
   print(f"error occurs when ref check: {e}")
  index_keep=index_keep_ref       
  # h range check
  if h_check==True:
   h_after_ref=np.zeros((len(index_keep_ref),np.size(h,1)))
   for i in range(0,len(index_keep_ref)):
    h_after_ref[i,:]=h[int(index_keep_ref[i]),:]
   try:
    index_keep=[] 
    for i in range(0,np.size(h_after_ref,0)):
     h_single=h_after_ref[i,:]
     h_single[np.logical_or(h_single==-999.0,h_single<0.0)]=np.NaN
     h_single=h_single[~np.isnan(h_single)] 
     if np.size(h_single!=0): 
      if np.min(h_single)<=h_min and np.max(h_single)>=h_max:
       index_keep.append(int(index_keep_ref[i]))
   except Exception as e:
    print(f"error occurs when h range check: {e}")
  index_delete=[i for i in range(0,np.size(ref,0)) if i not in index_keep] 
  return index_keep,index_delete

 @staticmethod
 def qc2(fh,h_interpolate,z_standard_qc2):
  # biweight check
  try:
   m=np.array([np.nanmedian(fh[:,i]) for i in range(0,np.size(h_interpolate))])
   mad=np.array([np.nanmedian(abs(fh[:,i]-m[i])) for i in range(0,np.size(h_interpolate))])
   # weight
   w=np.zeros((np.size(np.array(fh),0),np.size(h_interpolate)))
   for i in range(0,np.size(h_interpolate)):
    w[:,i]=(fh[:,i]-m[i])/(7.5*mad[i])
   w[abs(w)>1]=1
   # biweight mean/std
   w_cal1=np.zeros((np.size(np.array(fh),0),np.size(h_interpolate)))
   for i in range(0,np.size(h_interpolate)):
    w_cal1[:,i]=(fh[:,i]-m[i])*((1-(w[:,i])**2)**2)
   w_cal2=((1-w)**2)**2
   w_cal3=w_cal1**2
   w_cal4=w_cal2*(1-5*(w**2))
   # mean
   bm=m+(np.nansum(w_cal1,axis=0)/np.nansum(w_cal2,axis=0))
   # std
   bsd=np.sqrt(np.size(np.array(fh),0)*np.nansum(w_cal3,axis=0))/abs(np.nansum(w_cal4,axis=0))
   # Z-score
   z_score=np.zeros((np.size(np.array(fh),0),np.size(h_interpolate)))
   for i in range(0,np.size(h_interpolate)):  
    z_score[:,i]=abs(fh[:,i]-bm[i])/bsd[i]
  except Exception as e:
   print(f"error occurs when calculating Z-score: {e}")
  fh_qc2=fh.copy()
  fh_qc2[abs(z_score)>z_standard_qc2]=np.NaN
  return z_score,fh_qc2,bm,bsd

 @staticmethod
 def qc3(fh_qc2,fh_model,h_interpolate,z_standard_qc3):
  # model biweight check
  fh_diff=(fh_qc2-fh_model)/fh_model
  # qc3
  try:
   m=np.array([np.nanmedian(fh_diff[:,i]) for i in range(0,np.size(h_interpolate))])
   mad=np.array([np.nanmedian(abs(fh_diff[:,i]-m[i])) for i in range(0,np.size(h_interpolate))])
   # weight
   w=np.zeros((np.size(np.array(fh_qc2),0),np.size(h_interpolate)))
   for i in range(0,np.size(h_interpolate)):
    w[:,i]=(fh_diff[:,i]-m[i])/(7.5*mad[i])
   w[abs(w)>1]=1
   w_cal1=np.zeros((np.size(np.array(fh_qc2),0),np.size(h_interpolate)))
   for i in range(0,np.size(h_interpolate)):
    w_cal1[:,i]=(fh_diff[:,i]-m[i])*((1-(w[:,i])**2)**2)
    w_cal2=((1-w)**2)**2
    w_cal3=w_cal1**2
    w_cal4=w_cal2*(1-5*(w**2)) 
   # mean
   bm=m+(np.nansum(w_cal1,axis=0)/np.nansum(w_cal2,axis=0))
   # std
   bsd=np.sqrt(np.size(np.array(fh_qc2),0)*np.nansum(w_cal3,axis=0))/abs(np.nansum(w_cal4,axis=0))
   # Z-score
   z_score=np.zeros((np.size(np.array(fh_qc2),0),np.size(h_interpolate)))
   for i in range(0,np.size(h_interpolate)):  
    z_score[:,i]=abs(fh_diff[:,i]-bm[i])/bsd[i]
  except Exception as e:
   print(f"error occurs when calculating Z-score: {e}") 
  fh_qc3=fh_qc2.copy()
  fh_qc3[abs(z_score)>z_standard_qc3]=np.NaN    
  return z_score,fh_qc3
 
 # all bwqc process
 def BW_qc(self,step,**kwds):
  # Range Check
  h_check=kwds.pop('h_check',False)
  h=kwds.pop('h',np.zeros((1,1)))
  h_min=kwds.pop('h_min',50)
  h_max=kwds.pop('h_max',800)
  rfict=kwds.pop('rfict','rfict')
  rgeo=kwds.pop('rgeo','rgeoid')
  ref=GPSRO.get_single_variable(self.path,self.var_ref,byte_order='littleendian')
  h=GPSRO.get_single_variable(self.path,self.var_h,byte_order='littleendian')
  fh=GPSRO.get_single_variable(self.path,self.var_fh,byte_order='littleendian')
  if step==1:
   index_keep,index_delete=QC.qc1(ref,h_check=h_check,h=h,h_max=h_max,h_min=h_min)
   return index_keep,index_delete
  if step==2:
   index_keep,_=QC.qc1(ref,h_check=h_check,h=h,h_max=h_max,h_min=h_min)
   fh_interpolate,_=GPSRO.profile_interpolate(self.var_h,h,fh,self.h_interpolate)
   fh_qc1=np.zeros((np.size(np.array(index_keep)),np.size(self.h_interpolate)))
   for i in range(0,(np.size(np.array(index_keep)))):
    fh_qc1[i,:]=fh_interpolate[int(index_keep[i]),:]
   z_score,fh_qc2,bm,bsd=QC.qc2(fh_qc1,self.h_interpolate,self.z_standard_qc2)
   return index_keep,fh_interpolate,z_score,fh_qc2,bm,bsd
  if step==3:
   index_keep,_=QC.qc1(ref,h_check=h_check,h=h,h_max=h_max,h_min=h_min)
   fh_interpolate,_=GPSRO.profile_interpolate(self.var_h,h,fh,self.h_interpolate)
   fh_qc1=np.zeros((np.size(np.array(index_keep)),np.size(self.h_interpolate)))
   for i in range(0,(np.size(np.array(index_keep)))):
    fh_qc1[i,:]=fh_interpolate[int(index_keep[i]),:]
   z_score_qc2,fh_qc2,_,_=QC.qc2(fh_qc1,self.h_interpolate,self.z_standard_qc2)
   # check obs and model matched
   h_model=GPSRO.get_single_variable(self.add_path,self.var_h_model,byte_order='littleendian')
   fh_model=GPSRO.get_single_variable(self.add_path,self.var_fh_model,byte_order='littleendian')
   obs_list=os.listdir(self.path)
   obs_list=[str(str(obs_list[i])[7:]) for i in range(0,len(obs_list))]
   model_list=os.listdir(self.add_path)
   model_list=[str(str(model_list[i])[7:]) for i in range(0,len(model_list))]
   if len(model_list)>=len(obs_list):
    index_match=[i for i in range(0,len(obs_list)) if obs_list[i] in model_list]
   else:
    index_match=[i for i in range(0,len(model_list)) if model_list[i] in obs_list]
   # ensure h_model and h having the same dimension
   h_model2=np.ones((len(obs_list),np.size(h_model,1)))*(-999.0)
   fh_model2=np.ones((len(obs_list),np.size(fh_model,1)))*(-999.0)
   for i in range(0,np.size(obs_list)):
    if i in index_match:
     h_model2[i,:]=h_model[i,:]
     fh_model2[i,:]=fh_model[i,:]
    else:
     h_model2[i,:]=h_model2[i,:]
     fh_model2[i,:]=fh_model2[i,:]
   h_model=h_model2
   fh_model=fh_model2
   h_model_qc1=np.zeros((np.size(np.array(index_keep)),np.size(h_model,1)))   
   fh_model_qc1=np.zeros((np.size(np.array(index_keep)),np.size(fh_model,1)))
   for i in range(0,(np.size(np.array(index_keep)))):
    h_model_qc1[i,:]=h_model[int(index_keep[i]),:]
    fh_model_qc1[i,:]=fh_model[int(index_keep[i]),:]
   # msl to ih
   if np.min(abs(h_model))<=1:
    ref_model=GPSRO.get_single_variable(self.add_path,self.var_ref_model,byte_order='littleendian')
    ref_model_qc1=np.zeros((np.size(np.array(index_keep)),np.size(ref_model,1)))
    for i in range(0,(np.size(np.array(index_keep)))):
     ref_model_qc1[i,:]=ref_model[int(index_keep[i]),:]
    ref_model_qc1[ref_model_qc1==-999.0]=np.NaN
    ref_para_qc1=ref_model_qc1/(10**6)+1
    rfict=GPSRO.get_single_attr(self.path,rfict)
    rfict_qc1=[rfict[i] for i in range(0,len(rfict)) if i in index_keep]
    rfict_qc1=np.array(rfict_qc1)
    rgeo=GPSRO.get_single_attr(self.path,rgeo)
    rgeo_qc1=[rgeo[i] for i in range(0,len(rgeo)) if i in index_keep]
    rgeo_qc1=np.array(rgeo_qc1)
    h_model_qc1[h_model_qc1==-999.0]=np.NaN
    ih_model_qc1=np.zeros((np.size(h_model_qc1,0),np.size(h_model_qc1,1)))
    for i in range(0,np.size(h_model_qc1,1)):
     ih_model_qc1[:,i]=((h_model_qc1[:,i]+rfict_qc1+rgeo_qc1)*ref_para_qc1[:,i])-rfict_qc1-rgeo_qc1
    ih_model_qc1=np.nan_to_num(ih_model_qc1,nan=-999.0)
    h_model_qc1=np.nan_to_num(h_model_qc1,nan=-999.0)
   fh_model_interpolate,_=GPSRO.profile_interpolate(self.var_h,h_model_qc1,fh_model_qc1,self.h_interpolate)
   z_score_qc3,fh_qc3=QC.qc3(fh_qc2,fh_model_interpolate,self.h_interpolate,self.z_standard_qc3)   
   return index_keep,fh_interpolate,z_score_qc2,fh_qc2,fh_model_interpolate,z_score_qc3,fh_qc3


# calculate
class Gps_tools(GPSRO):
 
 def __init__(self,path,var_h,var_fh,h_interpolate,add_path,var_lat,var_lon,var_attr1,var_attr2):
  # obs data path
  super().__init__(path,var_h,var_fh,h_interpolate)
  self.add_path=add_path
  self.var_lat=var_lat
  self.var_lon=var_lon
  self.var_attr1=var_attr1
  self.var_attr2=var_attr2
 
 @staticmethod
 def time_convert(path):
  file_name=os.listdir(path)
  time_fmt='%Y-%m-%d'
  time=[]
  for i in range(0,len(file_name)):
   time.append(str(str(file_name[i])[12:16])+str(str(file_name[i])[17:20]))
  time_date=[]
  for i in range(0,len(time)):
   time_single=datetime.strptime(time[i],'%Y%j').date()
   time_date.append(time_single.strftime(time_fmt))
  return time_date
 
 @staticmethod
 def get_rs_txt(path,icol,ihead,dtype,sep=None):
  file_folders=os.listdir(path)
  rs=[]
  for i in range(0,len(file_folders)):
   try:
    rs_single=np.genfromtxt(str(path)+str(file_folders[i]),dtype=dtype,delimiter=sep,skip_header=ihead,usecols=icol-1)
    rs.append(rs_single)
   except FileNotFoundError:
    print('can not open file: {}'.format(str(path)+str(file_folders[i])))
   except LookupError:
    print('unknown code')
   except UnicodeDecodeError:
    print('decode error')
   except Exception as e:
    print(f"other error occurs: {e}")
  number=np.array([np.size(np.array(rs[i]),0) for i in range(0,len(rs))])
  rs_data=np.ones((len(file_folders),np.max(number),np.size(np.array(icol))))*(-999)
  for i in range(0,len(file_folders)):
    rs_data[i,0:int(number[i]),:]=rs[i]
  return rs_data
 
 @staticmethod
 def csv_reader(path):
  try:
   with open(path) as f:
    reader=csv.reader(f)
    header=next(reader)
    csv_file=[row for row in reader]
  except csv.Error as e:
   print(f"error occurs when reading file at line %s:%s" %(reader.line.num,e))
  return csv_file
 
 @staticmethod
 def dist_latlon(lat1,lon1,lat2,lon2):
  radius=6371
  dlat=np.radians(lat2-lat1)
  dlon=np.radians(lon2-lon1)
  a=np.sin(dlat/2.)*np.sin(dlat/2.)+np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2.)*np.sin(dlon/2.)
  c=2.*np.arctan2(np.sqrt(a),np.sqrt(1.-a))
  return radius*c

 @staticmethod
 def ref_reverse(temp,t_unit,rh,pres):
  try:
    # temp
    # convert ℃ to K
    if t_unit=='Degree':
     temp=temp+273.16
    if t_unit=='K':
     temp=temp
    # es
    # GOff-Gratch
    # e_s=np.array([10**(10.79574*(1-273.16/temp[i])-5.02800*np.log10(temp[i]/273.16)+1.50475*(10**(-4))*(1-10**(-1*8.2969*((temp[i]/273.16)-1)))+0.42873*(10**(-3))*(10**(4.76955*(1-(273.16/temp[i])))-1)+0.78614) for i in range(0,np.size(temp))])
    # Magnus
    e_s=6.1078*np.exp(((17.13*(temp-273.16))/(temp-38)))
    # e
    e=e_s*rh/100 
    # Ref
    ref=77.6*(pres/temp)+3.73*(10**5)*(e/(temp)**2)
  except Exception as e:
   print(f"error occurs when calculating Reflectivity: {e}")   
  return ref

 @staticmethod
 def ref_to_ba(ip,ref,scale):
  ba=ropp_abel.abel_trans.ropp_pp_abel_lin(ip,ref,ip,scale)
  return ba

 @staticmethod
 def ba_to_ref(ip,ba,scale):
  ref=ropp_abel.abel_trans.ropp_pp_invert_lin(ip,ba,ip,scale)
  return ref




