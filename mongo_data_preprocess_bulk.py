import os
import numpy as np
from pymongo import MongoClient   
import pandas as pd
from datetime import time, datetime
from dateutil.relativedelta import relativedelta
import scipy,re
from scipy.spatial.distance import euclidean,cdist,pdist
from sklearn.cluster import KMeans
import ckwrap
# from config import geolocator
from geopy.geocoders import Nominatim

def address_convertor(geolocator, Lat_Long):
    success = False
    while not success:
        try:
            location = geolocator.reverse(Lat_Long)
            address = location.raw['address']
            success = True
            return address
        except:
            geolocator = Nominatim(user_agent="galaxycard_wanda_exception"+str(random.randint(1,10000)))
            print('Open Street Map API Exception Occured!')
            continue
        finally:
            print('Address Found!!')

def mongo_preprocess(user_to_underwrite,cursor_users,cursor_devices):
    cursor = []
    cursor.extend(cursor_users)
    cursor.extend(cursor_devices)
    
    df = pd.DataFrame(cursor)
    df.drop_duplicates(inplace=True)
    #Battery
    if ('battery' in df.columns) and ('battery-level' in df.columns):
        df.battery.fillna(df['battery-level'],inplace=True)
        df.drop('battery-level',axis=1,inplace=True)
    else:
        df.rename(columns={'battery-level':'battery'},inplace=True)
    #Pin or fingerprint set
    if ('pin-or-fingerprint-set' in df.columns) and ('Pinorfingerprintset' in df.columns) and ('Pin Or Fingerprint Set' in df.columns):
        df['pin_or_finger_print'] = df['pin_or_finger_print'].fillna(df['pin-or-fingerprint-set'])
        df['pin_or_finger_print'] = df['pin_or_finger_print'].fillna(df['Pinorfingerprintset'])
        df['pin_or_finger_print'] = df['pin_or_finger_print'].fillna(df['Pin Or Fingerprint Set'])
        df.drop(['pin-or-fingerprint-set','Pinorfingerprintset','Pin Or Fingerprint Set'], axis=1, inplace=True)
    else:
        df.rename(columns={'pin-or-fingerprint-set' : 'pin_or_fingerprint_set',
                           'Pinorfingerprintset' : 'pin_or_fingerprint_set',
                           'Pin Or Fingerprint Set' : 'pin_or_fingerprint_set'}, inplace=True)
    #Memory Used
    if ('memory-used' in df.columns) and ('Memoryused' in df.columns) and ('Memory Used' in df.columns):
        df["memory_used"] = df["memory_used"].fillna(df['memory-used'])
        df['memory_used'] = df['memory_used'].fillna(df['Memoryused'])
        df['memory_used'] = df['memory_used'].fillna(df['Memory Used'])
        df.drop(['memory-used','Memoryused','Memory Used'], axis=1, inplace=True)
    else:
        df.rename(columns={"memory-used" : "memory_used",
                           "Memoryused" : "memory_used",
                           "Memory used" : "memory_used"}, inplace=True)

    #Memory total
    if ('memory-total' in df.columns) and ('Memorytotal' in df.columns) and ('Memory Total' in df.columns):
        df["memory_total"] = df["memory_total"].fillna(df['memory-total'])
        df['memory_total'] = df['memory_total'].fillna(df['Memorytotal'])
        df['memory_total'] = df['memory_total'].fillna(df['Memory Total'])
        df.drop(['memory-total','Memorytotal','Memory total'], axis=1, inplace=True)
    else:
        df.rename(columns={"memory-total" : "memory_total",
                           "Memorytotal" : "memory_total",
                           "Memory total" : "memory_total"}, inplace=True)

    # Disk Total
    if ('disk-total' in df.columns) and ('Disktotal' in df.columns) and ('Disk Total' in df.columns):
        df["disk_total"] = df["disk_total"].fillna(df['disk-total'])
        df['disk_total'] = df['disk_total'].fillna(df['Disktotal'])
        df['disk_total'] = df['disk_total'].fillna(df['Disk Total'])
        df.drop(['disk-total','Disktotal','Disk Total'], axis=1, inplace=True)
    else:
        df.rename(columns={"disk-total" : "disk_total",
                           "Disktotal" : "disk_total",
                           "Disk Total" : "disk_total"}, inplace=True)

    # Disk available
    if ('disk-available' in df.columns) and ('Diskavailable' in df.columns) and ('Disk Available' in df.columns):
        df["disk_available"] = df["disk_available"].fillna(df['disk-available'])
        df['disk_available'] = df['disk_available'].fillna(df['Diskavailable'])
        df['disk_available'] = df['disk_available'].fillna(df['Disk Available'])
        df.drop(['disk-available','Diskavailable','Disk Available'], axis=1, inplace=True)
    else:
        df.rename(columns={"disk-available" : "disk_available",
                           "Diskavailable" : "disk_available",
                           "Disk Available" : "disk_available"}, inplace=True)
    
    # ['_id', 'user_id', 'url', 'device', 'build', 'os-version', 'brand',
    #    'model', 'platform', 'preferred-locale', 'battery-level',
    #    'battery-state', 'headphones-connected', 'airplane-mode',
    #    'pin-or-fingerprint-set', 'location-enabled', 'install-referrer',
    #    'font-scale', 'carrier', 'disk-total', 'disk-available', 'memory-total',
    #    'memory-used', 'createdAt', 'updatedAt', '__v', 'latitude', 'longitude',
    #    'accuracy', 'altitude', 'heading', 'speed', 'altitudeaccuracy']


    df.rename(columns={ 'id':'_id',
                        'os_version':'os-version',
                        'preferred_locale':'preferred-locale', 
                        'updatedat':'updatedAt'},inplace=True)
    
    needed_columns = ['_id', 'user_id', 'platform', 'build', 'brand', 'battery',
                        'carrier', 'updatedAt', 'os-version', 'preferred-locale',
                        'device', 'latitude', 'longitude', 'url', 'pin_or_fingerprint_set', 'memory_pct', 'disk_pct']
    needed_columns = [i for i in needed_columns if i in df.columns]
    
    df = df[needed_columns]
    df.replace('',np.nan,inplace=True)
    
    print('mongo_preprocess data sliced!')

    ## Splitting into Time Periods
    df.updatedAt = pd.to_datetime(df.updatedAt, format = 'mixed')
    def timeperiod_splitter(x): # 10PM-7AM|7AM-12|12-5PM|5PM-10PM
        if x.time()<time(7,0,0):
            return "Night"
        elif x.time()<time(12,0,0):
            return "Morning"
        elif x.time()<time(17,0,0):
            return "Afternoon"
        elif x.time()<time(22,0,0):
            return "Evening"
        else:
            return "Night"
    df['Day_Period'] = df.updatedAt.apply(timeperiod_splitter)
    
    print('mongo_preprocess time period split!')
    
    # URL
    df.url = df.url.apply(lambda x:"/".join(str(x).split('/')[2:]))
    df.url.replace('',np.nan,inplace=True)
    df.url = df.url.str.replace('?','/').apply(lambda x:str(x).split('/')[0])
    df.url = df.url.replace(['_metrics', 'shop', 'document', 'sito', 'cms', 'site', 'wp2', 'media',
                        'test', 'wp1', 'news', '2019', 'wp', 'website', 'wordpress', 'web',
                        'blog', 'xmlrpc.php', 'wp-includes', '2018', 'aws.yml', 'users',
                        'phpunit', 'phpinfo', 'credentials', 'default', 'main.installer.php',
                        'docum=', 'config', 'sync', '=', 'favicon.ico', 'HEAD', 'nan'],np.nan)

    df.url.fillna('user',inplace=True)
    df.url.replace('user','home',inplace=True)
    
    # Build
    m = df.groupby('device')['build'].median()
    df['Phone_Build_Median'] = df.device.map(m)
    m = df.groupby('user_id')['Phone_Build_Median'].median()
    df['Phone_Build_Median'] = df.user_id.map(m)
    df.drop('build',axis=1,inplace=True)
    
    print("Done till build")
    
    if 'latitude' in df.columns:    
        #Address Features
        df.latitude = df.latitude.astype(float).transform(lambda x:np.round(x,4))
        df.longitude = df.longitude.astype(float).transform(lambda x:np.round(x,4))
        frequent_lat_long = df[(~df.latitude.isnull()) & (df.user_id == user_to_underwrite)].apply(lambda x:(x['latitude'],x['longitude']),axis=1)
        if frequent_lat_long.shape[0]>0:
            frequent_lat_long = frequent_lat_long.mode()[0]
            print('Frequent_Lat_Long', frequent_lat_long)
            address = address_convertor(geolocator, frequent_lat_long)
            print('Address!!!!:\t', address)

            keys_to_extract = ['county','suburb','city','town','village','city_district','neighbourhood']
            for i in keys_to_extract:
                df['address_'+i+'_specific'] = 1 if i in address else 0
        
        ## Lat,Long -- Userid Mode when rounded off to 3 decimals -- approximation to 111m
        df.latitude = df.latitude.astype(float).transform(lambda x:np.round(x,3))
        df.longitude = df.longitude.astype(float).transform(lambda x:np.round(x,3))
        m = df[~(df.device.isnull() | df.latitude.isnull())].groupby('device')['latitude'].agg(lambda x: pd.Series.mode(x)[0])
        n = df[~(df.device.isnull() | df.longitude.isnull())].groupby('device')['longitude'].agg(lambda x: pd.Series.mode(x)[0])
        df.latitude.fillna(df.device.map(m),inplace=True)
        df.longitude.fillna(df.device.map(n),inplace=True)
        
        ## Aggregating Lat Long!
        df = df[~(df['latitude'].isnull() | df['longitude'].isnull())]
        df['lat_long'] = df.apply(lambda x:(x['latitude'],x['longitude']),axis=1)
        
        ## User Level Lat Long
        m = df.groupby('user_id')['lat_long'].unique().reset_index()
        m.columns = ['user_id','user_lat_longs']
        df = pd.merge(df,m)
        df.user_lat_longs = df.user_lat_longs.apply(lambda x: [np.asarray(i) for i in x])
        
        ## Device Level Lat Long
        m = df.groupby('device')['lat_long'].unique().reset_index()
        m.columns = ['device','device_lat_longs']
        df = pd.merge(df,m)
        df.device_lat_longs = df.device_lat_longs.apply(lambda x: [np.asarray(i) for i in x])
        
        # Locations Visited
        df['user_different_locations_visited'] = df.user_lat_longs.apply(lambda x:len(x))
        df['device_different_locations_visited'] = df.device_lat_longs.apply(lambda x:len(x))

        # Distance Calculations:
        df.user_lat_longs = df.user_lat_longs.apply(lambda x:np.array(x))
        df['user_distances'] = df.user_lat_longs.apply(lambda x:pdist(x))
        df.user_distances = df.user_distances.apply(lambda x:[0] if len(x)==0 else x)

        df.device_lat_longs = df.device_lat_longs.apply(lambda x:np.array(x))
        df['device_distances'] = df.device_lat_longs.apply(lambda x:pdist(x))
        df.device_distances = df.device_distances.apply(lambda x:[0] if len(x)==0 else x)

        del df['device_lat_longs']

        df['User_Max_Movement'] = df.user_distances.apply(lambda x:max(x))
        df['User_Average_Movement'] = df.user_distances.apply(lambda x:np.mean(x))
        df['User_Total_Movement'] = df.user_distances.apply(lambda x:sum(x))

        del df['user_distances']

        df['Device_Max_Movement'] = df.device_distances.apply(lambda x:max(x))
        df['Device_Average_Movement'] = df.device_distances.apply(lambda x:np.mean(x))
        df['Device_Total_Movement'] = df.device_distances.apply(lambda x:sum(x))

        del df['device_distances']

        ## Lat Long Clusters
        km = KMeans(n_clusters=3)
        
        solo_location_guys = df[df.user_different_locations_visited==1][['user_id','lat_long']].drop_duplicates()
        solo_location_guys['location_class'] = 'class_0'
        
        cluster_form_df = df[df.user_different_locations_visited>2].groupby(['user_id'])['user_lat_longs'].first().reset_index()
        cluster_list = []
        if cluster_form_df.shape[0]>0:
            for i in range(cluster_form_df.shape[0]):
                identified_clusters = km.fit_predict(cluster_form_df['user_lat_longs'].iloc[i])
                cluster_list.append(identified_clusters)

        
            cluster_form_df['cluster_labels'] = cluster_list
            print('location clustering done')
        
            labels_dict = []
            for i in range(cluster_form_df.shape[0]):
                for j in range(len(cluster_form_df.user_lat_longs.iloc[i])):
                    temp_dict = {'user_id':cluster_form_df.user_id.iloc[i], 
                                'lat_long':cluster_form_df.user_lat_longs.iloc[i][j],
                                'cluster_label':cluster_form_df.cluster_labels.iloc[i][j]}
                    labels_dict.append(temp_dict)
        
            lat_long_cluster_label_df = pd.DataFrame(labels_dict)
            print(lat_long_cluster_label_df.columns)
            lat_long_cluster_label_df['lat_long'] = lat_long_cluster_label_df.lat_long.transform(lambda x: tuple(x))
        
            lat_long_cluster_label_df.cluster_label = lat_long_cluster_label_df.cluster_label.map({0 : "class_0", 1 : "class_1", 2 : "class_2"})
            lat_long_cluster_label_df.rename(columns={'cluster_label':'location_class'}, inplace=True)
            lat_long_cluster_label_df = pd.concat([solo_location_guys,lat_long_cluster_label_df])
        else:
            lat_long_cluster_label_df = solo_location_guys

        dual_location_guys = df[df.user_different_locations_visited==2][['user_id','lat_long']]
        if dual_location_guys.shape[0]>0:
            class0_dual_location_guys = dual_location_guys.groupby('user_id')[['lat_long']].agg(lambda x: x.value_counts().index[0]).reset_index()
            class0_dual_location_guys['location_class'] = 'class_0'

            dual_location_guys = dual_location_guys.drop_duplicates()
            dual_location_guys = pd.merge(dual_location_guys,class0_dual_location_guys, how='left')
            dual_location_guys.fillna('class_1',inplace=True)

            lat_long_cluster_label_df = pd.concat([dual_location_guys,lat_long_cluster_label_df])
        
        # Location Classes Divided
        df = pd.merge(df, lat_long_cluster_label_df)
        print('location clusters assigned')
        # del lat_long_cluster_label_df, cluster_form_df, solo_location_guys, dual_location_guys, class0_dual_location_guys, labels_dict, temp_dict, cluster_list, identified_clusters
        
    print("Done till Distances")
    
    # Timing Clusters!
    df.sort_values(['user_id','updatedAt'],inplace=True)
    
    df['updatedAt_timings'] = df.updatedAt.dt.strftime("%H:%M:%S")
    df['new_updatedAt'] = pd.to_datetime(pd.to_datetime(df.updatedAt_timings).dt.strftime("%H:%M"))
    m = df.groupby('user_id')['new_updatedAt'].nunique()
    df['total_timings'] = df.user_id.map(m)
    del m
    df['operating_time'] = df.updatedAt_timings.transform(lambda x: int(x.split(':')[0])*3600+int(x.split(':')[1])*60+int(x.split(':')[2]))
    
    km = KMeans(n_clusters=3)
    time_cluster_form_df = df[df.total_timings>2].groupby(['user_id'])['operating_time'].apply(list).reset_index()
    time_cluster_form_df.columns = ['user_id','timings']

    if time_cluster_form_df.shape[0]>0:
        new_cluster_list = []
        for i in range(time_cluster_form_df.shape[0]):
            to_train = time_cluster_form_df['timings'].iloc[i]
            identified_clusters = ckwrap.ckmeans(to_train,3)
            new_cluster_list.append(identified_clusters.labels)

        time_cluster_form_df['cluster_labels'] = new_cluster_list
        print('time slots clustering done')
        labels_dict = []
        for i in range(time_cluster_form_df.shape[0]):
            inside_dict = dict(zip(time_cluster_form_df.timings.iloc[i],time_cluster_form_df.cluster_labels.iloc[i]))
            inside_dict = [(time_cluster_form_df.user_id.iloc[i],k,v) for k,v in inside_dict.items()]
            labels_dict.extend(inside_dict)

        timings_cluster_label_df = pd.DataFrame(labels_dict,columns=['user_id','operating_time','cluster_label'])
        timings_cluster_label_df.cluster_label = timings_cluster_label_df.cluster_label.map({0 : "class_0", 1 : "class_1", 2 : "class_2"})
        timings_cluster_label_df.rename(columns={'cluster_label':'other_timing_class'}, inplace=True)
        
        df = pd.merge(df, timings_cluster_label_df, how='left')

    solo_timing_guys = df[df.total_timings==1][['user_id','new_updatedAt']].drop_duplicates()
    solo_timing_guys['timing_class'] = 'class_0'
    
    if df[df.total_timings==2].shape[0]>0:
        dual_timing_guys = df[df.total_timings==2][['user_id','new_updatedAt']]
        class0_dual_timing_guys = dual_timing_guys.groupby('user_id')[['new_updatedAt']].agg(lambda x: x.value_counts().index[0]).reset_index()
        class0_dual_timing_guys['timing_class'] = 'class_0'
        dual_timing_guys = dual_timing_guys.drop_duplicates()
        dual_timing_guys = pd.merge(dual_timing_guys,class0_dual_timing_guys, how='left')
        dual_timing_guys.fillna('class_1',inplace=True)
    
        df = pd.merge(df, pd.concat([solo_timing_guys,dual_timing_guys]), how='left')
    else:
        df = pd.merge(df, solo_timing_guys, how='left')
    
    if 'other_timing_class' in df.columns:
        df.timing_class.fillna(df.other_timing_class,inplace=True)
    
    print('time slots clusters assigned')
    if 'other_timing_class' in df.columns:
        del df['other_timing_class'], df['updatedAt_timings']
    # del labels_dict, new_cluster_list, to_train, identified_clusters, inside_dict, df['other_timing_class'], df['updatedAt_timings'], timings_cluster_label_df, solo_timing_guys, dual_timing_guys, class0_dual_timing_guys, time_cluster_form_df
    
    ### Total Entries
    df['Total_Entries'] = df.user_id.map(df.groupby('user_id')['_id'].count())
    
    ## Class Wise Time Spent in Minutes considering consecutively changing location classes as well
    df.sort_values(['user_id','updatedAt'],inplace=True)
    
    if 'location_class' in df:
        class_ts_df = df[['user_id','updatedAt','location_class']]

        class_ts_df['shift_flag'] = (class_ts_df.user_id.shift(fill_value=class_ts_df.user_id[0])==class_ts_df.user_id) & (class_ts_df.location_class.shift(fill_value=class_ts_df.location_class[0])==class_ts_df.location_class) & ((class_ts_df.updatedAt - class_ts_df.updatedAt.shift(fill_value=class_ts_df.updatedAt[0])).dt.total_seconds() < 1801) ## Considering session of 30Min
        nl = []
        ctr = 0
        for i in class_ts_df.shift_flag:
            if i == False:
                ctr+=1
                nl.append(ctr)
            else:
                nl.append(ctr)
        class_ts_df['rn'] = nl
        print('row_numbers_done')

        m = (class_ts_df.groupby(['user_id','rn'])['updatedAt'].max() - class_ts_df.groupby(['user_id','rn'])['updatedAt'].min()).dt.total_seconds()/60
        m = m.reset_index()
        m.rename(columns={'updatedAt':'time_spent'},inplace=True)
        class_ts_df = pd.merge(class_ts_df,m)
        print('subtraction Done!')

        classwise_ts_df = class_ts_df[['user_id','location_class','time_spent']].drop_duplicates().groupby(['user_id','location_class'])['time_spent'].median().reset_index()
        
        del class_ts_df
        classwise_ts_df = pd.pivot(classwise_ts_df,index='user_id',columns='location_class',values='time_spent').reset_index()
        classwise_ts_df.rename(columns={'class_0': 'time_spent_location_class_0',
                                        'class_1': 'time_spent_location_class_1',
                                        'class_2': 'time_spent_location_class_2'},inplace=True)
        classwise_ts_df.fillna(0,inplace=True)
        df = pd.merge(df,classwise_ts_df,on='user_id')

        ### Location and Class wise Battery
        try:
            m = df.groupby(['user_id','location_class','timing_class'])[['battery']].median().reset_index().fillna(0)
            m = pd.pivot(m,index=['user_id','location_class'],columns='timing_class')['battery'].reset_index()
            # renaming_columns 
            m_col_list = ['user_id']
            m_col_list.append('location_class') 
            for i in m.iloc[:,2:].columns:
                m_col_list.append('timing_'+i[-1])
            m.columns = m_col_list

            m = pd.pivot(m, index='user_id', columns= 'location_class', values=m.columns[2:]).reset_index()
            # renaming_columns 
            m_col_list = ['user_id'] 
            for i in m.iloc[:,1:].columns:
                m_col_list.append('battery_location_'+i[1][-1]+'_'+i[0])
            m.columns = m_col_list
            df = pd.merge(df, m, on='user_id')
            # m.columns = ['user_id', 'battery_location_0_timing_0', 'battery_location_1_timing_0', 'battery_location_2_timing_0',
            #             'battery_location_0_timing_1', 'battery_location_1_timing_1', 'battery_location_2_timing_1',
            #             'battery_location_0_timing_2', 'battery_location_1_timing_2', 'battery_location_2_timing_2']
        except:
            print('LOCATION TIMING LEVEL BATTERY FAILED!!!!!!!!')
            print('\n\n\n')

        ### Location Class wise Battery
        if 'battery' in df.columns:
            m = df.groupby(['user_id','location_class'])[['battery']].median().reset_index().fillna(0)
            # m = pd.pivot(m,index='user_id', columns='location_class', values='battery').reset_index()
            m = pd.pivot(m,index='user_id', columns='location_class')['battery'].reset_index()
            # renaming_columns 
            m_col_list = ['user_id'] 
            for i in m.iloc[:,1:].columns:
                m_col_list.append('battery_location_'+i)
            m.columns = m_col_list
            df = pd.merge(df, m, on='user_id')

        ### Location wise Entries

        m = df.groupby(['user_id','location_class'])[['_id']].count().reset_index().fillna(0)
        m = pd.pivot(m,index='user_id', columns='location_class', values='_id').reset_index()
        # renaming_columns 
        m_col_list = ['user_id'] 
        for i in m.iloc[:,1:].columns:
            m_col_list.append('entries_location_'+i)
        m.columns = m_col_list
        df = pd.merge(df, m, on='user_id')


    ### Time Class wise Battery
    if 'battery' in df.columns:
        m = df.groupby(['user_id','timing_class'])[['battery']].median().reset_index().fillna(0)
        # m = pd.pivot(m,index='user_id', columns='timing_class', values='battery').reset_index()
        m = pd.pivot(m,index='user_id', columns='timing_class')['battery'].reset_index()
        # renaming_columns 
        m_col_list = ['user_id'] 
        for i in m.iloc[:,1:].columns:
            m_col_list.append('battery_timing_'+i)
        m.columns = m_col_list
        df = pd.merge(df, m, on='user_id')

    ### Timing wise Entries

    m = df.groupby(['user_id','timing_class'])[['_id']].count().reset_index().fillna(0)
    if m.shape[0]>0:
        m = pd.pivot(m,index='user_id', columns='timing_class', values='_id').reset_index()
        m_col_list = ['user_id']
        for i in m.iloc[:,1:].columns:
            m_col_list.append('entries_timing_'+i)
        m.columns = m_col_list
        df = pd.merge(df, m, on='user_id')

    ## User Level Day Periods
    #Day period Counts - Frequency of Phone Usage according to time period
    m = df.groupby(['user_id','Day_Period'])[['_id']].count().reset_index()
    m = pd.pivot(m,index='user_id',columns='Day_Period',values='_id').reset_index().fillna(0)
    # renaming_columns 
    m_col_list = ['user_id']
    for i in m.iloc[:,1:].columns:
        m_col_list.append('Associated_User_'+i+'_Count')

    m.columns = m_col_list
    df = pd.merge(df,m)
        
    ## Device Level Day Periods
    #Day period Counts - Frequency of Phone Usage according to time period
    m = df.groupby(['device','Day_Period'])[['_id']].count().reset_index()
    m = pd.pivot(m,index='device',columns='Day_Period',values='_id').reset_index().fillna(0)
    # renaming_columns 
    m_col_list = ['device']
    for i in m.iloc[:,1:].columns:
        m_col_list.append('Associated_Device_'+i+'_Count')
    m.columns = m_col_list
    df = pd.merge(df,m)

    print("Done till Day Periods")

    ## Battery Handling
    if 'battery' in df.columns:
        battery_info = df.groupby(['user_id','Day_Period'])[['battery']].median().reset_index()
        # battery_df = pd.pivot(battery_info,index='user_id',columns='Day_Period',values='battery')
        battery_df = pd.pivot(battery_info,index='user_id',columns='Day_Period')['battery'].reset_index()
        battery_df_columns = battery_df.columns[1:]
        # for day_times in ['Morning','Afternoon','Evening','Night']:
        #     if day_times in battery_df.columns:
        #         battery_df_columns.append(day_times)
        battery_df['Median_Battery'] = battery_df[battery_df_columns].median(axis=1)

        battery_df.fillna(battery_df.Median_Battery,inplace=True)
    
        for day_times in battery_df_columns:
            battery_df.rename(columns={day_times:day_times+"_Battery"},inplace=True)
        battery_df.drop('Median_Battery',axis=1,inplace=True)
        battery_df.fillna(battery_df.median(),inplace=True)
        df = pd.merge(df,battery_df)
        df.drop('battery',axis=1,inplace=True)
    
        del df['Day_Period']
        del battery_df,battery_info
    
    ## Associated Devices & Users
    user_devices_df = df[['user_id','device']].drop_duplicates()
    m = user_devices_df.groupby('user_id')[['device']].count().reset_index()
    m.columns = ['user_id','associated_devices']
    n = user_devices_df.groupby('device')[['user_id']].count().reset_index()
    n.columns = ['device','associated_users']

    df = pd.merge(df,m)
    df = pd.merge(df,n)
    
    print('mongo_preprocess done till associated devices and users!')
    
    ## Brand Handling
    brand_mapper = {k:k for k in ['vivo', 'samsung', 'oppo', 'realme', 'redmi', 'xiaomi', 'oneplus', 'poco']}
    df.brand = df.brand.map(brand_mapper)
    df.brand.fillna('Others',inplace=True)
    
    ## User Level Brands
    m = df.groupby(['user_id','brand'])[['_id']].count().reset_index()
    m['_id'] = 1
    m = pd.pivot(m,index='user_id',columns='brand',values='_id').fillna(0).reset_index()
    m['user_total_brands'] = m.iloc[:,1:].sum(axis=1)
    # renaming_columns
    m_col_list = ['user_id']
    for i in m.iloc[:,1:-1].columns:
        m_col_list.append('user_associated_brand_'+i+'_count')
    m_col_list.append(m.columns[-1])
    m.columns = m_col_list

    df = pd.merge(df,m)
    
    ## Device Level Brands
    m = df.groupby(['device','brand'])[['_id']].count().reset_index()
    m['_id'] = 1
    m = pd.pivot(m,index='device',columns='brand',values='_id').fillna(0).reset_index()
    m['device_total_brands'] = m.iloc[:,1:].sum(axis=1)
    # renaming_columns
    m_col_list = ['device']
    for i in m.iloc[:,1:-1].columns:
        m_col_list.append('device_brand_'+i+'_count')
    m_col_list.append(m.columns[-1])
    m.columns = m_col_list
    df = pd.merge(df,m)
    
    del df['brand']
    
    # Carrier Null Values
    if 'carrier' in df.columns:
        try:
            m = df.groupby('device')['carrier'].agg(lambda x: pd.Series.mode(x)[0])
            df.carrier.fillna(df.device.map(m),inplace=True)
        except:
            print('Carrier HANDLING FAILEDDDD!!!')
        df.carrier.fillna('Others',inplace=True)
    
        #Cleaning Carrier Names
        df.carrier = df.carrier.apply(lambda x:" ".join(re.sub(r'[^a-zA-Z ]+', ' ', str(x).upper()).split()))
        def carrier_selection(x):
            if "JIO" in x:
                return "JIO"
            elif "AIRT" in x:
                return "AIRTEL"
            elif "VODA" in x:
                return "VODAFONE"
            elif "IDEA" in x:
                return "VODAFONE"
            elif "VI" in x and "SERVICE" not in x:
                return "VODAFONE"
            elif "BSNL" in x:
                return "BSNL"
            elif "MTNL" in x:
                return "MTNL"
            else:
                return x
    
        df.carrier = df.carrier.apply(carrier_selection)
        df.carrier.replace('',np.nan,inplace=True)
        carrier_mapper = {k:k for k in ['JIO', 'AIRTEL', 'VODAFONE', 'BSNL']}
        df.carrier = df.carrier.map(carrier_mapper)
        df.carrier.fillna('Other',inplace=True)
        m = df.groupby('user_id')['carrier'].agg(lambda x:scipy.stats.mode(x)[0])
        df.carrier.fillna(df.user_id.map(m),inplace=True)
        df.carrier.replace(0,'Others',inplace=True)
    
    ## User Level Carriers
        m = df.groupby(['user_id','carrier'])[['_id']].count().reset_index()
        m['_id']=1
        m = pd.pivot(m,index='user_id',columns='carrier',values='_id').reset_index().fillna(0)
        m['total_user_carriers'] = m.iloc[:,1:].sum(axis=1)
        # renaming_columns
        m_col_list = ['user_id']
        for i in m.iloc[:,1:-1].columns:
            m_col_list.append('user_carriers_'+i)
        m_col_list.append(m.columns[-1])
        m.columns = m_col_list
    
        df = pd.merge(df,m)
    
        ## Device Level Carriers
        m = df.groupby(['device','carrier'])[['_id']].count().reset_index()
        m['_id']=1
        m = pd.pivot(m,index='device',columns='carrier',values='_id').reset_index().fillna(0)
        m['device_total_carriers'] = m.iloc[:,1:].sum(axis=1)
        # renaming_columns
        m_col_list = ['device']
        for i in m.iloc[:,1:-1].columns:
            m_col_list.append('device_carriers_'+i)
        m_col_list.append(m.columns[-1])
        m.columns = m_col_list

        df = pd.merge(df,m)
    
        print('mongo_preprocess carriers handled!')
    
    # OS_Version Null Values
    df['os-version'] = df['os-version'].astype(str)
    m = df[df['os-version']!='nan'].groupby('device')['os-version'].agg(lambda x: pd.Series.mode(x)[0])
    df['os-version'].replace('nan',np.nan,inplace=True)
    df['os-version'].fillna(df.device.map(m),inplace=True)
    
    def osv_mapper(x):
        m = str(x).replace('[^0-9.]+', '').split('.')
        if len(m)>1:
            return float('.'.join(m[:2]))
        else:
            return float(m[0])

    df['os-version'] = df['os-version'].apply(osv_mapper)
    m = df[~df['os-version'].isnull()].groupby('device').tail(5).groupby('device')['os-version'].agg(lambda x: pd.Series.mode(x))
    df['os-version'].fillna(df.device.map(m),inplace=True)
    
    print('mongo_preprocess os_version mapped!')
    
    ## Aggregating OS_Version
    # Number of different versions,latest version
    m = df.groupby('user_id')['os-version'].nunique()
    df['different_os_versions'] = df.user_id.map(m)
    df.sort_values('updatedAt',ascending=True)
    m = df.groupby('user_id')['os-version'].max()
    df['latest_os_version'] = df.user_id.map(m)

    del df['os-version']
    
    # Preferred Locale
    df['preferred-locale'] = df['preferred-locale'].astype(str)
    m = df[df['preferred-locale']!='nan'].groupby('user_id')['preferred-locale'].agg(lambda x: scipy.stats.mode(x)[0])
    df['preferred-locale'].replace('nan',np.nan,inplace=True)
    df['preferred-locale'].fillna(df.user_id.map(m),inplace=True)
    df['preferred-locale'] = df['preferred-locale'].transform(lambda x:str(x)[:2])
    df['preferred-locale'].replace(['na','un'],'en',inplace=True)
    
    ## Aggregating Locales
    df['preferred-locale'] = df['preferred-locale'].apply(lambda x: str(x))
    df = df[df['preferred-locale']!="['en']"]
    m = df.groupby('user_id')['preferred-locale'].nunique()#.reset_index()
    df['different_locales'] = df.user_id.map(m)
    
    del df['preferred-locale']
    del m
    ## User Level URL Aggregating
    # Number of page visits
    m = df.groupby(['user_id','url'])[['_id']].count().reset_index()
    m = pd.pivot(m,index='user_id',columns='url',values='_id').reset_index().fillna(0)
    m['user_total_page_visits'] = m.iloc[:,1:].sum(axis=1)
    # renaming_columns
    m_col_list = ['user_id']
    for i in m.iloc[:,1:-1].columns:
        m_col_list.append('user_page_visit_'+i)
    m_col_list.append(m.columns[-1])
    m.columns = m_col_list
    
    df = pd.merge(df,m)
                
    ## Device Level URL Aggregating
    # Number of page visits
    m = df.groupby(['device','url'])[['_id']].count().reset_index().fillna(0)
    m.replace('device','device_page',inplace=True)
    m = pd.pivot(m,index='device',columns='url',values='_id').reset_index()
    m['device_total_page_visits'] = m.iloc[:,1:].sum(axis=1)
    # renaming_columns
    m_col_list = ['device']
    for i in m.iloc[:,1:-1].columns:
        if i == 'device_page': ## Special handling in case of page name as device ---> conflict with the device id column named as device
            i = 'device'
        m_col_list.append('device_page_visit_'+i)
    m_col_list.append(m.columns[-1])
    m.columns = m_col_list
    m = m.reset_index()
    
    df = pd.merge(df,m)
    # delete Unnecessary Columns
    if 'location_class' in df.columns:
        df.drop(['latitude','longitude','lat_long','location_class'],axis=1,inplace=True)
    df.drop(['operating_time','timing_class'],axis=1,inplace=True)
    df.drop(['url','_id','updatedAt','new_updatedAt'],axis=1,inplace=True)

    print('mongo_preprocess url handled!')

    duplicate_handling_remaining_columns = ['device_different_locations_visited',
                 'device_brand_Other_count',
                 'device_brand_oneplus_count',
                 'device_brand_oppo_count',
                 'device_brand_poco_count',
                 'device_brand_realme_count',
                 'device_brand_redmi_count',
                 'device_brand_samsung_count',
                 'device_brand_vivo_count',
                 'device_brand_xiaomi_count',
                 'device_total_brands',
                 'device_carriers_AIRTEL',
                 'device_carriers_BSNL',
                 'device_carriers_JIO',
                 'device_carriers_Other',
                 'device_carriers_VODAFONE',
                 'device_total_carriers',
                 'device_page_visit_app',
                 'device_page_visit_device',
                 'device_page_visit_documents',
                 'device_page_visit_files',
                 'device_page_visit_giftcards',
                 'device_page_visit_gold',
                 'device_page_visit_home',
                 'device_page_visit_payments',
                 'device_page_visit_qr',
                 'device_page_visit_recharges',
                 'device_page_visit_session',
                 'device_page_visit_shopping',
                 'device_page_visit_subscriptions',
                 'device_page_visit_virtual_cards',
                 'device_total_page_visits']
    
    df = df[~df.user_id.isnull()]
    lll = [i for i in duplicate_handling_remaining_columns if i in df.columns]
    m = df.groupby('user_id')[lll].sum().reset_index()
    new_m_col = ['user_id']
    rest = ['associated_'+i for i in m.columns[1:]]
    new_m_col.extend(rest)
    m.columns = new_m_col
    
    df.drop(lll,axis=1,inplace=True)
    df = pd.merge(df,m)
    
    m = df.groupby('user_id')[['associated_users']].sum().reset_index()
    df.drop('associated_users',axis=1,inplace=True)
    df = pd.merge(df,m)
    
    newll = [i for i in ['Device_Max_Movement', 'Device_Average_Movement'] if i in df.columns]
    if len(newll)>0:
        m = df.groupby('user_id')[newll].max().reset_index()
        new_m = ['user_id']
        new_m.extend(['Associated_'+i for i in m.columns[1:]])
        m.columns = new_m
        df.drop(newll,axis=1,inplace=True)
        df = pd.merge(df,m)
    
    print('mongo_preprocess movements processed!')
    
    newll = [i for i in ['Associated_Device_Afternoon_Count','Associated_Device_Evening_Count','Associated_Device_Morning_Count','Associated_Device_Night_Count'] if i in df.columns]
    if len(newll)>0:
        m = df.groupby('user_id')[newll].sum().reset_index()
        df.drop(newll,axis=1,inplace=True)
        df = pd.merge(df,m)
        
    ### Final Columns Handling
    final_cols = ['Afternoon_Battery','Evening_Battery','Morning_Battery','Night_Battery',
                'Phone_Build_Median', 'user_different_locations_visited', 'User_Max_Movement',
                'User_Average_Movement', 'Associated_User_Afternoon_Count',
                'Associated_User_Evening_Count', 'Associated_User_Morning_Count',
                'Associated_User_Night_Count', 'associated_devices',
                'user_associated_brand_Other_count','user_associated_brand_oneplus_count',
                'user_associated_brand_oppo_count', 'user_associated_brand_poco_count',
                'user_associated_brand_realme_count','user_associated_brand_redmi_count',
                'user_associated_brand_samsung_count','user_associated_brand_vivo_count',
                'user_associated_brand_xiaomi_count', 'user_total_brands',
                'user_carriers_AIRTEL', 'user_carriers_BSNL', 'user_carriers_JIO',
                'user_carriers_Other', 'user_carriers_VODAFONE', 'total_user_carriers',
                'different_os_versions', 'latest_os_version', 'different_locales',
                'user_page_visit_app', 'user_page_visit_device',
                'user_page_visit_documents', 'user_page_visit_files',
                'user_page_visit_giftcards', 'user_page_visit_gold',
                'user_page_visit_home', 'user_page_visit_payments',
                'user_page_visit_qr', 'user_page_visit_recharges',
                'user_page_visit_session', 'user_page_visit_shopping',
                'user_page_visit_subscriptions', 'user_page_visit_virtual_cards',
                'user_page_visit_contacts', 'user_page_visit_notifications',
                'user_page_visit_onboarding', 'user_page_visit_references',
                'user_page_visit_referrer', 'user_page_visit_repayments',
                'user_page_visit_service', 'user_total_page_visits', 
                'associated_device_different_locations_visited',
                'associated_device_brand_Other_count',
                'associated_device_brand_oneplus_count',
                'associated_device_brand_oppo_count',
                'associated_device_brand_poco_count',
                'associated_device_brand_realme_count',
                'associated_device_brand_redmi_count',
                'associated_device_brand_samsung_count',
                'associated_device_brand_vivo_count',
                'associated_device_brand_xiaomi_count',
                'associated_device_total_brands', 'associated_device_carriers_AIRTEL',
                'associated_device_carriers_BSNL', 'associated_device_carriers_JIO',
                'associated_device_carriers_Other',
                'associated_device_carriers_VODAFONE',
                'associated_device_total_carriers', 'associated_device_page_visit_app',
                'associated_device_page_visit_device',
                'associated_device_page_visit_documents',
                'associated_device_page_visit_files',
                'associated_device_page_visit_giftcards',
                'associated_device_page_visit_gold',
                'associated_device_page_visit_home',
                'associated_device_page_visit_payments',
                'associated_device_page_visit_qr',
                'associated_device_page_visit_recharges',
                'associated_device_page_visit_session',
                'associated_device_page_visit_shopping',
                'associated_device_page_visit_subscriptions',
                'associated_device_page_visit_virtual_cards',
                'associated_device_page_visit_contacts',
                'associated_device_page_visit_notifications',
                'associated_device_page_visit_onboarding',
                'associated_device_page_visit_references',
                'associated_device_page_visit_referrer',
                'associated_device_page_visit_repayments',
                'associated_device_page_visit_service',
                'associated_device_total_page_visits', 'associated_users',
                'Associated_Device_Max_Movement', 'Associated_Device_Average_Movement',
                'Associated_Device_Afternoon_Count', 'Associated_Device_Evening_Count',
                'Associated_Device_Morning_Count', 'Associated_Device_Night_Count',
                'Associated_Device_Total_Movement', 'Total_Entries', 'User_Total_Movement', 
                'battery_location_0_timing_0', 'battery_location_0_timing_1', 'battery_location_0_timing_2', 
                'battery_location_1_timing_0', 'battery_location_1_timing_1', 'battery_location_1_timing_2', 
                'battery_location_2_timing_0', 'battery_location_2_timing_1', 'battery_location_2_timing_2', 
                'battery_location_class_0', 'battery_location_class_1', 'battery_location_class_2', 
                'battery_timing_class_0', 'battery_timing_class_1', 'battery_timing_class_2', 
                'entries_location_class_0', 'entries_location_class_1', 'entries_location_class_2', 
                'entries_timing_class_0', 'entries_timing_class_1', 'entries_timing_class_2', 
                'time_spent_location_class_0', 'time_spent_location_class_1', 'time_spent_location_class_2', 
                'total_timings','address_county_specific', 'address_suburb_specific', 'address_city_specific', 
                'address_town_specific', 'address_village_specific', 'address_city_district_specific', 'address_neighbourhood_specific']

    print('mongo_preprocess reached_till_final_columns!')

    to_be_handled_cols = [i for i in final_cols if i not in df.columns]
    df[to_be_handled_cols] = 0
    removal_columns = [i for i in df.columns if i not in final_cols]
    df = df[df.user_id == user_to_underwrite].drop(removal_columns,axis=1).drop_duplicates().reset_index(drop=True)
    return df.iloc[0].to_dict()
 
##################################################################################################################################################################


def mongo_query_user(collection,users):
    pipeline = [{
                    "$match":
                        {"user_id": users
                            # {
                            #     "$in":users
                            # }
                        }
                }]
    cursor = collection.aggregate(pipeline)
    cursor_list = list(cursor)
    return cursor_list

def mongo_query_device(collection,devices):
    pipeline = [{
                    "$match":
                        {"device": 
                            {
                                "$in": devices
                            }
                        }
                }]
    cursor = collection.aggregate(pipeline)
    cursor_list = list(cursor)
    return cursor_list

def find_unique_devices(cursor):
    devices = []
    for i in cursor:
        if type(i)==dict and 'device' in i and i['device'] not in devices:
            devices.append(i['device'])
    return devices


###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################

def mongo_handler(client,users, geolocator):
    db = client['fury']
    collection = db['userphones']
    print('Mongo Handler for Users:',users)
    cursor1 = mongo_query_user(collection,users)
    print('Mongo Users Queried')
    unique_devices = find_unique_devices(cursor1)
    print(f'Associated Devices of {users}:{len(unique_devices)}')
    if len(unique_devices)>0:
        cursor2 = mongo_query_device(collection,unique_devices)
        print('USER sent to Mongo preprocess:\n',users)
        if len(cursor1)+len(cursor2)>0:
            preprocessed_dictionary = mongo_preprocess(users,cursor1,cursor2)
            print(len(preprocessed_dictionary))
            return preprocessed_dictionary
        else:
            return "No Output"
    else:
        return "No DEVICES!!!!"
###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################

# connection_URI = os.getenv("MONGO_DB_URL")

mongo_client = MongoClient("mongodb://localhost/fury?ssl=false&authSource=admin",port = 27017)
# mongo_client = MongoClient("mongodb://admin:8K4E5A6U.oMAtw@localhost/fury?ssl=false&authSource=admin",port = 27020)
# mongo_client = MongoClient(connection_URI,port = 27017)

df = pd.read_csv('profitability_target.csv')
# already = pd.read_csv('logs_DB_processed.csv')
# print(already.user_id.nunique())
user_list = df.user_id.to_list()
if 1 in user_list:
    user_list.remove(1)

# for i in already.user_id:
#     if i in user_list:
#         user_list.remove(i)

print('total users:\t',df.shape)
print('to be found users:\t',len(user_list))

# ndf = already.copy()
ndf = pd.DataFrame()

# print(ndf.shape)

print('Users List Length\t',len(user_list))

# temp_list = [628112, 628121, 628132, 628152, 628154, 628181, 628190, 628192, 628198, 628206, 628224, 628254, 628267, 628288, 628310, 378652, 628581, 628607, 628611, 628616, 628641, 628665, 628671, 628684, 628692, 628700, 628705, 628777, 628772, 628822, 628829, 629004, 629019, 629022, 629058, 629066, 629098, 629105, 629108, 629118, 629132, 629138, 629242, 629309, 629316, 629446, 629447, 629465, 629507, 629612, 629658, 629730, 629808, 629829, 629881, 629902, 629910, 629913, 629920, 629924, 629935, 629944, 630186, 630252, 630339, 630383, 630401, 630493, 630500, 630542, 630628, 630791, 631039, 631054, 631102, 631138, 631163, 631171, 631178, 631179, 631195, 631217, 631246, 631251, 631345, 631378, 631387, 631570, 631590, 631602, 631612, 631668, 631738, 631810, 631886, 631913, 631914, 631918, 631931, 631974, 632120, 632141, 632175, 632219, 632297, 632298, 632319, 632331, 632483, 632601, 632608, 632626, 632709, 632744, 633094, 633132, 633175, 633183, 633215, 633258, 633382, 633526, 633631, 633645, 633735, 633800, 633832, 633889, 633935, 633951, 633977, 633986, 634091, 634115, 634145, 634167, 634183, 634191, 634218, 634257, 634264, 634300, 634322, 634342, 634344, 634372, 634432, 634435, 634441, 634486, 634491, 634523, 634529, 634596, 634603, 634609, 634636, 634694, 634699, 634706, 634720, 634730, 628104, 634826, 634830, 634842, 634852, 634870, 634871, 634893, 634909, 634921, 634926, 634931, 634948, 634960, 634966, 634981, 634992, 635033, 635104, 635112, 635117, 635176, 635226, 635228, 635234, 635236, 635269, 635299, 635305, 635317, 635346, 635365, 635381, 635436, 635481, 635533, 635534, 635558, 635561, 635566, 635621, 635623, 635647, 635772, 635807, 635830, 635946, 635951, 635964, 636000, 636014, 636061, 636140, 636144, 636163, 636207, 636208, 636267, 636292, 636311, 636330, 636429, 636446, 636463, 636508, 636554, 636575, 636661, 636687, 636696, 636724, 636760, 636808, 636811, 636833, 636848, 636902, 636918, 636921, 636984, 637008, 637032, 637040, 637061, 637125, 637141, 637154, 637188, 637276, 637277, 637281, 637298, 637334, 637434, 637476, 637485, 637488, 637528, 637532, 637543, 637548, 637553, 637579, 637580, 637582, 637586, 637590, 637621, 637644, 637650, 637651, 637663, 637711, 637720, 637722, 637732, 637733, 637760, 637763, 637767, 637772, 637829, 637839, 637845, 637866, 637905, 637909, 637919, 637934, 637946, 637957, 637965, 638054, 638110, 638126, 638134, 638152, 638206, 305656, 638250, 638287, 638326, 638338, 638343, 638369, 638373, 638384, 638389, 638396, 638404, 638410, 638413, 638416, 638435, 638437, 638443, 638458, 638477, 638495, 638505, 638535, 638548, 638587, 638598, 638601, 638602, 638614, 638635, 638636, 638638, 638649, 638678, 638687, 638693, 638695, 638701, 638730, 638733, 638737, 638751, 638754, 638761, 638770, 638773, 638777, 638778, 638795, 638804, 638808, 638818, 638832, 638834, 638838, 638941, 638991, 639006, 639013, 639027, 639031, 639060, 639075, 639096, 639097, 639176, 639246, 639253, 639276, 639305, 639315, 639332, 639364, 639367, 639412, 639428, 639540, 639575, 639579, 639582, 639599, 639630, 639634, 639651, 639668, 639682, 639690, 639715, 639716, 639717, 639725, 639734, 639752, 639789, 639805, 639818, 639824, 639875, 639880, 639916, 640011, 640038, 640061, 640072, 640097, 640098, 640121, 640126, 640169, 640179, 640194, 640218, 640226, 640239, 640272, 640273, 640312, 640325, 640338, 640340, 640379, 640387, 640400, 640414, 640417, 640418, 640421, 640423, 640445, 480379, 640453, 640454, 640469, 640474, 640519, 640524, 640531, 640534, 640548, 640555, 640563, 640566, 640576, 640590, 640600, 640612, 640640, 640667, 640670, 640699, 640722, 640735, 640736, 640740, 640749, 640787, 640792, 640794, 640796, 640819, 640833, 640835, 640860]
user_list = user_list[::-1]

# user_list = [412263]

from time import sleep

geolocator = Nominatim(user_agent="geoapiExercises")

# for i in range(0,len(user_list),300):
for i in range(0,len(user_list),1):
    # preprocessed = mongo_handler(mongo_client, user_list[i:min(i+300,len(user_list)):])
    print('\n\n\n\n\n') 
    print('USER_HANDLING:\t\t',user_list[i])
    print('\n\n\n\n\n')
    # preprocessed = mongo_handler(mongo_client, user_list[i:min(i+1,len(user_list)+1)])
    preprocessed = mongo_handler(mongo_client, user_list[i], geolocator)
    if (type(preprocessed)==dict) and (len(preprocessed) > 0):    
        ndf = pd.concat([ndf, pd.DataFrame(preprocessed, index=np.arange(len(list(preprocessed.keys())[0])))], ignore_index = True) 
        print('~!@#$$%^&*()(*&^%$#@!~~!@#$%^&*(())')
        print('Users Handled\t',i)
        ndf.drop_duplicates().to_csv('logs_DB_processed.csv')
    if i+1%10001==0:
        sleep(100)
    if i+1%10==0:
        sleep(10)
    break
ndf.drop_duplicates().to_csv('logs_DB_processed.csv')
