# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 01:06:55 2019

@author: pri_p
"""



import pandas as pd
import json
import urllib
import requests as rq
import base64

def get_oauth_token():
    url = "https://api.idealista.com/oauth/token"    
    apikey= 'yourapikey' #sent by idealista
    secret= 'yoursecret'  #sent by idealista
    auth = base64.b64encode(bytes(apikey + ':' + secret, "utf-8"))
    headers = {'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' ,'Authorization' : 'Basic ' + auth.decode("utf-8")}
    params = urllib.parse.urlencode({'grant_type':'client_credentials'})
    content = rq.post(url,headers = headers, params=params)
    bearer_token = json.loads(content.text)['access_token']
    return bearer_token

def search_api(token, url):  
    headers = {'Content-Type': 'Content-Type: multipart/form-data;', 'Authorization' : 'Bearer ' + token}
    content = rq.post(url, headers = headers)
    result = json.loads(content.text)
    return result


country = 'es' #values: es, it, pt
locale = 'es' #values: es, it, pt, en, ca
language = 'es' #
max_items = '50'
operation = 'sale' 
property_type = 'homes'
order = 'publicationDate' 
center = '40.4167,-3.70325' 
distance = '15000'
sort = 'desc'
bankOffer = 'false'

df_tot = pd.DataFrame()
limit = 2

for i1 in ('true','false'):
    for i2 in ('true','false'):
        for i3 in ('true','false'):
            for i4 in ('furnished','furnishedKitchen'):
                for i in range(1,limit):
                    url = ('https://api.idealista.com/3.5/'+country+'/search?operation='+operation+#"&locale="+locale+
                           '&maxItems='+max_items+
                           '&order='+order+
                           '&center='+center+
                           '&distance='+distance+
                           '&propertyType='+property_type+
                           '&sort='+sort+ 
                           '&numPage=%s'+
                           '&airConditioning=%s'+
                           '&swimmingPool=%s'+
                           '&terrance=%s'+
                           '&furnished=%s'+                           
                           '&language='+language) %(i,i1,i2,i3,i4)  
                    a = search_api(get_oauth_token(), url)
                    df = pd.DataFrame.from_dict(a['elementList'])
                    df['AC'] = i1
                    df['Piscina']=i2
                    df['Terraza']=i3
                    df['Amueblado']=i4
                    df_tot = pd.concat([df_tot,df])
                

df_tot = df_tot.reset_index()
df_tot.shape

df_tot.to_csv('C:\\Sales310719.csv', sep='\t', encoding='utf-8')