#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
import requests

import numpy as np
import pandas as pd

#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

def request_a_diccionario(url):
    r = requests.get(url)
    return r.json()

#------------------------------------------------------------------------------------------------------------------------

def generar_df(items):
    
    campo_id = []
    campo_site_id = []
    campo_title = []
    campo_price = []
    campo_currency_id = []
    campo_available_quantity = []
    campo_sold_quantity = []
    campo_buying_mode = []
    campo_listing_type_id = []
    campo_condition = []
    campo_accepts_mercadopago = []
    campo_address__state_name = []
    campo_address__city_name = []
    campo_shipping__free_shipping = []
    campo_shipping__store_pick_up = []
    campo_original_price = []
    campo_category_id = []
    campo_official_store_id = []
    campo_catalog_product_id = []
    campo_attributes__marca = []
    campo_attributes__modelo= []

    for item in items["results"]:

        campo_id.append(item["id"])
        campo_site_id.append(item["site_id"])
        campo_title.append(item["title"])
        campo_price.append(item["price"])
        campo_currency_id.append(item["currency_id"])
        campo_available_quantity.append(item["available_quantity"])
        campo_sold_quantity.append(item["sold_quantity"])
        campo_buying_mode.append(item["buying_mode"])
        campo_listing_type_id.append(item["listing_type_id"])
        campo_condition.append(item["condition"])
        campo_accepts_mercadopago.append(item["accepts_mercadopago"])
        campo_address__state_name.append(item["address"]["state_name"])
        campo_address__city_name.append(item["address"]["city_name"])
        campo_shipping__free_shipping.append(item["shipping"]["free_shipping"])
        campo_shipping__store_pick_up.append(item["shipping"]["store_pick_up"])
        campo_original_price.append(item["original_price"])
        campo_category_id.append(item["category_id"])
        campo_official_store_id.append(item["official_store_id"])
        campo_catalog_product_id.append(item["catalog_product_id"])
         
        valor_campo_attributes__marca = None
        valor_campo_attributes__modelo = None
        atributos = item["attributes"]
        for atributo in atributos:
            name = atributo["name"]
            if str(name).lower() == "marca":
                valor_campo_attributes__marca = atributo["value_name"]
            elif str(name).lower() == "modelo":
                valor_campo_attributes__modelo = atributo["value_name"]
        campo_attributes__marca.append(valor_campo_attributes__marca)
        campo_attributes__modelo.append(valor_campo_attributes__modelo)
        

    dicc_columnas = {"id":campo_id, "site_id":campo_site_id, "title":campo_title, "price":campo_price,
                     "currency_id":campo_currency_id, "available_quantity":campo_available_quantity,
                     "sold_quantity":campo_sold_quantity, "buying_mode":campo_buying_mode, 
                     "listing_type_id":campo_listing_type_id, "condition":campo_condition, 
                     "accepts_mercadopago":campo_accepts_mercadopago, 
                     "address__state_name":campo_address__state_name, "address__city_name":campo_address__city_name,
                     "shipping__free_shipping":campo_shipping__free_shipping, "shipping__store_pick_up":campo_shipping__store_pick_up,
                     "original_price":campo_original_price, "category_id":campo_category_id, 
                     "official_store_id":campo_official_store_id, "catalog_product_id":campo_catalog_product_id,
                     "attributes__marca":campo_attributes__marca, "attributes__modelo":campo_attributes__modelo
                     }
    
    return pd.DataFrame(dicc_columnas)

#------------------------------------------------------------------------------------------------------------------------

def analizar_nulos(df, variable):
    cant_nulos = df[variable].isnull().sum()
    porc_nulos = round(cant_nulos * 100 / len(df), 2)
    print("La variable", str(variable), "tiene", cant_nulos, "registros nulos (que representan un", porc_nulos, "% del total).")
    filtro_nulos = df[variable].isnull()
    return filtro_nulos

#------------------------------------------------------------------------------------------------------------------------

def calcular_precio_sin_dcto(x):
    tiene_dcto = x["tiene_dcto"]
    precio_con_dcto = x["price"]
    precio_original = x["original_price"]
    if tiene_dcto:
        return precio_original
    else:
        return precio_con_dcto

#------------------------------------------------------------------------------------------------------------------------

def calcular_dcto_porcentual(x):
    precio_sin_dcto = x["precio_sin_dcto"]
    dcto_nominal = x["dcto_nominal"]
    return round(dcto_nominal * 100 / precio_sin_dcto, 2)

#------------------------------------------------------------------------------------------------------------------------