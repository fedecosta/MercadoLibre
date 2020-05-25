#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
import requests

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_score, recall_score, f1_score

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

    campo_seller__power_seller_status = []
    campo_seller__car_dealer = []
    campo_seller__eshop__eshop_experience = []
    campo_seller__eshop__eshop_id = []
    campo_tags = []

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

        campo_seller__power_seller_status.append(item["seller"]["power_seller_status"])
        campo_seller__car_dealer.append(item["seller"]["car_dealer"])
        if "eshop" in item["seller"].keys() and "eshop_experience" in item["seller"]["eshop"].keys():
            campo_seller__eshop__eshop_experience.append(item["seller"]["eshop"]["eshop_experience"])
        else:
            campo_seller__eshop__eshop_experience.append(None)
        if "eshop" in item["seller"].keys():  
            campo_seller__eshop__eshop_id.append(item["seller"]["eshop"]["eshop_id"])
        else:
            campo_seller__eshop__eshop_id.append(None)
        campo_tags.append(item["tags"])
        

    dicc_columnas = {"id":campo_id, "site_id":campo_site_id, "title":campo_title, "price":campo_price,
                     "currency_id":campo_currency_id, "available_quantity":campo_available_quantity,
                     "sold_quantity":campo_sold_quantity, "buying_mode":campo_buying_mode, 
                     "listing_type_id":campo_listing_type_id, "condition":campo_condition, 
                     "accepts_mercadopago":campo_accepts_mercadopago, 
                     "address__state_name":campo_address__state_name, "address__city_name":campo_address__city_name,
                     "shipping__free_shipping":campo_shipping__free_shipping, "shipping__store_pick_up":campo_shipping__store_pick_up,
                     "original_price":campo_original_price, "category_id":campo_category_id, 
                     "official_store_id":campo_official_store_id, "catalog_product_id":campo_catalog_product_id,
                     "attributes__marca":campo_attributes__marca, "attributes__modelo":campo_attributes__modelo,
                     "seller__power_seller_status":campo_seller__power_seller_status, "seller__car_dealer":campo_seller__car_dealer,
                     "seller__eshop__eshop_experience":campo_seller__eshop__eshop_experience, 
                     "seller__eshop__eshop_id":campo_seller__eshop__eshop_id, "tags":campo_tags 
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

def generar_df_metricas(y_true, y_pred, labels):
    
    columna_clase = []
    columna_precision = []
    columna_recall = []
    columna_cant_casos = []
    columna_f1 = []
    columna_f1_sklearn = []
    
    clases = labels
    #para construir la cm bien ordenada y prolija: (ya esta chequeado que anda bien; igual que sklearn)
    #cm = confusion_matrix(y_true_pais, y_pred_pais, labels = incomes)
    #df_confusion_matrix = pd.DataFrame({"pred_low":cm.T[0], "pred_medium":cm.T[1], "pred_high":cm.T[2]},
    #              index = ["true_low", "true_medium", "true_high"],
    #              columns = ["pred_low", "pred_medium", "pred_high"])

    precision = precision_score(y_true, y_pred, average=None, labels = clases)
    recall = recall_score(y_true, y_pred, average=None, labels = clases)
    for i, clase in enumerate(clases):
        precision_clase = precision[i]
        recall_clase = recall[i]
        f1_clase = 2 * (precision_clase * recall_clase) / (precision_clase + recall_clase)
        cant_casos = (y_true == clase).sum()

        columna_clase.append(clase)
        columna_precision.append(precision_clase)
        columna_recall.append(recall_clase)
        columna_cant_casos.append(cant_casos)
        columna_f1.append(f1_clase)

    df_metricas_clase = pd.DataFrame({
                          "clase" : columna_clase,
                          "precision" : columna_precision,
                          "recall" : columna_recall,
                          "cantidad_casos" : columna_cant_casos,
                          "f1":columna_f1}, 
                          columns = ["clase", "precision", "recall", "f1", "cantidad_casos"]
                        )        

    cantidad_casos_total = df_metricas_clase["cantidad_casos"].sum()
    df_metricas_clase["porcentaje_casos"] = round(df_metricas_clase["cantidad_casos"] * 100 / cantidad_casos_total, 2)
    df_metricas_clase.sort_values(["clase"], inplace = True)   
    return df_metricas_clase

#------------------------------------------------------------------------------------------------------------------------

def generar_df_para_grafico_comparacion_modelos(y_true, y_pred, clases):
    
    df_metricas = generar_df_metricas(y_true, y_pred, clases)
    columna_clase = []
    columna_tipo_score = []
    columna_valor_score = []
    for clase in clases:
        for tipo_score in ["precision", "recall"]:
            filtro_clase = df_metricas["clase"] == clase
            valor_score = round(df_metricas[filtro_clase][tipo_score].iloc[0] * 100, 1)
            columna_clase.append(clase)
            columna_tipo_score.append(tipo_score)
            columna_valor_score.append(valor_score)
    df_metricas_para_grafico = pd.DataFrame({"clase": columna_clase,
                                             "tipo_score": columna_tipo_score,
                                             "valor_score": columna_valor_score})
    return df_metricas_para_grafico

#------------------------------------------------------------------------------------------------------------------------

def graficar_comparacion_modelos(y_true_baseline, y_pred_baseline, y_true_modelo, y_pred_modelo, clases):

    df_metricas_baseline_para_grafico = generar_df_para_grafico_comparacion_modelos(y_true_baseline, y_pred_baseline, clases)
    df_metricas_modelo_para_grafico = generar_df_para_grafico_comparacion_modelos(y_true_modelo, y_pred_modelo, clases)

    labels = clases

    precision_modelo = df_metricas_modelo_para_grafico[df_metricas_modelo_para_grafico["tipo_score"] == "precision"].sort_values("clase")["valor_score"].values
    recall_modelo = df_metricas_modelo_para_grafico[df_metricas_modelo_para_grafico["tipo_score"] == "recall"].sort_values("clase")["valor_score"].values

    precision_baseline = df_metricas_baseline_para_grafico[df_metricas_baseline_para_grafico["tipo_score"] == "precision"].sort_values("clase")["valor_score"].values
    recall_baseline = df_metricas_baseline_para_grafico[df_metricas_baseline_para_grafico["tipo_score"] == "recall"].sort_values("clase")["valor_score"].values

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, precision_modelo, width, edgecolor = "black", label='Precision mejor modelo')
    rects2 = ax.bar(x + width/2, recall_modelo, width, edgecolor = "black", label='Recall mejor modelo')
    rects3 = ax.bar(x - width/2, precision_baseline, width, edgecolor = "black", 
                    color = "k", alpha = .5, label='Precision baseline')
    rects4 = ax.bar(x + width/2, recall_baseline, width, edgecolor = "black", 
                    color = "k", alpha = .5, label='Recall baseline ')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Clase de la variable target', fontsize=12)
    ax.set_ylabel(u'Valor de la métrica', fontsize=12)
    ax.set_title('Comparación Mejor modelo vs Baseline', fontsize=17)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    yticks = np.arange(0, 110, 10)
    plt.yticks(yticks, [str(int(item)) + " %" for item in yticks])


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}%'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    fig.tight_layout()

    plt.show()

#------------------------------------------------------------------------------------------------------------------------

def generar_df_feature_importances(modelo, lista_nombre_variables):
    
    importances = modelo.feature_importances_
    std = np.std([modelo.feature_importances_ for tree in modelo.estimators_], axis=0)
    
    df_feature_importances = pd.DataFrame({"feature" : lista_nombre_variables, 
                                      "importance_std" : std,
                                      "importance_percentage" : importances * 100})
    df_feature_importances.sort_values("importance_percentage", ascending = False, inplace = True)
    df_feature_importances["accumulated_importance_percentage"] = df_feature_importances["importance_percentage"].cumsum()
    
    return df_feature_importances

#------------------------------------------------------------------------------------------------------------------------

def graficar_feature_importances(df_feature_importances, cantidad_a_graficar):
    
    df_aux = df_feature_importances.head(cantidad_a_graficar)
    rectas = plt.barh(width = df_aux["importance_percentage"], y = df_aux["feature"], color = "slateblue", edgecolor = "k")
    
    plt.gca().invert_yaxis()
    xticks = np.arange(0, 110, 10)
    plt.xticks(xticks, [str(int(item)) + " %" for item in xticks])
    plt.title("Feature importances", fontsize=17)
    plt.xlabel("Importancia", fontsize=12)
    plt.ylabel(u"Variable", fontsize=12)
    autolabel_horizontal(rectas)
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------------------------------------------------

def autolabel_horizontal(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        width = rect.get_width()
        plt.annotate('{:.1f}%'.format(width),
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(50, 0),  # 3 points vertical offset
                    textcoords="offset points", fontsize = 12,
                    ha='right', va='center')

#------------------------------------------------------------------------------------------------------------------------

def graficar_var_booleana_vs_target(nombre_variable, nombre_target, df_dataset):
    
    df_aux = df_dataset.groupby([nombre_target])[nombre_variable].mean() * 100
    df_aux = pd.DataFrame(df_aux).reset_index()
    df_aux.rename(columns = {nombre_variable:'porcentaje'}, inplace = True)

    rectas = plt.barh(width = df_aux["porcentaje"], y = df_aux[nombre_target], 
                color = "slateblue", edgecolor = "k")

    yticks = df_aux[nombre_target]
    plt.yticks(yticks)

    #maximo_valor = min(int(max(df_aux["porcentaje"]) * 1.2), 110)
    xticks = np.arange(0, 110, 10)
    plt.xticks(xticks, [str(int(item)) + " %" for item in xticks])

    plt.xlabel("Porcentaje de publicaciones con " + str(nombre_variable) + " =1", fontsize=12)
    plt.ylabel(nombre_target, fontsize=12)
    plt.title(nombre_variable + " vs " + nombre_target, fontsize=17)

    autolabel_horizontal(rectas)

    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------------------------------------------------