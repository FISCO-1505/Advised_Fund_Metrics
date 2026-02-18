import pandas as pd
from os import getenv as ENV_VARS
from os import path
import streamlit as st
from cryptography.fernet import Fernet, InvalidToken


def cargar_clave():
    """
    Carga la clave del archivo .key
    """
    key_string=st.secrets["KEY"]
    return key_string.encode()

def limpiar_y_convertir_base64(cadena_base64):
    """
    Añade el relleno necesario y convierte la cadena a bytes
    """
    if not isinstance(cadena_base64, str) or not cadena_base64.strip():
        return None
    
    cadena_base64 = cadena_base64.strip()
    longitud_invalida = len(cadena_base64) % 4
    if longitud_invalida != 0:
        cadena_base64 += '=' * (4 - longitud_invalida)
    
    return cadena_base64.encode('utf-8')

def desencriptar_con_manejo_errores(valor_encriptado_base64, fernet_instancia):
    """
    Intenta desencriptar y maneja el error InvalidToken.
    """
    if valor_encriptado_base64 is None:
        return None
    
    try:
        valor_bytes = limpiar_y_convertir_base64(valor_encriptado_base64)
        return fernet_instancia.decrypt(valor_bytes).decode('utf-8')
    except InvalidToken:
        return f"ERROR_TOKEN_INVALIDO"
    except Exception as e:
        return f"ERROR_INESPERADO: {e}"
    

@st.cache_data
def obtener_nombres_hojas(file):
    excel_file = pd.ExcelFile(file)
    return excel_file.sheet_names

# 2. Función para cargar los datos de una hoja específica (pesada)
@st.cache_data(show_spinner=False)
def cargar_datos_excel(file, sheet_name=None):
    # El parámetro 'file' se usa como llave para el caché
    return pd.read_excel(file, sheet_name=None)