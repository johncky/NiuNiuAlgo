import os


supported_dtypes = ('K_DAY', 'K_1M', 'K_3M', 'K_5M', 'K_15M', 'QUOTE', 'ORDER_UPDATE')


SANIC_HOST = os.getenv('SANIC_HOST')
SANIC_PORT = os.getenv('SANIC_PORT')
FUTU_TRADE_PWD = os.getenv('FUTU_TRADE_PWD')
FUTU_HOST = os.getenv('FUTU_HOST')
FUTU_PORT = int(os.getenv('FUTU_PORT'))
ZMQ_PORT = os.getenv('ZMQ_PORT')
MYSQL_DB = os.getenv('MYSQL_DB')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PWD = os.getenv('MYSQL_PWD')