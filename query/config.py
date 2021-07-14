ES_HOSTS = "192.168.9.10:9201"
ES_INDEXES = ["sockshop-front-end", "sockshop-shipping", "sockshop-carts", "sockshop-user",
              "sockshop-orders", "sockshop-rabbitmq", "sockshop-user-db", "sockshop-orders-db",
              "sockshop-carts-db", "sockshop-queue-master", "sockshop-catalogue", "sockshop-catalogue-db",
              "sockshop-payment"]

PROM_URL = "http://192.168.9.24:30090"
PROM_METRICS_STEP = "5s"
SMOOTHING_WINDOW = 12
DATA_DIR = './data/'
METRIC_DIR = DATA_DIR + 'metrics/'
LOG_DIR = DATA_DIR + 'logs/'
