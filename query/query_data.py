import requests
from elasticsearch import Elasticsearch
import query.config as config
from datetime import datetime
import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import Birch
from sklearn import preprocessing


def mkdir(path):
    if os.path.exists(path):
        return
    os.mkdir(path)


def query_range_prom_data(query, start, end, step=config.PROM_METRICS_STEP, instant=False):
    """
    query over a range of time or instant
    :param instant: <bool> whether to execute instant query
    :param query: <string> Prometheus expression query string.
    :param start: <rfc3339 | unix_timestamp> Start timestamp.
    :param end: <rfc3339 | unix_timestamp> End timestamp.
    :param step: <duration | float> Query resolution step width in duration format or float number of seconds.
    :return: list of query result
    """
    if instant:
        prom_url = config.PROM_URL + "/api/v1/query"
        params = {
            "query": query
        }
    else:
        prom_url = config.PROM_URL + "/api/v1/query_range"
        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": step
        }
    resp = requests.post(url=prom_url, data=params)
    return resp.json()


# 创建mpg
def generate_mpg_data(fault_name, data_dir):
    df = pd.DataFrame(columns=['source', 'destination'])
    query_arr = [["source_workload", "destination_workload", "service", "service",
                  "sum(istio_tcp_received_bytes_total) by (source_workload, destination_workload)"],
                 ["source_workload", "destination_workload", "service", "service",
                  "sum(istio_requests_total{destination_workload_namespace=\'sock-shop\'}) "
                  "by (source_workload, destination_workload)"],
                 ["pod", "instance", "container", "host",
                  "sum(container_cpu_usage_seconds_total{namespace=\"sock-shop\", "
                  "container!~\"POD|istio-proxy|\"}) by (instance, pod)"],
                 ["instance", "pod", "host", "container",
                  "sum(container_cpu_usage_seconds_total{namespace=\"sock-shop\", "
                  "container!~\"POD|istio-proxy|\"}) by (instance, pod)"],
                 ["kubernetes_pod_name", "source_workload", "container", "service",
                  "sum(istio_requests_total{destination_workload_namespace='sock-shop', reporter='source'})"
                  " by (kubernetes_pod_name, source_workload)"],
                 ["source_workload", "kubernetes_pod_name", "service", "container",
                  "sum(istio_requests_total{destination_workload_namespace='sock-shop', reporter='source'})"
                  " by (kubernetes_pod_name, source_workload)"],
                 ["kubernetes_pod_name", "destination_workload", "container", "service",
                  "sum(istio_requests_total{destination_workload_namespace='sock-shop', reporter='destination'})"
                  " by (kubernetes_pod_name, destination_workload)"],
                 ["destination_workload", "kubernetes_pod_name", "service", "container",
                  "sum(istio_requests_total{destination_workload_namespace='sock-shop', reporter='destination'})"
                  " by (kubernetes_pod_name, destination_workload)"],
                 ["kubernetes_pod_name", "source_workload", "container", "service",
                  "sum(istio_tcp_received_bytes_total{destination_workload_namespace='sock-shop', reporter='source'})"
                  " by (kubernetes_pod_name, source_workload)"],
                 ["source_workload", "kubernetes_pod_name", "service", "container",
                  "sum(istio_tcp_received_bytes_total{destination_workload_namespace='sock-shop', reporter='source'})"
                  " by (kubernetes_pod_name, source_workload)"],
                 ["kubernetes_pod_name", "destination_workload", "container", "service",
                  "sum(istio_tcp_received_bytes_total{destination_workload_namespace='sock-shop', "
                  "reporter='destination'}) by (kubernetes_pod_name, destination_workload)"],
                 ["destination_workload", "kubernetes_pod_name", "service", "container",
                  "sum(istio_tcp_received_bytes_total{destination_workload_namespace='sock-shop', "
                  "reporter='destination'}) by (kubernetes_pod_name, destination_workload)"],
                 ]
    for query_info in query_arr:
        results = query_range_prom_data(query_info[-1], None, None, instant=True)['data']['result']
        for result in results:
            metric = result['metric']
            source = metric[query_info[0]]
            destination = metric[query_info[1]]
            df = df.append({'source': source + '_' + query_info[2],
                            'destination': destination + '_' + query_info[3]}, ignore_index=True)
    df = df.drop_duplicates()
    df.to_csv(data_dir + fault_name + '_mpg.csv')


# 查询节点相关的指标
def query_node_metrics(start_ts, end_ts, metrics_dir):
    # 节点发包率，顺便获取节点名称
    # 去除了/1000
    net_query = "rate(node_network_transmit_packets_total{device=\"eth0\", job=\"kubernetes-service-endpoints\"}[1m])"
    net_data_arr = query_range_prom_data(net_query, start_ts, end_ts)['data']['result']
    for net_data in net_data_arr:
        instance = net_data['metric']['instance']
        node_name = net_data['metric']['kubernetes_node']
        data_dir = metrics_dir + node_name + '_host/'
        mkdir(data_dir)
        dump_metric_data(net_data['values'], data_dir + node_name + '_host_net.csv')

        # 节点cpu使用率
        query = "sum(rate(node_cpu_seconds_total{mode != \"idle\",  mode!= \"iowait\", mode!~\"^(?:guest.*)$\", " \
                "instance=\"%s\", job=\"kubernetes-service-endpoints\" }[1m])) / " \
                "count(node_cpu_seconds_total{mode=\"system\", instance=\"%s\"," \
                " job=\"kubernetes-service-endpoints\"})" % (instance, instance)
        data = query_range_prom_data(query, start_ts, end_ts)['data']['result'][0]
        dump_metric_data(data['values'], data_dir + node_name + '_host_cpu.csv')
        # 节点内存使用率
        query = "1 - sum(node_memory_MemAvailable_bytes{instance=\"%s\", job=\"kubernetes-service-endpoints\"}) / " \
                "sum(node_memory_MemTotal_bytes{instance=\"%s\", job=\"kubernetes-service-endpoints\"})" % (
                    instance, instance)
        data = query_range_prom_data(query, start_ts, end_ts)['data']['result'][0]
        dump_metric_data(data['values'], data_dir + node_name + '_host_mem.csv')


# 查询运行的pod相关的指标
def query_pod_metrics(start_ts, end_ts, metrics_dir):
    # pod cpu使用率
    query = "sum(rate(container_cpu_usage_seconds_total{namespace=\"sock-shop\", container!~\"POD|istio-proxy|\"}[1m]" \
            ")) by (pod, container)"
    cpu_data_arr = query_range_prom_data(query, start_ts, end_ts)['data']['result']
    for cpu_data in cpu_data_arr:
        svc = cpu_data['metric']['container']
        pod_name = cpu_data['metric']['pod']
        data_dir = metrics_dir + pod_name + '_container/'
        mkdir(data_dir)
        dump_metric_data(cpu_data['values'], data_dir + pod_name + '_container_cpu.csv')
        # pod 内存使用率
        query = "sum(rate(container_memory_working_set_bytes{namespace=\"sock-shop\", pod=\"%s\"}[1m]))/1000" % pod_name
        data = query_range_prom_data(query, start_ts, end_ts)['data']['result'][0]
        dump_metric_data(data['values'], data_dir + pod_name + '_container_mem.csv')
        # pod网络传输率的平方
        query = "sum(rate(container_network_transmit_packets_total{namespace=\"sock-shop\", pod=\"%s\"}[1m])) / 1000 " \
                "* sum(rate(container_network_transmit_packets_total{namespace=\"sock-shop\", pod=\"%s\"}[1m])) " \
                "/ 1000" % (pod_name, pod_name)
        data = query_range_prom_data(query, start_ts, end_ts)['data']['result'][0]
        dump_metric_data(data['values'], data_dir + pod_name + '_container_net.csv')


# 查询服务相关的指标
def query_service_metrics(start_ts, end_ts, fault_name, metrics_dir):
    data_dir = metrics_dir + 'services/'
    mkdir(data_dir)
    # 50%时延
    # source
    source_latency_df = pd.DataFrame()
    query = "histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"source\", " \
            "destination_workload_namespace=\"sock-shop\"}[1m])) by (destination_workload, source_workload, le)) / 1000"
    results = query_range_prom_data(query, start_ts, end_ts)['data']['result']
    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['values']

        values = list(zip(*values))
        if 'timestamp' not in source_latency_df:
            timestamp = values[0]
            source_latency_df['timestamp'] = timestamp
            source_latency_df['timestamp'] = source_latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        source_latency_df[name] = pd.Series(metric)
        source_latency_df[name] = source_latency_df[name].astype('float64') * 1000
    query = "sum(irate(istio_tcp_sent_bytes_total{reporter=\"source\"}[1m])) " \
            "by (destination_workload, source_workload) / 1000"
    results = query_range_prom_data(query, start_ts, end_ts)['data']['result']
    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['values']

        values = list(zip(*values))
        if 'timestamp' not in source_latency_df:
            timestamp = values[0]
            source_latency_df['timestamp'] = timestamp
            source_latency_df['timestamp'] = source_latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        source_latency_df[name] = pd.Series(metric)
        source_latency_df[name] = source_latency_df[name].astype('float64'). \
            rolling(window=config.SMOOTHING_WINDOW, min_periods=1).mean()
    filename = data_dir + fault_name + '_latency_source_50.csv'
    source_latency_df = source_latency_df.set_index(['timestamp'])
    source_latency_df.to_csv(filename)

    # destination
    dest_latency_df = pd.DataFrame()
    query = "histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", " \
            "destination_workload_namespace=\"sock-shop\"}[1m])) by (destination_workload, source_workload, le)) / 1000"
    results = query_range_prom_data(query, start_ts, end_ts)['data']['result']
    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['values']

        values = list(zip(*values))
        if 'timestamp' not in dest_latency_df:
            timestamp = values[0]
            dest_latency_df['timestamp'] = timestamp
            dest_latency_df['timestamp'] = dest_latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        dest_latency_df[name] = pd.Series(metric)
        dest_latency_df[name] = dest_latency_df[name].astype('float64') * 1000

    query = "sum(irate(istio_tcp_sent_bytes_total{reporter=\"destination\"}[1m])) " \
            "by (destination_workload, source_workload) / 1000"
    results = query_range_prom_data(query, start_ts, end_ts)['data']['result']
    for result in results:
        dest_svc = result['metric']['destination_workload']
        src_svc = result['metric']['source_workload']
        name = src_svc + '_' + dest_svc
        values = result['values']

        values = list(zip(*values))
        if 'timestamp' not in dest_latency_df:
            timestamp = values[0]
            dest_latency_df['timestamp'] = timestamp
            dest_latency_df['timestamp'] = dest_latency_df['timestamp'].astype('datetime64[s]')
        metric = values[1]
        dest_latency_df[name] = pd.Series(metric)
        dest_latency_df[name] = dest_latency_df[name].astype('float64'). \
            rolling(window=config.SMOOTHING_WINDOW, min_periods=1).mean()
    filename = data_dir + fault_name + '_latency_destination_50.csv'
    dest_latency_df = dest_latency_df.set_index(['timestamp'])
    dest_latency_df.to_csv(filename)
    return source_latency_df, dest_latency_df


# 指标查询结果转为dataframe,保存到csv中
def dump_metric_data(values, filename):
    values = list(zip(*values))
    df = pd.DataFrame()
    df['timestamp'] = values[0]
    df['timestamp'] = df['timestamp'].astype('datetime64[s]')
    df['value'] = pd.Series(values[1])
    df['value'] = df['value'].astype('float64')
    df.set_index('timestamp')
    df.to_csv(filename)


# 时间戳转零时区格式字符串
def ts2date(ts):
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S.000Z')


# 查询es
def es_query(es, index, query):
    scroll_size = 10000
    data = es.search(body=query, index=index, params={"scroll": "3m", "size": scroll_size})
    result = data['hits']['hits']
    total = data['hits']['total']['value']
    scroll_id = data['_scroll_id']

    for i in range(0, int(total / scroll_size) + 1):
        query_scroll = es.scroll(scroll_id=scroll_id, params={"scroll": "3m"})['hits']['hits']
        result += query_scroll

    return result, total


# 查询服务的日志
def query_service_logs(start_ts, end_ts, indexes, data_dir):
    log_data_dict = {}
    es = Elasticsearch(hosts=config.ES_HOSTS)
    range_query = {
        "query": {
            "range": {
                "@timestamp": {
                    "gte": ts2date(start_ts),
                    "lte": ts2date(end_ts)
                }
            }
        },
        "sort": [
            {
                "@timestamp": {
                    "order": "asc"
                }
            }
        ]
    }
    output_dir = data_dir + 'logs/'
    mkdir(output_dir)
    for index in indexes:
        log_data, log_size = es_query(es, index + '-2021*', range_query)
        # log_data_dict[index] = log_data
        logs = [d['_source']['message'] + '\n' for d in log_data]
        raw_data_name = output_dir + '/raw_' + index + '.json'
        log_data_name = output_dir + '/' + index + '.log'
        with open(log_data_name, 'w', encoding='utf-8') as f:
            f.writelines(logs)
        with open(raw_data_name, 'w', encoding='utf-8') as f:
            json.dump(log_data, f)
        print("finish fetching log for {}, size: {}".format(index, log_size))
    # return log_data_dict


# Anomaly Detection
def birch_ad_with_smoothing(latency_df, threshold):
    # anomaly detection on response time of service invocation.
    # input: response times of service invocations, threshold for birch clustering
    # output: anomalous service invocation

    anomalies = []
    for svc, latency in latency_df.iteritems():
        # No anomaly detection in db
        if svc != 'timestamp' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
            latency = latency.rolling(window=config.SMOOTHING_WINDOW, min_periods=1).mean()
            x = np.array(latency)
            x = np.where(np.isnan(x), 0, x)
            normalized_x = preprocessing.normalize([x])

            X = normalized_x.reshape(-1, 1)

            #            threshold = 0.05

            brc = Birch(branching_factor=50, n_clusters=None, threshold=threshold, compute_labels=True)
            brc.fit(X)
            brc.predict(X)

            labels = brc.labels_
            #            centroids = brc.subcluster_centers_
            n_clusters = np.unique(labels).size
            if n_clusters > 1:
                anomalies.append(svc)
    return anomalies


if __name__ == '__main__':
    mkdir(config.DATA_DIR)
    # mkdir(config.METRIC_DIR)
    # mkdir(config.LOG_DIR)
    # test_query = "sum by (mode)(rate(node_cpu_seconds_total{mode=\"idle\",instance=\"10.10.64.163:9100\",job=\"node-exporter\"}[20s])) * 100"
    # 5/18 向服务注入cpu故障
    # fault_info = {
    #     "catalogue": [1621302300, 1621302600, 1621302900, 1621303200, 1621303500],
    #     "carts": [1621304100, 1621304400, 1621304700, 1621305000, 1621305300],
    #     "orders": [1621305600, 1621305900, 1621306200, 1621306500, 1621306800],
    #     "user": [1621307400, 1621308000, 1621308300, 1621308600, 1621308900]
    # }
    # 5/18 向服务注入内存故障
    # fault_info = {
    #     "catalogue": [1621346400, 1621346700, 1621347000, 1621347300, 1621347600],
    #     "carts": [1621342500, 1621342800, 1621343100, 1621343400, 1621343700],
    #     "orders": [1621341000, 1621341300, 1621341600, 1621341900, 1621342200],
    #     "user": [1621344000, 1621345200, 1621345500, 1621345800, 1621346100]
    # }
    # 5/19 向catalogue,user加大内存压力注入故障，
    # fault_info = {
    #     "catalogue": [1621409055, 1621410420, 1621410720, 1621411205, 1621411505],
    #     "user": [1621410050, 1621411800, 1621412100, 1621412400, 1621412700]
    # }
    # 5/20 向orders注入一次cpu故障，取共15分钟的数据
    # fault_info = {
    #     "orders": [1621477110]
    # }
    # 5/20 向其他服务注入cpu故障
    # fault_info = {
    #     "front-end": [1621509780, 1621510080, 1621510380, 1621510680, 1621510980],
    #     "payment": [1621511280, 1621511580, 1621511880, 1621512180, 1621512480]
    # }
    fault_info = {
        "carts": [1623080250]
    }
    for key in fault_info.keys():
        fault_times = fault_info[key]
        print(key)
        for fault_time in fault_times:
            exp_start = fault_time - 60 * 2
            exp_end = fault_time + 60 * 3
            fault_name = "cpu"
            ad_threshold = 0.045
            mkdir(config.DATA_DIR + key)
            data_dir = config.DATA_DIR + key + "/" + fault_name + "_" + str(exp_start) + "_" + str(exp_end) + "/"

            metrics_dir = data_dir + "metrics/"
            mkdir(data_dir)
            mkdir(metrics_dir)
            generate_mpg_data(fault_name, data_dir)
            query_node_metrics(exp_start, exp_end, metrics_dir)
            query_pod_metrics(exp_start, exp_end, metrics_dir)
            source_df, dest_df = query_service_metrics(exp_start, exp_end, fault_name, metrics_dir)
            latency_df = dest_df.add(source_df, fill_value=0)
            anomalies = birch_ad_with_smoothing(latency_df, ad_threshold)
            print(anomalies)
            query_service_logs(exp_start, exp_end, config.ES_INDEXES, data_dir)
    # test_step = 10
    # print(test_query)
    # query_range_prom_data(test_query, test_start, test_end, test_step)

    # print(log_data_dict.keys())
