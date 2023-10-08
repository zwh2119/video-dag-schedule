import time

import routeros_api


def router_test(password, ip):
    connection = routeros_api.RouterOsApiPool(
        ip,
        username='admin',
        password=password,
        port=8728,
        plaintext_login=True
    )
    api = connection.get_api()
    list_queues = api.get_resource('/queue/tree')
    list_queues.set(id='*1000000', max_limit='10K')
    print(list_queues.get(name='cloud-queue'))

    connection.disconnect()


def router_limit_control(ip, password, bandwidths, last_time):
    connection = routeros_api.RouterOsApiPool(
        ip,
        username='admin',
        password=password,
        port=8728,
        plaintext_login=True
    )
    api = connection.get_api()
    list_queues = api.get_resource('/queue/tree')

    for bandwidth in bandwidths:
        list_queues.set(id='*1000000', max_limit=bandwidth)
        print(f'set max bandwidth limit of {bandwidth}bps..')
        time.sleep(last_time)

    connection.disconnect()


if __name__ == '__main__':
    band_list = ['200M', '100M', '10M', '1M', '100K', '800K', '3M', '50M', '100M', '500M']
    router_limit_control('dorm.ifan.dev', 'dislab_router2023', band_list, 10)
