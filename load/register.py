import requests
import base64


def register(host, user_list):
    for user in user_list:
        register_params = {
            "username": user,
            "password": "password",
            "email": "email"
        }
        resp = requests.post(host + "/register", json=register_params)
        print(resp.json())
        user_id = resp.json()['id']

        # base64string = base64.b64encode((user + ':password').encode("utf-8")).decode("utf-8")
        # resp = requests.get(host + "/login", headers={"Authorization": "Basic %s" % base64string})
        # print(resp.text)
        card_params = {
            "longNum": "5953580604169678",
            "expires": "05/05",
            "ccv": "123",
            "userID": user_id
        }
        address_params = {
            "street": "Baker Street",
            "number": "22",
            "country": "United Kingdom",
            "city": "London",
            "postcode": "G67 3DL",
            "userID": user_id
        }
        resp = requests.post(host + "/cards", json=card_params)
        print(resp.json())
        resp = requests.post(host + "/addresses", json=address_params)
        print(resp.json())


if __name__ == '__main__':
    host = "http://192.168.9.24:30080"
    user_id_start = 0
    user_id_end = 1
    user_list = []
    for i in range(user_id_start, user_id_end):
        user_list.append("user" + str(i))
    register(host, user_list)
