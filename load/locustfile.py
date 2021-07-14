from locust import task, between
from locust.contrib.fasthttp import FastHttpUser
import base64
from random import choice

user_list = ['user' + str(i) for i in range(1000)]


class MyUser(FastHttpUser):
    wait_time = between(0.001, 0.003)

    @task
    def index(self):
        catalogue = self.client.get("/catalogue").json()
        if len(catalogue) > 0:
            category_item = choice(catalogue)
            item_id = category_item["id"]

            self.client.get("/detail.html?id={}".format(item_id))
            self.client.delete("/cart")
            self.client.post("/cart", json={"id": item_id, "quantity": 1})
            self.client.get("/basket.html")
            self.client.post("/orders")

    def on_start(self):
        if len(user_list) > 0:
            username = choice(user_list)
            user_list.remove(username)
            data_bytes = (username + ':password').encode("utf-8")
            base64string = base64.b64encode(data_bytes).decode("utf-8")
            self.client.get("/login", headers={"Authorization": "Basic %s" % base64string})
