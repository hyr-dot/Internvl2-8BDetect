import requests
import json
import datetime


def request_demo(): 
    url = f'http://10.26.2.240:8868/video_test'
    start_time = datetime.datetime.now()

    data = {
        "filepath": "../examples/Crafter_1.mp4"
    }
        
    res = requests.post(url, data=json.dumps(data))
    print(json.loads(res.text))

    data = {
        "filepath": "../examples/Crafter_12_frame14.jpg"
    }
        
    res = requests.post(url, data=json.dumps(data))
    print(json.loads(res.text))

    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds()
    print("Execution time is {:.4f}s".format(execution_time))

if __name__ == '__main__':
    request_demo()

