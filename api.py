import requests
import json
def post_data_to_api(api_url, data):
    try:
        response = requests.post(api_url, data=data)

        # 检查响应状态码，通常 200 表示成功
        if response.status_code == 200:
            # 使用 .json() 方法解析 JSON 数据
            json_data = response.json()

            return json_data
        else:
            print(f"请求失败，状态码: {response.status_code}")
            return None
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None

def post_permit_to_api(data):
    api_url = "http://120.46.140.85:8080/entrance/permit"
    try:
        response = requests.post(api_url, data=data)
        # 检查响应状态码，通常 200 表示成功
        if response.status_code == 200:
            # 使用 .json() 方法解析 JSON 数据
            json_data = response.json()
            return json_data
        else:
            print(f"请求失败，状态码: {response.status_code}")
            return None
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None


# 要发送的 JSON 数据
data_to_send = {
    "resident_id": "3",
    "telephone": "13700000011",
    "invite_code": "y59KYhcz3KKaFqmoC9HmTHvK2kPq1hjtGTVqEYxvqG5T75qrzxsz"
}

data = {
    "reservation_id": "1"
}

# 调用该方法并传递接口的 URL 和要发送的数据
api_url = "http://120.46.140.85:8080/calculator/add/5/3"  # 替换成实际的接口 URL

api_url1 = "http://120.46.140.85:8080/entrance/permit"  # 替换成实际的接口 URL
json_data = post_data_to_api(api_url1, data)
#print(json_data["data"])

data = json_data["data"]
#print(data)
if data == None:
    print("unknown")

if json_data:
    # 处理 JSON 数据
    print(type(json_data["message"]))
    print(json_data)
else:
    print("无法获取 JSON 数据")
