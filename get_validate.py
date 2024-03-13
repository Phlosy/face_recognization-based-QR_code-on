import json
import base64
from io import BytesIO
import cv2
import numpy as np
from PIL import Image


def get_json():
    global validate_data
    json_data = '''
    {
        "code": 20000,
        "data": {
            "reservation_id": 2,
            "photo": {
                "id": 2,
                "imageStr": "/9j/4A..."
            }
        },
        "message": "success"
    }
    '''

    # 将JSON字符串解析为Python对象
    validate_data = json.loads(json_data)
    print(type(validate_data))
    data = validate_data["data"]
    print(data)
    if data == None:
        print("unknown")
    # 提取变量
    '''
    code = validate_data["code"]
    reservation_id = validate_data["data"]["reservation_id"]
    photo_id = validate_data["data"]["photo"]["id"]
    image_str = validate_data["data"]["photo"]["imageStr"]
    message = validate_data["message"]
    '''

def get_Base64():
    return validate_data["data"]["photo"]["imageStr"]

def get_reservation_id():
    return validate_data["data"]["reservation_id"]

def decode_Base64():
    # Base64编码的图像数据
    global base64_image_str

    # 获取Base64编码
    #base64_image_str=get_Base64()
    base64_image_str = encode_Base64()      #测试
    # 解码Base64数据
    decoded_image_data = base64.b64decode(base64_image_str)

    # 创建一个BytesIO对象，用于将二进制数据加载到PIL图像
    image_stream = BytesIO(decoded_image_data)

    # 使用PIL库打开图像
    pil_image = Image.open(image_stream)

    # 转换PIL图像为NumPy数组
    image_array = np.array(pil_image)

    # 使用OpenCV将NumPy数组转换为与cv2.imread相同格式的图像
    cv2_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    return cv2_image

def encode_Base64():
    global base64_image_str
    # 打开JPEG图片文件
    with open("data/3.jpg", "rb") as image_file:
        # 读取图片文件的二进制数据
        image_binary = image_file.read()

    # 将二进制数据编码为Base64字符串
    base64_image_str = base64.b64encode(image_binary).decode("utf-8")

    return base64_image_str

if __name__ == '__main__':
    encode_Base64()
    decode_Base64()