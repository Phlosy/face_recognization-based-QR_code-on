import cv2
import dlib
import numpy as np
import copy
import threading
import queue
from pyzbar.pyzbar import decode
import requests
import json
import base64
import time
import datetime
from io import BytesIO
from PIL import Image
# 测试用的库
import get_validate
def capture_frames(queue):
    global exit_flag
    camera = cv2.VideoCapture(0)  # 打开摄像头
    while not exit_flag:
        ret, frame = camera.read()  # 读取摄像头帧
        if cv2.waitKey(1) & 0xFF == 27:
            exit_flag = True  # 设置退出标志
            break
        if queue.qsize() >= max_queue_size:
            queue.get()  # 如果队列已满，移除最旧的一帧
        queue.put(frame)  # 将新帧放入队列
        cv2.imshow("camera", frame)  # 显示帧图片
    # 释放摄像头
    camera.release()
    # 删除建立的窗口
    cv2.destroyAllWindows()

# 检测图像中的码（解码）
def Read_Decode_Pic(image):
    global qrcode_flag, face_reco_flag
    # 遍历解码
    data = None
    for code in decode(image):
        data = code.data.decode('utf-8')
        #print("条形码/二维码数据：", data)  # 解码数据
        qrcode_flag = False
        face_reco_flag = True
        return data
# 检测视频中的码（解码）
def Read_Decode_Cam(queue):
    global exit_flag, qrcode_flag, face_reco_flag, QR_data
    while not exit_flag and qrcode_flag and not face_reco_flag:
        if cv2.waitKey(1) & 0xFF == 27:  # 按下Esc键退出循环
            exit_flag = True
            break
        image = queue.get()  # 获取每一帧图片
        if Read_Decode_Pic(image) is not None:
            QR_data = Read_Decode_Pic(image)  # 对每一帧图片检测

        #cv2.waitKey(1)  # 延时1ms

# 检测视频中的人脸
def find_face(queue,img1):
    global exit_flag, qrcode_flag, face_reco_flag, success_flag
    #计数器
    count = 0
    sec = 0
    #img1 = get_validate.decode_Base64()
    dets1 = detector(img1, 1)
    shape1 = predictor(img1, dets1[0])
    face_descriptor1 = facerec.compute_face_descriptor(img1, shape1)

    while not exit_flag and not qrcode_flag and face_reco_flag and not success_flag:
        img_rd = queue.get()
        count += 1
        frame_interval = 10

        # 每帧数据延时 1ms，延时为 0 读取的是静态帧
        #k = cv2.waitKey(1)

        # 取灰度
        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)


        face_reco_thread = threading.Thread(target=face_recognition, args=(img_rd,face_descriptor1,sec))
        if count % frame_interval == 0:
            faces = detector(img_gray, 1)
            if faces is not None and sec < 20:
                face_reco_thread.start()
                sec += 1
                face_reco_thread.join()
            elif sec >= 20:
                print("unknown")
                print("Please show QRcode")
                qrcode_flag = True
                face_reco_flag = False
                face_reco_thread.join()

        #cv2.imshow("camera", img_rd)  # 显示帧图片

        # 按下 'esc' 键退出
        if cv2.waitKey(1) & 0xFF == 27:  # 按下Esc键退出循环
            exit_flag = True
            break


def face_recognition(frame1,face_descriptor1,sec):
    global exit_flag, qrcode_flag, face_reco_flag, success_flag

    if not exit_flag and sec < 20:
        dets = detector(frame1, 1)

        for det in dets:
            shape = predictor(frame1, det)
            face_descriptor2 = facerec.compute_face_descriptor(frame1, shape)

            # 计算两个人脸特征向量的欧氏距离
            distance = np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2), ord=2)

            # 如果距离小于某个阈值，认为是同一个人
            threshold = 0.5  # 调整阈值以满足你的需求
            if distance < threshold:
                success_flag = True
                qrcode_flag = True
                face_reco_flag = False
                print("success")
                print("Welcome!")
                break

        if cv2.waitKey(1) & 0xFF == 27:  # 按下Esc键退出循环
            exit_flag = True

def post_validate_to_api(data):
    api_url = "http://120.46.140.85:8080/entrance/validate"
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


def decode_Base64(Base_64_data):

    #base64_image_str = encode_Base64()      #测试
    # 解码Base64数据
    decoded_image_data = base64.b64decode(Base_64_data)

    # 创建一个BytesIO对象，用于将二进制数据加载到PIL图像
    image_stream = BytesIO(decoded_image_data)

    # 使用PIL库打开图像
    pil_image = Image.open(image_stream)

    # 转换PIL图像为NumPy数组
    image_array = np.array(pil_image)

    # 使用OpenCV将NumPy数组转换为与cv2.imread相同格式的图像
    cv2_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    return cv2_image


if __name__ == "__main__":
    # 加载dlib模型
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    # 线程运行标识
    exit_flag = False   #全局
    qrcode_flag = True
    face_reco_flag = False
    success_flag = False
    # 开启摄像头，创建线程，获取画面，将图片放在队列中
    max_queue_size = 5  # 设置队列的大小
    frame_queue = queue.Queue(max_queue_size)
    capture_thread = threading.Thread(target=capture_frames, args=(frame_queue,))
    capture_thread.start()  # 启动捕获帧的线程

    QR_data = None          # 二维码信息

    while not exit_flag:
        success_flag = False
        QR_data = None
        exit_flag = False  # 全局
        qrcode_flag = True
        face_reco_flag = False

        # 识别有无二维码
        QRCode_thread = threading.Thread(target=Read_Decode_Cam, args=(frame_queue,))
        print("Please show your QRcode")
        QRCode_thread.start()
        QRCode_thread.join()
        if QR_data != None:
            print(QR_data)
            #print(type(QR_data))
            QR_data_json=json.loads(QR_data)

        # 获取当前时间
        current_time = datetime.datetime.now()
        # 将时间转换为数字格式
        time_as_number = str(current_time.strftime("%m%d%H%M%S"))
        # 生成QR_data
        #print(type(QR_data))

        QR_post_data = {
            "resident_id": time_as_number,
            "telephone": QR_data_json["telephone"],
            "invite_code": QR_data_json["invite_code"]
        }

        # 测试
        '''
        QR_post_data = {
            "resident_id": time_as_number,
            "telephone": "13700000000",
            "invite_code": "zxede/re341113212"
        }
        '''
        #print(type(QR_post_data))
        # 上传二维码信息
        validate_data = post_validate_to_api(QR_post_data)
        '''
        if validate_data["data"] is None:
            print("data is None")
            time.sleep(1)
            continue
        else:
            image = decode_Base64(["data"]["photo"]["imageStr"])
        '''
        print(validate_data)
        image = get_validate.decode_Base64()
        # 人脸对比
        if not exit_flag:
            face_find_thread = threading.Thread(target=find_face, args=(frame_queue, image,))
            face_find_thread.start()
            face_find_thread.join()
        if success_flag:
            permit_post_data = {
                "reservation_id": "1"
            }
            #print(permit_post_data)
            permit_data = post_permit_to_api(permit_post_data)
            #print(permit_data)
        if cv2.waitKey(1) & 0xFF == 27:  # 按下Esc键退出循环
            exit_flag = True
            break


    exit_flag = True  # 设置退出标志

