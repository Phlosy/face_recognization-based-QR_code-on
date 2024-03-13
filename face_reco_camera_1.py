import cv2
import dlib
import numpy as np
import copy
import threading

# 加载dlib模型
predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 加载第一张照片（包含一个人脸）
img_path1 = 'data/3.jpg'
img1 = cv2.imread(img_path1)
dets1 = detector(img1, 1)
if len(dets1) == 1:  # 第一张照片中只有一个人脸
    shape1 = predictor(img1, dets1[0])
    face_descriptor1 = facerec.compute_face_descriptor(img1, shape1)
else:
    print("Error: 第一张照片中不包含人脸或包含多个人脸")
    exit()

# 创建一个全局变量，用于共享帧数据
frame1 = None
frame_lock = threading.Lock()

# 创建一个标志，用于控制线程退出
exit_flag = False


# 定义一个函数，作为第一个任务，实时显示摄像头拍摄到的画面
def display_camera():
    global frame1
    global exit_flag

    frame_count = 0
    frame_interval = 15  # 每隔frame_interval帧进行分析

    # 打开计算机摄像头
    cap = cv2.VideoCapture(0)

    while not exit_flag:
        ret, frame = cap.read()
        frame_count += 1
        with frame_lock:
            if frame_count % frame_interval == 0:
                frame1 = copy.deepcopy(frame)

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # 按下Esc键退出循环
            exit_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()

# 定义一个函数，作为第二个任务，进行人脸识别
def face_recognition():
    global frame1
    global exit_flag

    count = 0
    max_count = 0

    while not exit_flag:
        if frame1 is not None:
            if max_count >= 1:
                print("pass")
                max_count = 0
                count = 0
                exit_flag = True
            else:
                print("unknown")

            frame_copy = copy.deepcopy(frame1)
            dets = detector(frame_copy, 1)

            for det in dets:
                shape = predictor(frame_copy, det)
                face_descriptor2 = facerec.compute_face_descriptor(frame_copy, shape)

                # 计算两个人脸特征向量的欧氏距离
                distance = np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2), ord=2)

                # 如果距离小于某个阈值，认为是同一个人
                threshold = 0.4  # 调整阈值以满足你的需求
                if distance < threshold:
                    #x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
                    #cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 标记通过的人脸
                    count = count + 1
                    if max_count < count:
                        max_count = count
                else:
                    #x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
                    #cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 标记不通过的人脸
                    if max_count < count:
                        max_count = count
                    count = 0

            #cv2.imshow('Camera', frame_copy)
            frame1 = None

        if cv2.waitKey(1) & 0xFF == 27:  # 按下Esc键退出循环
            exit_flag = True
            break

# 创建两个线程并启动它们
display_thread = threading.Thread(target=display_camera)
recognition_thread = threading.Thread(target=face_recognition)

display_thread.start()
recognition_thread.start()

# 主线程等待两个线程结束
display_thread.join()
recognition_thread.join()

# 设置退出标志，以便两个线程可以退出
exit_flag = True
