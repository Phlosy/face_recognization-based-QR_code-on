import cv2
import dlib
import numpy as np
import copy
import threading
import get_validate
import QRCode_reco

# 加载dlib模型
predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


img1 = get_validate.decode_Base64()
dets1 = detector(img1, 1)
shape1 = predictor(img1, dets1[0])
face_descriptor1 = facerec.compute_face_descriptor(img1, shape1)


# 创建一个全局变量，用于共享帧数据
frame1 = None
frame_lock = threading.Lock()

# 创建一个标志，用于控制线程退出
exit_flag = False


# 定义一个函数，作为第一个任务，实时显示摄像头拍摄到的画面
def display_camera():
    global frame1
    global exit_flag
    cap = cv2.VideoCapture(0)

    # 设置视频参数，propId 设置的视频参数，value 设置的参数值
    cap.set(3, 960)

    #计数器
    count = 0

    while cap.isOpened():
        flag, img_rd = cap.read()
        count += 1
        frame_interval = 10

        # 每帧数据延时 1ms，延时为 0 读取的是静态帧
        k = cv2.waitKey(1)

        # 取灰度
        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

        # 待会要写的字体
        font = cv2.FONT_HERSHEY_SIMPLEX

        if count % frame_interval == 0:
            frame1 = copy.deepcopy(img_rd)

        # 人脸数
        faces = detector(img_gray, 1)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == 27:  # 按下Esc键退出循环
            exit_flag = True
            break
        else:
            # 检测到人脸
            if len(faces) != 0:
                for face in faces:
                    # 绘制矩形框
                    cv2.rectangle(img_rd, tuple([face.left(), face.top()]), tuple([face.right(), face.bottom()]),
                              (0, 255, 255), 2)

                    height = face.bottom() - face.top()
                    width = face.right() - face.left()

                cv2.putText(img_rd, "Faces in all: " + str(len(faces)), (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

            else:
                # 没有检测到人脸
                cv2.putText(img_rd, "no face", (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

            # 添加说明
            img_rd = cv2.putText(img_rd, "Press 'Esc': Quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("camera", img_rd)

    # 释放摄像头
    cap.release()

    # 删除建立的窗口
    cv2.destroyAllWindows()

# 定义一个函数，作为第二个任务，进行人脸识别
def face_recognition():
    global frame1
    global exit_flag

    count = 0
    max_count = 0

    while not exit_flag:
        if frame1 is not None:
            if max_count >= 2:
                print("pass")
                max_count = 0
                count = 0
                #exit_flag = True
            else:
                print("unknown")

            #frame_copy = copy.deepcopy(frame1)
            dets = detector(frame1, 1)

            for det in dets:
                shape = predictor(frame1, det)
                face_descriptor2 = facerec.compute_face_descriptor(frame1, shape)

                # 计算两个人脸特征向量的欧氏距离
                distance = np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2), ord=2)

                # 如果距离小于某个阈值，认为是同一个人
                threshold = 0.4  # 调整阈值以满足你的需求
                if distance < threshold:
                    count = count + 1
                    if max_count < count:
                        max_count = count
                else:
                    if max_count < count:
                        max_count = count
                    count = 0
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

