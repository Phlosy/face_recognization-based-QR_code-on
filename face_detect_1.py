import cv2
import dlib
import numpy as np
import copy

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

# 创建 cv2 摄像头对象
cap = cv2.VideoCapture(0)

# 设置视频参数，propId 设置的视频参数，value 设置的参数值
cap.set(3, 960)

# 截图 screenshots 的计数器
count = 0

while cap.isOpened():
    flag, img_rd = cap.read()
    count += 1

    # 每帧数据延时 1ms，延时为 0 读取的是静态帧
    k = cv2.waitKey(1)

    # 取灰度
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    if count % 10 ==0:
        frame1 = copy.deepcopy(img_rd)

    # 人脸数
    faces = detector(img_gray, 1)

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == 27:  # 按下Esc键退出循环
        break
    else:
        # 检测到人脸
        if len(faces) != 0:
            for det in faces:
                shape = predictor(img_rd, det)
                face_descriptor2 = facerec.compute_face_descriptor(img_rd, shape)

                # 计算两个人脸特征向量的欧氏距离
                distance = np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2), ord=2)

                # 如果距离小于某个阈值，认为是同一个人
                threshold = 0.5  # 调整阈值以满足你的需求
                if distance < threshold:
                    x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
                    cv2.rectangle(img_rd, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 标记通过的人脸
                else:
                    x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
                    cv2.rectangle(img_rd, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 标记不通过的人脸i

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