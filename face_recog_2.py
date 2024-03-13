import cv2
import dlib
import numpy as np

# 加载dlib模型
predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 加载第一张照片（包含一个人脸）
img_path1 = 'data/1.jpg'
img1 = cv2.imread(img_path1)

# 加载第二张照片（可能包含多个人脸）
img_path2 = 'data/4.jpg'
img2 = cv2.imread(img_path2)

# 识别第一张照片中的人脸并提取特征
dets1 = detector(img1, 1)
if len(dets1) == 1:  # 假设第一张照片中只有一个人脸
    shape1 = predictor(img1, dets1[0])
    face_descriptor1 = facerec.compute_face_descriptor(img1, shape1)
else:
    print("Error: 第一张照片中不包含人脸或包含多个人脸")
    exit()

# 识别第二张照片中的人脸并比对
dets2 = detector(img2, 1)
for det in dets2:
    shape2 = predictor(img2, det)
    face_descriptor2 = facerec.compute_face_descriptor(img2, shape2)

    # 计算两个人脸特征向量的欧氏距离
    distance = np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2), ord=2)

    # 如果距离小于某个阈值，认为是同一个人
    threshold = 0.5  # 调整阈值以满足你的需求
    if distance < threshold:
        x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
        cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 标记通过的人脸
    else:
        x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
        cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 标记不通过的人脸

# 保存标记后的第二张照片
cv2.imshow('detect face', img2)
cv2.imwrite('result/output_image.jpg', img2)
