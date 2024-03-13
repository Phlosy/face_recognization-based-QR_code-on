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
dets1 = detector(img1, 1)
if len(dets1) == 1:  # 假设第一张照片中只有一个人脸
    shape1 = predictor(img1, dets1[0])
    face_descriptor1 = facerec.compute_face_descriptor(img1, shape1)
else:
    print("Error: 第一张照片中不包含人脸或包含多个人脸")
    exit()

# 视频输入和输出文件路径
video_input_path = 'data/video/1.mp4'
video_output_path = 'result/output_video.mp4'

# 打开视频文件
cap = cv2.VideoCapture(video_input_path)

# 获取输入视频的帧参数
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(5))

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编解码器
out = cv2.VideoWriter(video_output_path, fourcc, frame_fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 识别视频帧中的人脸
    dets = detector(frame, 1)
    for det in dets:
        shape = predictor(frame, det)
        face_descriptor2 = facerec.compute_face_descriptor(frame, shape)

        # 计算两个人脸特征向量的欧氏距离
        distance = np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2), ord=2)

        # 如果距离小于某个阈值，认为是同一个人
        threshold = 0.5  # 调整阈值以满足你的需求
        if distance < threshold:
            x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 标记通过的人脸
        else:
            x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 标记不通过的人脸

    # 将帧写入输出视频
    out.write(frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
