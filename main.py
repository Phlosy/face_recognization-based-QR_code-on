import cv2
import threading
import queue

exit_flag = False
def capture_frames(queue):
    global exit_flag
    camera = cv2.VideoCapture(0)  # 打开摄像头
    while not exit_flag:
        ret, frame = camera.read()  # 读取摄像头帧
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True  # 设置退出标志
            break
        if queue.qsize() >= max_queue_size:
            queue.get()  # 如果队列已满，移除最旧的一帧
        queue.put(frame)  # 将新帧放入队列
    camera.release()

def display_frames(queue):
    global exit_flag
    while not exit_flag:
        frame = queue.get()  # 从队列中获取帧图片
        cv2.imshow("Camera Feed", frame)  # 显示帧图片
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True  # 设置退出标志
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    max_queue_size = 20  # 设置队列的最大大小
    frame_queue = queue.Queue(max_queue_size)

    # 创建两个线程，一个用于捕获帧，另一个用于显示帧
    capture_thread = threading.Thread(target=capture_frames, args=(frame_queue,))
    capture_thread.start()  # 启动捕获帧的线程

    display_thread = threading.Thread(target=display_frames, args=(frame_queue,))


    display_thread.start()  # 启动显示帧的线程

    capture_thread.join()  # 等待捕获线程结束
    #display_thread.join()  # 等待显示线程结束

    exit_flag = True  # 设置退出标志