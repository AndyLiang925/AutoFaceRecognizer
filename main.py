import face_recognition
import cv2
import numpy as np
import ctypes
from pynput import keyboard, mouse

# 获取摄像头
video_capture = cv2.VideoCapture(0)

# 加载图片并获取人脸编码
a_image = face_recognition.load_image_file("")
a_face_encoding = face_recognition.face_encodings(a_image)[0]

b_image = face_recognition.load_image_file("")
b_face_encoding = face_recognition.face_encodings(b_image)[0]

c_image = face_recognition.load_image_file("")
c_face_encoding = face_recognition.face_encodings(c_image)[0]

known_face_encodings = [b_face_encoding,  c_face_encoding, a_face_encoding]
known_face_names = ["b",  "c", "a"]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# 标记是否有鼠标或键盘事件触发
event_trigger = False

# 定义全局退出标志
exit_program = False

# --- 定义 pynput 事件回调函数 ---

# 鼠标事件：移动、点击或滚动均视为触发事件
def on_move(x, y):
    global event_trigger
    event_trigger = True

def on_click(x, y, button, pressed):
    global event_trigger
    if pressed:
        event_trigger = True

def on_scroll(x, y, dx, dy):
    global event_trigger
    event_trigger = True

# 键盘事件：任意按键均视为触发事件，同时检测退出键（例如 'q'）
def on_press(key):
    global event_trigger, exit_program
    event_trigger = True
    try:
        if key.char == 'q':  # 按下 q 键退出程序
            exit_program = True
            return False  # 停止键盘监听
    except AttributeError:
        # 针对特殊键不作处理
        pass

# --- 启动 pynput 监听器 ---
mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
keyboard_listener = keyboard.Listener(on_press=on_press)
mouse_listener.start()
keyboard_listener.start()

# 主循环：读取摄像头并进行人脸检测
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # 每隔一帧进行人脸检测并绘制结果
    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # 转换 BGR -> RGB
        rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
    process_this_frame = not process_this_frame

    # 如果检测到鼠标或键盘事件，则重新检测当前帧中是否出现“lhx”
    if event_trigger:
        # 对当前帧进行事件时的人脸检测
        small_frame_event = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame_event = small_frame_event[:, :, ::-1]
        rgb_small_frame_event = cv2.cvtColor(rgb_small_frame_event, cv2.COLOR_BGR2RGB)
        face_locations_event = face_recognition.face_locations(rgb_small_frame_event)
        face_encodings_event = face_recognition.face_encodings(rgb_small_frame_event, face_locations_event)
        face_names_event = []
        for face_encoding in face_encodings_event:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names_event.append(name)
        print("事件检测结果：", face_names_event)
        # 如果检测结果中不包含“lhx”，则锁定工作站
        if "lhx" not in face_names_event:
            print("警告：未检测到 lhx，锁定屏幕！")
            ctypes.windll.user32.LockWorkStation()
        event_trigger = False  # 重置事件标志

    # 绘制人脸检测结果
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # 恢复缩放比例
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    # 检测退出：如果 exit_program 被置为 True 或者窗口被关闭则退出循环
    if exit_program or cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源并关闭窗口
video_capture.release()
cv2.destroyAllWindows()

# 停止 pynput 监听器
mouse_listener.stop()
keyboard_listener.stop()
