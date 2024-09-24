import cv2 
import mediapipe as mp
import pyautogui
import collections
import math
import time
import threading
import tkinter as tk
from tkinter import messagebox
import queue
import dlib
import numpy as np
import subprocess

class Notification:
    def __init__(self, root, message):
        self.notification = tk.Toplevel(root)
        self.notification.overrideredirect(True)  # Remove window border and title bar
        self.notification.attributes("-topmost", True)  # Always on top of other windows

        # Get screen width and set position at top right corner
        screen_width = self.notification.winfo_screenwidth()
        self.notification.geometry(f"300x100+{screen_width - 320}+50")

        label = tk.Label(self.notification, text=message, font=("Arial", 24, "bold"), padx=20, pady=10)
        label.pack()

        # Close notification after 2 seconds
        self.notification.after(2000, self.close)

    def close(self):
        self.notification.destroy()

class GestureRecognitionApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.drawing = mp.solutions.drawing_utils
        self.hands = mp.solutions.hands
        self.hand_obj = self.hands.Hands(max_num_hands=1)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1)
        self.gesture_history = collections.deque(maxlen=30)
        self.start_init = False
        self.prev = -1
        self.pinch_active = False
        self.pinch_start = False
        self.pinch_end = False
        self.cooldown = False
        self.cooldown_start_time = 0
        self.cooldown_duration = 1.0
        self.running = False
        self.frame_queue = queue.Queue()
        self.head_orientation = 0
        self.front_orientation = 0
        self.is_front = True
        self.front_set = False
        
    def calculate_disstance(self, point1, point2):
        return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)**0.5

    def is_finger_extended(self, lst, tip_idx, pip_idx, threshold=0.02):
        return (lst.landmark[pip_idx].y - lst.landmark[tip_idx].y) > threshold

    def are_fingers_touching(self, lst, finger1_tip_idx, finger2_tip_idx, threshold=0.05):
        return self.calculate_disstance(lst.landmark[finger1_tip_idx], lst.landmark[finger2_tip_idx]) < threshold

    def check_four_fingers_extended_and_attached(self,lst):
        extended = [
            self.is_finger_extended(lst, 8, 6),
            self.is_finger_extended(lst, 12, 10),
            self.is_finger_extended(lst, 16, 14),
            self.is_finger_extended(lst, 20, 18)  
        ]

        attached = [
            self.are_fingers_touching(lst, 8, 12),
            self.are_fingers_touching(lst, 12, 16),
            self.are_fingers_touching(lst, 16, 20)
        ]

        return all(extended) and all(attached)

    def check_three_fingers_extended_and_attached(self, lst):
        extended = [
            self.is_finger_extended(lst, 8, 6),   # Index finger
            self.is_finger_extended(lst, 12, 10), # Middle finger
            self.is_finger_extended(lst, 16, 14)  # Ring finger
        ]

        attached = [
            self.are_fingers_touching(lst, 8, 12),  # Index to Middle
            self.are_fingers_touching(lst, 12, 16)  # Middle to Ring
        ]

        return all(extended) and all(attached)


    def count_fingers(self, lst):
        # cnt = 0
        extended = [False] * 5
        thresh = (lst.landmark[0].y*100 - lst.landmark[9].y*100)/2

        # if (lst.landmark[5].y*100 - lst.landmark[8].y*100) > thresh:
        #     cnt += 1

        # if (lst.landmark[9].y*100 - lst.landmark[12].y*100) > thresh:
        #     cnt += 1

        # if (lst.landmark[13].y*100 - lst.landmark[16].y*100) > thresh:
        #     cnt += 1

        # if (lst.landmark[17].y*100 - lst.landmark[20].y*100) > thresh:
        #     cnt += 1

        # if (lst.landmark[5].x*100 - lst.landmark[4].x*100) > 6:
        #     cnt += 1

        extended[0] = (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh
        extended[1] = (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh
        extended[2] = (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh
        extended[3] = (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh
        extended[4] = (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6

        cnt = sum(extended)
        return cnt, extended
        # return cnt 

    def detect_swipe(self, gesture_history):
        if len(gesture_history) < 10:
            return None, 0
        
        start_point = gesture_history[0]
        end_point = gesture_history[-1]

        delta_x = end_point.landmark[0].x - start_point.landmark[0].x
        delta_y = end_point.landmark[0].y - start_point.landmark[0].y

        scroll_amount = int(delta_y * -100)

        if abs(delta_x) > abs(delta_y):
            if delta_x > 0.1:
                return 'right_swipe', scroll_amount
            elif delta_x < -0.1:
                return 'left_swipe', scroll_amount
        else:
            if delta_y > 0.1:
                return 'down_swipe', scroll_amount
            elif delta_y < -0.1:
                return 'up_swipe', scroll_amount
        
        return None, 0

    def fingers_attached(self, lst, finger1_idx, finger2_idx, threshold=0.05):
        point1 = lst.landmark[finger1_idx]
        point2 = lst.landmark[finger2_idx]
        
        return self.calculate_disstance(point1, point2) < threshold

    def move_cursor(self, hand_landmarks, frm):
        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        thumb_x, thumb_y = int(thumb_tip.x * frm.shape[1]), int(thumb_tip.y * frm.shape[0])
        pyautogui.moveTo(thumb_x, thumb_y)

    def detect_continuous_move(self, hand_keyPoints):
        # Example logic to detect continuous up or down movement based on hand position
        tip_y = hand_keyPoints.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y
        base_y = hand_keyPoints.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y

        if tip_y < base_y - 0.05:
            return 'up'
        elif tip_y > base_y + 0.05:
            return 'down'
        else:
            return None

    def volume_up_continuous(self):
        # Increase volume using osascript command on macOS
        subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 5)"])

    def volume_down_continuous(self):
        # Decrease volume using osascript command on macOS
        subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 5)"])

    def start_recognition(self):
        self.running = True
        while self.running:
            end_time = time.time()
            ret, frm = self.cap.read()

            if not ret:
                continue

            frm = cv2.flip(frm, 1)
            rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

            res = self.hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            if res.multi_hand_landmarks:

                hand_keyPoints = res.multi_hand_landmarks[0]
                self.gesture_history.append(hand_keyPoints)

                cnt, extended = self.count_fingers(hand_keyPoints)
                print(f"Finger Count: {cnt}, Extended Fingers: {extended}")

                index_middle_attached = self.fingers_attached(hand_keyPoints, 8, 12)
                index_thumb_attached = self.fingers_attached(hand_keyPoints, mp.solutions.hands.HandLandmark.THUMB_TIP, 8)
                print(f"Ready for Swipe Motion")

                if cnt == 0:
                    self.gesture_history.clear()

                if index_middle_attached and extended[0] and extended[1] and not any(extended[2:]):
                    if not self.pinch_active:
                        self.pinch_start = True
                    self.pinch_active = True

                    if not self.cooldown:
                        self.move_cursor(hand_keyPoints, frm)
                else:
                    if self.pinch_active:
                        self.pinch_end = True
                        self.cooldown = True
                        self.cooldown_start_time = time.time()
                    self.pinch_active = False

                    if self.pinch_end:
                        cursor_radius = 20
                        thumb_tip = hand_keyPoints.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
                        thumb_x, thumb_y = int(thumb_tip.x * frm.shape[1]), int(thumb_tip.y * frm.shape[0])
                        pyautogui.click()
                        self.pinch_start = False
                        self.pinch_end = False

                if not self.pinch_active and not self.cooldown:
                    if self.check_four_fingers_extended_and_attached(hand_keyPoints):
                        swipe_direction, scroll_amount = self.detect_swipe(self.gesture_history)
                        print(f"Swipe Direction: {swipe_direction}, Scroll Amount: {scroll_amount}")
                        if swipe_direction:
                            if swipe_direction == 'up_swipe':
                                pyautogui.scroll(scroll_amount)
                                
                            elif swipe_direction == 'down_swipe':
                                pyautogui.scroll(scroll_amount)
                            
                            self.gesture_history.clear()
                    # elif self.check_three_fingers_extended_and_attached(hand_keyPoints):
                    #     move_direction = self.detect_continuous_move(hand_keyPoints)
                    #     if move_direction == 'up':
                    #         self.volume_up_continuous()
                    #     elif move_direction == 'down':
                    #         self.volume_down_continuous()
                    #     else:
                    #         pass
                    else:
                        if not(self.prev==cnt):
                            if not(self.start_init):
                                start_time = time.time()
                                self.start_init = True

                            elif (end_time-start_time) > 0.2:
                                if extended[4] and not any(extended[0:4]):
                                    pyautogui.hotkey('command', 't')

                                elif extended.count(True) == 1 and extended[3]:
                                    pyautogui.hotkey('command', 'w')

                                elif extended.count(True) == 2 and extended[4] and extended[0]:
                                    pyautogui.press("left")
                                
                                elif extended.count(True) == 2 and extended[2] and extended[3]:
                                    pyautogui.press("right")

                                elif (cnt == 3):
                                    pyautogui.press("up")

                                elif (cnt == 4):
                                    pyautogui.press("down")

                                elif (cnt == 5):
                                    pyautogui.press("space")

                                self.prev = cnt
                                self.start_init = False
                if self.cooldown and (time.time() - self.cooldown_start_time) > self.cooldown_duration:
                    self.cooldown = False

                self.drawing.draw_landmarks(frm, hand_keyPoints, self.hands.HAND_CONNECTIONS)

            self.frame_queue.put(frm)

    def stop_recognition(self):
        self.running = False
        cv2.destroyAllWindows()
        self.cap.release()

if __name__ == "__main__":
    app = GestureRecognitionApp()
    app.start_recognition()

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    app.stop_recognition()