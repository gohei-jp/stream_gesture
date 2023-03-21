from typing import Optional
from dataclasses import dataclass
import time
from enum import Enum

import cv2
import numpy as np
import mediapipe as mp
from config.config import (
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_CNR_MOUSE,
    RIGHT_CNR_MOUSE,
    COLOR_LIST,
    ON_DELAY_SEC,
    BLACK_CANVAS_SIZE,
    RESIZE_SIZE,
)


class HandGestures(Enum):
    NOT_PA = 0
    PA = 1


class MouseGestures(Enum):
    NORMAL = 0
    CLOSE_TALL = 1
    OPEN_TALL = 2
    OPEN_RISE = 3


class EyeGestures(Enum):
    OPEN = 0
    CLOSE = 1


class GestureApp:
    def __init__(self):
        self.face = Face()
        self.hand = Hand()
        self.hue = 100
        self.brightness = 128
        self.color = [127, 127, 127]

    def update(self, image):
        face_image = self.face.update(image)
        hand_image = self.hand.update(image, self.color)
        concat_image = cv2.hconcat([hand_image, face_image])

        self.create_color()
        self.draw_hue_status(concat_image)

        return concat_image

    def create_color(self):
        if (
            self.face.left_eye.fixed_gesture == EyeGestures.CLOSE.value
            and self.face.right_eye.fixed_gesture == EyeGestures.CLOSE.value
        ):
            self.brightness -= 10
            self.brightness = max(20, self.brightness)
        else:
            self.brightness = self.hand.left_hand.value

        if self.face.left_cnr_mouse.fixed_gesture == MouseGestures.CLOSE_TALL.value:
            self.hue += 5
            self.hue = min(179, self.hue)
            self.hand.right_hand.value = self.hue
        elif self.face.left_cnr_mouse.fixed_gesture == MouseGestures.OPEN_TALL.value:
            self.hue -= 5
            self.hue = max(0, self.hue)
            self.hand.right_hand.value = self.hue
        else:
            self.hue = self.hand.right_hand.value

        saturation = 255
        hsv_color = np.array(
            [[[self.hue, saturation, self.brightness]]], dtype=np.uint8
        )
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        self.color = bgr_color.flatten().tolist()

    def draw_hue_status(self, image):
        cv2.rectangle(
            image,
            pt1=(40, 40),
            pt2=(90, 90),
            color=self.color,
            thickness=-1,
        )
        cv2.rectangle(
            image,
            pt1=(40, 40),
            pt2=(90, 90),
            color=(222, 222, 222),
            thickness=1,
        )
        cv2.putText(
            image,
            f"Hue: {self.hue} Brightness: {self.brightness}",
            (110, 80),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(200, 200, 200),
            thickness=1,
            lineType=cv2.LINE_AA,
        )


class Hand:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.left_hand = BodyPart(
            marks=self.mp_hands.HAND_CONNECTIONS, default_value=100
        )
        self.right_hand = BodyPart(
            marks=self.mp_hands.HAND_CONNECTIONS, default_value=100
        )
        self.hand_label = "Left"

    def update(self, image, color):
        results = self.get_hands(image)
        self.update_gestures(results)
        hand_image = self.draw(results, color)
        return hand_image

    def get_hands(self, image):
        with self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
            max_num_hands=1,
        ) as hands:
            image.flags.writeable = False
            results = hands.process(image)
        return results

    def detect_gesture(self, landmark):
        palm = Utils.get_points_from_marks(landmark, [0, 5, 17])
        finger_angle = Utils.calculate_angle(landmark, 9, 10, 12)
        is_finger_straight = True if 170 < finger_angle < 180 else False

        for i, m in enumerate(landmark):
            if i in [0, 5, 17]:
                continue
            point = (m.x, m.y)
            is_finger_inside_palm = Utils.is_inside_triangle(point, palm)
            if not is_finger_inside_palm and is_finger_straight:
                gesture = HandGestures.PA.value
            else:
                gesture = HandGestures.NOT_PA.value
                break
        return gesture

    def get_yh_pos(self, landmark):
        black_canvas = Utils.create_blank_image()
        h, _ = black_canvas.shape[:2]
        bbox = Utils.get_bbox_from_landmarks(black_canvas, landmark)
        _, cy = Utils.get_center_from_bbox(bbox)
        y = Utils.linear_interpolation(cy)
        return y, h

    def update_gestures(self, results):
        if results.multi_hand_landmarks:
            landmark = results.multi_hand_landmarks[0].landmark
            gesture = self.detect_gesture(landmark)
            self.hand_label = results.multi_handedness[0].classification[0].label
            y, h = self.get_yh_pos(landmark)

            if self.hand_label == "Left":
                self.left_hand.detect(gesture)
                if self.left_hand.fixed_gesture == HandGestures.PA.value:
                    value = int(255 - (y * 255 / h))
                    value = max(0, min(255, value))
                    self.left_hand.value = value
            else:
                self.right_hand.detect(gesture)
                if self.right_hand.fixed_gesture == HandGestures.PA.value:
                    value = int(179 - (y * 179 / h))
                    value = max(0, min(179, value))
                    self.right_hand.value = value

    def draw(self, results, color):
        black_canvas = Utils.create_blank_image()
        image_h, image_w = black_canvas.shape[:2]
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if (
                    self.left_hand.fixed_gesture == HandGestures.PA.value
                    or self.right_hand.fixed_gesture == HandGestures.PA.value
                ):
                    bbox = Utils.get_bbox_from_landmarks(
                        black_canvas, hand_landmarks.landmark
                    )
                    x1, y1, x2, y2 = bbox

                    cv2.rectangle(
                        black_canvas,
                        pt1=(x1, y1),
                        pt2=(x2, y2),
                        color=color,
                        thickness=3,
                    )
                    center = Utils.get_center_from_bbox(bbox)
                    cv2.circle(
                        black_canvas,
                        center=center,
                        radius=28,
                        color=(0, 255, 255),
                        thickness=6,
                    )

                self.mp_drawing.draw_landmarks(
                    black_canvas,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

        if self.hand_label == "Left":
            x1, y1 = 0, 0
            x2, y2 = image_w // 2, image_h
        else:
            x1, y1 = image_w // 2, 0
            x2, y2 = image_w, image_h
        black_canvas = Utils.clip_image(black_canvas, bbox=[x1, y1, x2, y2])
        black_canvas = cv2.resize(black_canvas, dsize=RESIZE_SIZE)
        black_canvas = cv2.flip(black_canvas, 1)

        return black_canvas


class Face:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.right_eye = BodyPart(marks=RIGHT_EYE)
        self.left_eye = BodyPart(marks=LEFT_EYE)
        self.left_cnr_mouse = BodyPart(marks=LEFT_CNR_MOUSE)
        self.right_cnr_mouse = BodyPart(marks=RIGHT_CNR_MOUSE)

    def update(self, image):
        results = self.get_mesh(image)
        self.detect_gestures(results)
        face_image = self.draw(results)

        return face_image

    def get_mesh(self, image):
        """Get facemesh coords powered by mediapipe"""
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(image)
            image.flags.writeable = True
            return results

    def draw(self, results):
        black_canvas = Utils.create_blank_image(BLACK_CANVAS_SIZE)
        if results.multi_face_landmarks:
            marks = results.multi_face_landmarks[0].landmark
            black_canvas = self.draw_results(results, marks)
            if all(black_canvas.shape[:2]):
                black_canvas = cv2.resize(black_canvas, dsize=RESIZE_SIZE)
            else:
                black_canvas = Utils.create_blank_image(BLACK_CANVAS_SIZE)

        return black_canvas

    def detect_gestures(self, results):
        if results.multi_face_landmarks:
            landmark = results.multi_face_landmarks[0].landmark
            self.update_eyes(landmark)
            self.update_mouth(landmark)

    def update_mouth(self, landmark):
        open = self.mouse_open(landmark)
        tall = self.mouse_tall(landmark)
        rise = self.mouse_rise(landmark)

        if not open and tall:
            gesture = MouseGestures.CLOSE_TALL.value
        elif open and tall:
            gesture = MouseGestures.OPEN_TALL.value
        elif open and rise:
            gesture = MouseGestures.OPEN_RISE.value
        else:
            gesture = MouseGestures.NORMAL.value

        self.left_cnr_mouse.detect(gesture)
        self.right_cnr_mouse.detect(gesture)

    def update_eyes(self, landmark):
        b1, b2 = Utils.get_points_from_marks(landmark, [144, 145])
        base_distance = Utils.calculate_distance(b1, b2)
        l1, l2 = Utils.get_points_from_marks(landmark, [374, 386])
        left_eye_distance = Utils.calculate_distance(l1, l2)
        r1, r2 = Utils.get_points_from_marks(landmark, [145, 159])
        right_eye_distance = Utils.calculate_distance(r1, r2)
        left_gesture = (
            EyeGestures.CLOSE.value
            if base_distance > left_eye_distance
            else EyeGestures.OPEN.value
        )
        right_gesture = (
            EyeGestures.CLOSE.value
            if base_distance > right_eye_distance
            else EyeGestures.OPEN.value
        )
        self.left_eye.detect(left_gesture)
        self.right_eye.detect(right_gesture)

    def mouse_open(self, landmark):
        pt1, pt2 = Utils.get_points_from_marks(landmark, [13, 14])
        distance = Utils.calculate_distance(pt1, pt2)
        mouse_open = True if distance > 0.01 else False
        return mouse_open

    def mouse_tall(self, marks):
        mouse_marks = [[0, 62, 17], [0, 291, 17]]
        for mark in mouse_marks:
            m1, m2, m3 = mark
            angle = Utils.calculate_angle(marks, m1, m2, m3)
            mouse_tall = True if angle >= 70 else False
            if not mouse_tall:
                break
        return mouse_tall

    def mouse_rise(self, marks):
        mouse_marks = [62, 291]
        base_mark = 13
        for mark in mouse_marks:
            mouse_rise = True if marks[mark].y <= marks[base_mark].y else False
            if not mouse_rise:
                break
        return mouse_rise

    def draw_face_mesh(self, image, results):
        """Return image drawn a face mesh"""
        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
        return image

    def draw_results(self, results, marks):
        """Return image drawn a face mesh and eyes contour line"""
        black_canvas = Utils.create_blank_image()
        black_canvas = self.draw_face_mesh(black_canvas, results)
        clip_canvas = black_canvas.copy()
        bbox = self.get_face_bbox(clip_canvas)
        clip_canvas = self.left_eye.draw(clip_canvas, marks)
        clip_canvas = self.right_eye.draw(clip_canvas, marks)
        clip_canvas = self.left_cnr_mouse.draw(clip_canvas, marks)
        clip_canvas = self.right_cnr_mouse.draw(clip_canvas, marks)
        clip_canvas = Utils.clip_image(clip_canvas, bbox)

        return clip_canvas

    @staticmethod
    def get_face_bbox(image):
        image_h, image_w = image.shape[:2]
        contours_image = image.copy()
        contours_image = cv2.cvtColor(contours_image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(
            contours_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        padding = 100
        x -= padding // 2
        y -= padding // 2
        w += padding
        h += padding
        x = max(0, min(image_w, x))
        y = max(0, min(image_h, y))
        w = max(0, min(image_w, w))
        h = max(0, min(image_h, h))
        return [x, y, x + w, y + h]


class BodyPart:
    def __init__(
        self, marks, default_value=0, on_delay_sec=ON_DELAY_SEC, color_list=COLOR_LIST
    ):
        # drawing params
        self.marks = marks
        self.color_list = color_list

        # gesture params
        self.fixed_gesture = 0
        self.detecting_gesture_id = 0
        self.value = default_value

        # delay params
        self.on_delay_sec = on_delay_sec
        self.on_sec = 0
        self.sec = 0

    def detect(self, gesture):
        if self.sec > 0:
            self.sec = time.perf_counter() - self.sec
        # -----------------------------------------
        if gesture != self.detecting_gesture_id:
            self.detecting_gesture_id = gesture
            self.on_sec = 0
        self._on_delay(gesture)
        # -----------------------------------------
        self.sec = time.perf_counter()

    def _on_delay(self, gesture_id):
        self.on_sec += self.sec
        if self.on_sec > self.on_delay_sec:
            self.fixed_gesture = gesture_id

    def draw(self, image, marks):
        h, w = image.shape[:2]
        color = self.color_list[self.fixed_gesture]
        if len(self.marks) > 1:
            for mark in self.marks:
                sm, em = mark
                sp = (int(marks[sm].x * w), int(marks[sm].y * h))
                ep = (int(marks[em].x * w), int(marks[em].y * h))
                cv2.line(image, sp, ep, color, thickness=2)
        else:
            m = self.marks[0]
            center = (int(marks[m].x * w), int(marks[m].y * h))
            cv2.circle(image, center=center, radius=5, color=color, thickness=-1)
        return image


from motpy import Detection, MultiObjectTracker
from motpy.testing_viz import draw_track


class Tracker:
    def __init__(self):
        self.tracker = MultiObjectTracker(
            dt=1 / 30,
            tracker_kwargs={"max_staleness": 3},
            model_spec={
                "order_pos": 2,
                "dim_pos": 2,
                "order_size": 0,
                "dim_size": 2,
                "q_var_pos": 8000.0,
                "r_var_pos": 0.1,
            },
            matching_fn_kwargs={"min_iou": 0.001, "multi_match_min_iou": 1.1},
        )
        self.plotters = Plotters()

    def track(self, image, detections):
        active_tracks = self.tracker.step(detections)
        for track in active_tracks:
            draw_track(image, track, thickness=3, random_color=True)
        self.plotters.update(active_tracks)

    def plot(self, image):
        self.plotters.draw(image)

    def plot_clear(self):
        self.plotters.cleanup()


class Plotters:
    def __init__(self) -> None:
        self.plotters = []

    def update(self, tracks):
        for track in tracks:
            exists = False
            bbox = list(map(int, track.box))
            for p in self.plotters:
                if track.id == p.track_id:
                    p.add_dot(bbox)
                    p.unstale()
                    exists = True
            if not exists:
                new_plotter = ColorPlotter(track_id=track.id, max_dot_count=50)
                new_plotter.add_dot(bbox)
                self.plotters.append(new_plotter)

    def cleanup(self):
        self.plotters = []

    def draw(self, image):
        for p in self.plotters:
            p.draw(image)
            p.stale()


class ColorPlotter:
    def __init__(self, track_id, max_dot_count):
        self.track_id = track_id
        self.color = [ord(c) * ord(c) % 256 for c in track_id[:3]]
        self.dots = []
        self.max_dot_count = max_dot_count
        self.staleness = 0

    def add_dot(self, bbox):
        dot = Utils.get_center_from_bbox(bbox)
        self.dots.append(dot)
        if len(self.dots) > self.max_dot_count:
            self.dots.pop(0)

    def pop_dot(self):
        if len(self.dots) > 0:
            self.dots.pop(0)

    def draw(self, image):
        for dot in self.dots:
            cv2.circle(image, center=dot, radius=10, color=self.color, thickness=-1)

    def stale(self):
        self.staleness += 1
        if self.staleness > 10:
            self.pop_dot()

    def unstale(self):
        self.staleness = 0


class Utils:
    def clip_image(image, bbox):
        x1, y1, x2, y2 = bbox
        image = image[y1:y2, x1:x2]
        return image

    def create_blank_image(canvas_size=[1080, 1920], color=None):
        height, width = canvas_size
        blank = np.zeros((height, width, 3), np.uint8)
        if color:
            blank += np.array(color, dtype=np.uint8)
        return blank

    def calculate_distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def calculate_angle(marks, m1, m2, m3):
        p1, p2, p3 = (
            (marks[m1].x, marks[m1].y, marks[m1].z),
            (marks[m2].x, marks[m2].y, marks[m2].z),
            (marks[m3].x, marks[m3].y, marks[m3].z),
        )
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_angle = dot_product / (norm_v1 * norm_v2)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)

    def get_points_from_marks(landmark, marks=[]):
        return [(landmark[m].x, landmark[m].y) for m in marks]

    def get_center_from_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

    def get_bbox_from_landmarks(image, landmark):
        h, w = image.shape[:2]
        xlist = [l.x for l in landmark]
        ylist = [l.y for l in landmark]
        x1 = int(min(xlist) * w)
        y1 = int(min(ylist) * h)
        x2 = int(max(xlist) * w)
        y2 = int(max(ylist) * h)
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        bbox = [x1, y1, x2, y2]
        return bbox

    def is_inside_triangle(point, triangle):
        x, y = point
        (x1, y1), (x2, y2), (x3, y3) = triangle

        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        alpha = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
        beta = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
        gamma = 1.0 - alpha - beta

        return 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1

    def linear_interpolation(x):
        x1, y1 = 300, 0
        x2, y2 = 800, 1080

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        y = m * x + b
        return y
