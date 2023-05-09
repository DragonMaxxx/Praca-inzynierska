import os

from kivy.app import App
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.video import Video, Image
import cv2
import numpy as np

Builder.load_file('GUI.kv')


class Videos(Video):
    pass

class File(Popup):
    load = ObjectProperty()


class MyLayout(FloatLayout):
    file_path = StringProperty("No file chosen")
    the_popup = ObjectProperty(None)

    def open_popup(self):
        self.the_popup = File(load=self.load)
        self.the_popup.open()

    def load(self, selection):
        self.file_path = str(selection[0])
        self.the_popup.dismiss()
        print(self.file_path)

        # check for non-empty list i.e. file selected
        if self.file_path:
            self.ids.video1.source = self.file_path
    def run(self):
        file_path = self.file_path
        def detect_face(frame):
            net = cv2.dnn.readNetFromCaffe("weights-prototxt.txt", "res_ssd_300Dim.caffeModel")

            (height, width) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            net.setInput(blob)
            detections = net.forward()

            faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence < 0.5:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")

                faces.append((x1, y1, x2, y2))
            return faces

        # Read the input video and create a video capture object
        video = cv2.VideoCapture(file_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Check if the video was opened successfully
        if video.isOpened():
            print("Video opened successfully")
        else:
            print("Error opening video")
            exit()

        # Set the feature detector
        detector = cv2.xfeatures2d.SIFT_create()

        # Set the feature matching algorithm
        matcher = cv2.BFMatcher(cv2.NORM_L2)

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

        PSNR_values = []
        MAE_values = []
        VIF_val = []
        smoothness_values = []
        prev_points = None

        while True:
            # Read the next frame
            ret, frame = video.read()
            if not ret:
                break

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect the features in the frame
            kp, desc = detector.detectAndCompute(gray, None)
            points = np.array([p.pt for p in kp], dtype=np.float32)

            if prev_points is not None:
                matches = matcher.match(desc, prev_desc)
                src_pts = np.array([points[m.queryIdx] for m in matches], dtype=np.float32)
                dst_pts = np.array([prev_points[m.trainIdx] for m in matches], dtype=np.float32)
                M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

                compensated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_CUBIC)
                difference_frame = cv2.absdiff(frame, compensated_frame)

                difference_frame = cv2.cvtColor(difference_frame, cv2.COLOR_BGR2GRAY)

                smoothness = np.mean(difference_frame)

                smoothness_values.append(smoothness)
            else:
                # If this is the first frame, there is no motion to compensate for
                compensated_frame = frame

            # Detect the face in the compensated frame
            faces = detect_face(compensated_frame)

            if len(faces) > 0:
                x1, y1, x2, y2 = faces[0]
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                face_center = (x + w // 2, y + h // 2)

                M = np.float32([[1, 0, frame.shape[1] // 2 - face_center[0]],
                                [0, 1, frame.shape[0] // 2 - face_center[1]]])
                compensated_frame = cv2.warpAffine(compensated_frame, M, (frame.shape[1], frame.shape[0]))

            # Compute the PSNR and MAE values
            # PSNR = cv2.PSNR(frame, compensated_frame)
            # VIF = vifp_mscale(frame, compensated_frame)
            # MAE = np.mean(np.abs(compensated_frame-frame))

            # Append the values to the lists
            # PSNR_values.append(PSNR)
            ##VIF_val.append(VIF)
            # MAE_values.append(MAE)
            # frame_concat = np.concatenate((frame, compensated_frame), axis=1)
            frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cframe_resized = cv2.resize(compensated_frame, (0, 0), fx=0.5, fy=0.5)

            # Show the original and compensated frames
            cv2.imshow("Original", frame_resized)
            cv2.imshow("Compensated", cframe_resized)
            out.write(compensated_frame)


class App(App):
    def build(self):
        Window.clearcolor = (0,0,0,0)
        return MyLayout()


if __name__ == '__main__':
    App().run()
