import os
from  detection.retinaface import RetinaNetDetector
import cv2
import numpy as np 
from copy import deepcopy

from models.net import ModelAgeGender
from statistics import mean 


def get_face(frame, coord):
    frame = deepcopy(frame)
    height, width = frame.shape[0:2]
    l, t, r, b, _ = coord
    w_f = int(r-l)
    h_f = int(b-t)

    # l_f = max(0, int(l - 0.2*w_f))
    # t_f = max(0, int(t - 0.2*h_f))
    # r_f = min(width, int(r + 0.2*w_f))
    # b_f = min(height, int(b + 0.05*h_f))

    l_f = max(0, int(l))
    t_f = max(0, int(t))
    r_f = min(width, int(r))
    b_f = min(height, int(b))

    face_img = frame[t_f:b_f, l_f:r_f, :]
    return face_img, (l_f, t_f, r_f, b_f)


def draw_face(frame, face_crd, age, gender):
    x1, y1, x2, y2 = face_crd
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
   
    info_face = "Age:{}-Gender:{}".format(age, gender)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)
    thickness = 1

    cv2.putText(frame, info_face, (x1, y2), font, font_scale, color, thickness)
    return frame

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def demo_video(video_path=None):
    # import ipdb; ipdb.set_trace()
    # video_name = "thangld"#video_path.split("/")[-1].split(".")[0]
    
    age_range_lst = chunkIt(range(0,100),10)
    if video_path is None:
        video_name = "thangld"
        video_path = 0
    else:
        video_name = video_path.split("/")[-1].split(".")[0]
    if not os.path.exists("./{}".format(video_name)):
        os.makedirs("./{}".format(video_name))
    vid = cv2.VideoCapture(video_path)
    detector = RetinaNetDetector()
    # estimator = ModelAgeGender()
    # estimator.init_model("mobilenet_v2", num_age_classes=81, widen_factor=1, pretrained=False)
    # estimator.load_statedict("weights/new_age-gender_genos.pt")
    i = 0
    while True:
        ret, frame = vid.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if not ret:
            break

        det_faces = detector.predict(frame)
        if len(det_faces) == 0:
            cv2.imshow("", frame)
            key = cv2.waitKey(1) & 0xff
            if key == ord("q"):
                break
        
        for face_info in det_faces[0]:
            face_img, face_crd = get_face(frame, face_info)
            cv2.imwrite("./{}/{}.png".format(video_name, i), face_img)

            age_lst = []
            # for i in range(5):
            #     age, gender = estimator.predict_image(face_img[:,:, ::-1])
            #     age_lst.append(age)
            # # import ipdb; ipdb.set_trace()
            # age = sum(age_lst)//len(age_lst)
            # age_range = str([a for a in age_range_lst if age in a][0])
            frame = draw_face(frame, face_crd, "", "")
            i = i + 1
        cv2.namedWindow("a", cv2.WINDOW_NORMAL)
        cv2.imshow("", frame)
        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):
            break

def demo_image(image_path):
    detector = RetinaNetDetector()
    estimator = ModelAgeGender()
    estimator.init_model("mobilenet_v2", num_age_classes=81, widen_factor=1, pretrained=False)
    estimator.load_statedict("weights/new_age-gender_genos.pt")

    frame = cv2.imread(image_path)

    # for i in range(200):
    det_faces = detector.predict(frame)
    if len(det_faces) != 0:
        for face_info in det_faces[0]:
            face_img, face_crd = get_face(frame, face_info)
            for i in range(5):
                age, gender = estimator.predict_image(face_img[:,:, ::-1])
            frame = draw_face(frame, face_crd, age, gender)
            print(gender)
    
    cv2.namedWindow("", cv2.WINDOW_NORMAL)
    cv2.imshow("", frame)
    key = cv2.waitKey(0) & 0xff


if __name__ == "__main__":
    # demo_video("/Users/thangld/Downloads/congp_MI0124.mov")
    demo_video()
    # demo_image("test_dataset/408.jpg")