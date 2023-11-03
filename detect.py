from keras.models import load_model
import cv2
import numpy as np
from yoloface import face_analysis
import time, math, argparse, cv2, sys, torch
import json

# Load model
model_age = load_model('./model/model_age.hdf5')
model_gender = load_model('./model/model_gender.hdf5')
model_ethnicity = load_model('./model/model_ethnicity.hdf5')

# Label
label_gender = ['Male', 'Female']
label_ethnicity = ['White', 'Black', 'Asian', 'Indian', 'Other']


class Sample:

    def __init__(self, args):

        self.args = args

    # Detect Image
    def detect_image(self):

        with open(self.args.input, 'r') as json_file:
            data = json.load(json_file)

        num_sets = len(data)

        print(f'There are {num_sets} sets in the JSON file.')

        New_Jdata = []
  
        for entry in data:
            source_path = entry.get('Refernece_img', '')
            target_path = entry.get('reference_video', '')
            output_path = entry.get('output_video', '')

            m_count = 0
            fm_count = 0


            img = cv2.imread(source_path)
            #face detection box
            face=face_analysis()
            _,box,_=face.face_detection(image_path=source_path,model='full')
            for x,y,w,h in box:
                cv2.rectangle(img, (x,y), (x+h,y+w), (0,255,0), 2)
                img_detect = cv2.resize(img[y:y+w, x:x+h], dsize=(50, 50)).reshape(1, 50, 50, 3)
                #Detect Age
                age = np.round(model_age.predict(img_detect/255.))[0][0]
                #Detect Gender
                gender_arg = np.round(model_gender.predict(img_detect/255.)).astype(np.uint8)
                gender = label_gender[gender_arg[0][0]]
                #Detect Ethnicity
                ethnicity = label_ethnicity[np.argmax(model_ethnicity.predict(img_detect/255.))]
                
                #Draw
                # cv2.putText(img, f'Age: {age}, {gender}, {ethnicity}', (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (np.random.randint(150, 230),np.random.randint(50, 150),np.random.randint(80, 180)), 1, cv2.LINE_AA)
            # cv2.imwrite(f'./image/test/test1.jpg', img)
            # cv2.imshow('detect', img)
            # cv2.waitKey(0)
            # return img

            print(age)
            print(gender)
            print(ethnicity)

            item = {
                "output_video": output_path,
                "Refernece_img": source_path,
                "reference_video": target_path,
                "Gender":gender,
                "Age":age
            }
            New_Jdata.append(item)

        with open(self.args.output, 'w') as json_file:
            json.dump(New_Jdata, json_file, indent=4)

    def detect_video(url):
        frame = cv2.VideoCapture(url)
        while True:
            _, img = frame.read()
            img = cv2.flip(img, 1)
            face=face_analysis()
            _,box,_=face.face_detection(frame_arr=img,frame_status=True,model='tiny')
            for x,y,w,h in box:
                # cv2.rectangle(img, (x,y), (x+h,y+w), (0,255,0), 2)
                img_detect = cv2.resize(img[y:y+w, x:x+h], dsize=(50, 50)).reshape(1, 50, 50, 3)
                #Detect Age
                age = np.round(model_age.predict(img_detect/255.))[0][0]
                #Detect Gender
                gender_arg = np.round(model_gender.predict(img_detect/255.)).astype(np.uint8)
                gender = label_gender[gender_arg[0][0]]
                #Detect Ethnicity
                ethnicity = label_ethnicity[np.argmax(model_ethnicity.predict(img_detect/255.))]
                
                #Draw
                # cv2.putText(img, f'Age: {age}, {gender}, {ethnicity}', (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (np.random.randint(150, 230),np.random.randint(50, 150),np.random.randint(80, 180)), 1, cv2.LINE_AA)
            # cv2.imshow('detect', img)
            # if cv2.waitKey(1) == ord('q'):
            #     break
        # frame.release()
        # cv2.destroyAllWindows()

    # detect_video(0)

parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('-i', '--input', type=str,
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('-o', '--output', type=str, default="",
                    help='Path to output the prediction in case of single image.')

args = parser.parse_args()
s = Sample(args)
s.detect_image()