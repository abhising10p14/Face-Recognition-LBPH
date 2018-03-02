import cv2
import os
import data_create as dc
import numpy as np
#create our LBPH face recognizer 
face_recognizer =cv2.face.LBPHFaceRecognizer_create()
 
#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.createEigenFaceRecognizer()
 
#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.createFisherFaceRecognizer()

#train our face recognizer of our training faces
face_recognizer.train(dc.faces, np.array(dc.labels))
