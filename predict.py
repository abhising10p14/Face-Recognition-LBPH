import detect_face as detect
import data_create as dc
import train as tr
import mapping as mp
import cv2
import numpy as np
#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


face_recognizer =cv2.face.LBPHFaceRecognizer_create()
 
#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.createEigenFaceRecognizer()
 
#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.createFisherFaceRecognizer()

#train our face recognizer of our training faces
face_recognizer.train(dc.faces, np.array(dc.labels))

#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
	#make a copy of the image as we don't want to change original image
	img = test_img
	#detect face from the image
	face, rect = detect.detect_face(img)
	 
	print "Here"
	#predict the image using our face recognizer 
	label, confidence = face_recognizer.predict(face)
	#get name of respective label returned by face recognizer
	print "There"
	label_text = mp.subjects[label]
	 
	#draw a rectangle around face detected
	draw_rectangle(img, rect)
	#draw name of predicted person
	draw_text(img, label_text, rect[0], rect[1]-5)
	 
	return img



print("Predicting images...")
 
#load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test12.jpg")
test_img3 = cv2.imread("test-data/test3.jpeg")
 
#perform a prediction
#predicted_img1 = predict(test_img1)
predicted_img3 = predict(test_img3)
#predicted_img2 = predict(test_img2)
print("Prediction complete")
 
#display both images
#cv2.imshow("figure",predicted_img1)
#cv2.imshow("figure",predicted_img2)
cv2.imshow("figure",  cv2.resize(predicted_img3, (400, 500)))

cv2.waitKey(0)
cv2.destroyAllWindows()
