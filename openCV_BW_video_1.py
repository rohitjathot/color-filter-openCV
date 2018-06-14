import numpy as np
import cv2
import math

def capture():
	cap = cv2.VideoCapture(0)
	i=0
	h_count_list = {}
	s_count_list = {}
	v_count_list = {}
	delta = 50

	h_min_max = [2,18]
	s_min_max = [50,255]
	v_min_max = [50,255]

	while(i<100):
		i+=1
		ret, frame = cap.read()
		frame = cv2.flip( frame, 1 )
		img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		frame ,h,s,v = threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

		cv2.imshow('frame',cv2.cvtColor(frame, cv2.COLOR_HSV2BGR))

		if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	cap.release()
	cv2.destroyAllWindows()

def threshold(image):

	h = image.shape[0]
	w = image.shape[1]

	h_dic = {}
	s_dic = {}
	v_dic = {}

	for x in range(200, 400):
		for y in range(200, 400):
			h_temp = image[x,y][0]
			s_temp = image[x,y][1]
			v_temp = image[x,y][2]
			if h_temp in h_dic:
				h_dic[h_temp] = h_dic[h_temp] + 1
			else:
				h_dic[h_temp] = 1

			if s_temp in s_dic:
				s_dic[s_temp] = s_dic[s_temp] + 1
			else:
				s_dic[s_temp] = 1

			if v_temp in v_dic:
				v_dic[v_temp] = v_dic[v_temp] + 1
			else:
				v_dic[v_temp] = 1

	h_mode = 0
	s_mode = 0
	v_mode = 0


	h_count_max = 0
	s_count_max = 0
	v_count_max = 0

	
	for i,j in h_dic.iteritems():
		if j>h_count_max:
			h_count_max = j
			h_mode = i

	for i,j in s_dic.iteritems():
		if j>s_count_max:
			s_count_max = j
			s_mode = i

	for i,j in v_dic.iteritems():
		if j>v_count_max:
			v_count_max = j
			v_mode = i

	for x in range(200, 400):
		for y in range(200, 400):
			image[x,y]=h_mode,s_mode,v_mode

	return image , h_mode,s_mode,v_mode
	
if __name__ == '__main__':
	capture()