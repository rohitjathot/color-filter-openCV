import cv2
import numpy as np
def main():
	cap = cv2.VideoCapture(0)
	lower_skin = np.array([2,50,50])
	upper_skin = np.array([18,255,255])
	delta_h = 10
	delta_s = 50
	delta_v = 100
	temp_h,temp_s,temp_v = 0,0,0
	i = 0
	count = 0
	while(1):
		count = count + 1
		_, frame = cap.read()
		frame = cv2.flip( frame, 1 )
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		
		mask = cv2.inRange(hsv, lower_skin, upper_skin)
		res = cv2.bitwise_and(frame,frame, mask= mask)

		frame ,h,s,v = threshold(cv2.cvtColor(res, cv2.COLOR_BGR2HSV),lower_skin,upper_skin)
		frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
		cv2.imshow('frame',frame)
		# cv2.imshow('mask',mask)
		# cv2.imshow('res',res)

		temp_h = temp_h + h
		temp_s = temp_s + s
		temp_v = temp_v + v
		
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
			break
	temp_h=temp_h/count;
	temp_s=temp_s/count;
	temp_v=temp_v/count;

	cv2.destroyAllWindows()
	cap.release()

	cap = cv2.VideoCapture(0)

	# cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
	# cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);

	print(temp_h,temp_s,temp_v)

	while(i<800):
		i = i + 1
		_, frame = cap.read()
		frame = cv2.flip( frame, 1 )
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		# mask = cv2.inRange(hsv, lower_skin, upper_skin)
		mask = cv2.inRange(hsv, np.array([temp_h-delta_h,temp_s-delta_s,temp_v-delta_v]), np.array([temp_h+delta_h,temp_s+delta_s,temp_v+delta_v]))
		res = cv2.bitwise_and(frame,frame, mask= mask)
		res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
		thresh = 127
		res = cv2.threshold(res, thresh, 255, cv2.THRESH_BINARY)[1]
		# cv2.imshow('mask',mask)

		cv2.imwrite("C:/workplace/openCV/hand_like/hand_like_{0}.jpg".format(i),res)

		cv2.imshow('res',res)
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
			break
		# print(temp_h,temp_s,temp_v)

	cv2.destroyAllWindows()
	cap.release()


def threshold(image,lower_skin,upper_skin):

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
			if h_temp>=lower_skin[0] and h_temp<=upper_skin[0]:
				if h_temp in h_dic:
					h_dic[h_temp] = h_dic[h_temp] + 1
				else:
					h_dic[h_temp] = 1
			if s_temp>=lower_skin[1] and s_temp<=upper_skin[1]:
				if s_temp in s_dic:
					s_dic[s_temp] = s_dic[s_temp] + 1
				else:
					s_dic[s_temp] = 1
			if v_temp>=lower_skin[2] and v_temp<=upper_skin[2]:
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
	main()