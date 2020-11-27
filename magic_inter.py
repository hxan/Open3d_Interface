import open3d 
import numpy as np
import pyrealsense2 as rs
import time
import cv2
import random
from sympy import *

class MyPlane:
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def __str__(self):
        return f'{self.A}x + {self.B}y + {self.C}z +{self.D}=0 '


class MyPoint:
    x = 0.0
    y = 0.0
    z = 0.0
    distance = 0.0

    def __init__(self, coordinate):
        self.x = coordinate[0]
        self.y = coordinate[1]
        self.z = coordinate[2]

    def computeDistance(self, plane: MyPlane):
        self.distance = abs(self.x * plane.A + self.y * plane.B + self.z * plane.C + plane.D) \
                        / (plane.A ** 2 + plane.B ** 2 + plane.C ** 2) ** 0.5
        return self.distance

    def __str__(self):
        return f'({self.x},{self.y},{self.z})'

def getBehindTwoPoints(front_left_point: MyPoint, front_right_point: MyPoint, desk_plane: MyPlane) -> tuple:
    x3 = Symbol('x3')
    y3 = Symbol('y3')
    z3 = Symbol('z3')
    x4 = Symbol('x4')
    y4 = Symbol('y4')
    z4 = Symbol('z4')
    x1, y1, z1 = front_left_point.x, front_left_point.y, front_left_point.z
    x2, y2, z2 = front_right_point.x, front_right_point.y, front_right_point.z
    A, B, C, D = desk_plane.A, desk_plane.B, desk_plane.C, desk_plane.D
    a = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5 # 边长
    results = solve([
        (x4 - x1) * (x1 - x2) + (y4 - y1) * (y1 - y2) + (z4 - z1) * (z1 - z2),
        ((x4 - x1) ** 2 + (y4 - y1) ** 2 + (z4 - z1) ** 2) ** 0.5 - 0.55,
        A * x4 + B * y4 + C * z4 + D
    ], [x4, y4, z4])
    #print(results)
    behind_left_point = MyPoint(results[0]) if results[0][2] > results[1][2] else MyPoint(results[1])
    # x4,y4,z4 = behind_left_point.x,behind_left_point.y,behind_left_point.z
    results = solve([
        (x3 - x2) * (x2 - x1) + (y3 - y2) * (y2 - y1) + (z3 - z2) * (z2 - z1),
        ((x3 - x2) ** 2 + (y3 - y2) ** 2 + (z3 - z2) ** 2) ** 0.5 - a,
        A * x3 + B * y3 + C * z3 + D
    ], [x3, y3, z3])
    behind_right_point = MyPoint(results[0]) if results[0][2] > results[1][2] else MyPoint(results[1])

    return behind_left_point, behind_right_point        

def ransacLine(points, iter=2000, good_len=3, sample_points=10):
    '''
    ransacLine is to make robust prediction of a straight line out of a point set
    the algorithm is referred on wikipedia, check it out!

    [Input]
    points: 2-D points list
    [Parameters]
    iterations: more iterations might come out better model at the expense of more time consumption
    good_len: how many points around do you think is good enough to further
    sample point: how many points randomly choose at first
    '''

    bestLoss = 99999999999

    for i in range(0, iter):
        # choose maybeInliners
        maybeInliners = random.sample(points, sample_points)

        # make maybeModel
        maybeModel = cv2.fitLine(np.asanyarray(maybeInliners), 1, 0.1, 0.01, 0.01).tolist()
        # maybeModel: [dx, dy, x0, y0]
        # maybeFunc = ax + by + c = [a, b, c] = [dy, -dx, dx*y0 - dy*x0]
        maybeFunc = [maybeModel[1][0], -maybeModel[0][0], maybeModel[0][0]*maybeModel[3][0] - maybeModel[1][0]*maybeModel[2][0]]

        alsoInliners = []
        for i in points:
            #print(maybeInliners)
            #print("xxxxxxxxxxxxx")
            if not i in maybeInliners:
                # judge if near the line
                if i[0]*maybeFunc[0] + i[1]*maybeFunc[1] + maybeFunc[2] < 4:
                    alsoInliners.append(i)
        
        if len(alsoInliners) > good_len or 1==1:
            # This implies that we may have found a good model, now test how good it is
            maybeInliners.extend(alsoInliners)
            betterModel = cv2.fitLine(np.asanyarray(maybeInliners), 1, 0.1, 0.01, 0.01).tolist()
            betterFunc = [betterModel[1][0], -betterModel[0][0], betterModel[0][0]*betterModel[3][0] - betterModel[1][0]*betterModel[2][0]]
            thisLoss = 0
            for i in maybeInliners:  # use L2 as the loss function
                thisLoss += (i[0]*betterFunc[0] + i[1]*betterFunc[1] + betterFunc[2]) * (i[0]*betterFunc[0] + i[1]*betterFunc[1] + betterFunc[2])
            if thisLoss <= bestLoss or 1==1:
                bestModel = betterModel
                bestFunc = betterFunc
                bestLoss = thisLoss

    return bestModel



class Open3dCalc:
	def __init__(self, xyz):#初始化，传入一个n*3的矩阵（numpy类型 ）
		self.xyz=xyz#
		self.pcd=open3d.geometry.PointCloud()#生成点云
		self.pcd.points = open3d.utility.Vector3dVector(xyz)
		self.pcd = self.pcd.voxel_down_sample(voxel_size=0.005)
		self.axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

	def __del__(self):
		pass

	def planeSeg(self):#平面分割，在初始化之后调用，返回一个n*3的矩阵
		self.plane_model, self.inliers = self.pcd.segment_plane(distance_threshold=0.003	,
                                         ransac_n=400,
                                         num_iterations=1000)
		[self._a, self._b, self._c, self._d] = self.plane_model
		print(f"Plane equation: {self._a:.2f}x + {self._b:.2f}y + {self._c:.2f}z + {self._d:.2f} = 0")

		self.inlier_cloud = self.pcd.select_by_index(self.inliers)

		return np.asarray(self.inlier_cloud.points)

	def showStatic(self,_xyz):#可视化，传入一个n*3的矩阵,这个函数是阻塞的
		show_pcd=open3d.geometry.PointCloud()
		show_pcd.points = open3d.utility.Vector3dVector(_xyz)
		self.inlier_cloud.paint_uniform_color([0, 1, 0])
		open3d.visualization.draw_geometries([show_pcd,self.inlier_cloud])

def getDesk3d(dets,depth_frame,depth_intrin):#传入三维点集
	
	a = Open3dCalc(dets)
	pic_raw = cv2.imread('aa.jpg')
	for i in a.planeSeg():
	    temp=rs.rs2_project_point_to_pixel( depth_intrin,[i[0],i[1],i[2]] )
	    cv2.circle(pic_raw,(round(temp[0]),round(temp[1]) ),1,(0,0,255))#画出平面点集

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))#定义结构元素的形状和大小
	pic_raw = cv2.erode(pic_raw, kernel)#腐蚀操作
	pic_raw  = cv2.blur(pic_raw,(2,2))#滤波
	
	left_x = 640
	left_y = 0

	right_x = 0
	right_y = 0

	bottom_x = 0 
	bottom_y = 0

	for x_ in range(0,640):
		for y_ in range(0,480):
			
			if pic_raw[y_,x_,0] != 255 and pic_raw[y_,x_,1] != 255:
				 if left_x >= x_:
				 	left_x = x_
				 	left_y = y_

				 if right_x <= x_:
				 	right_x = x_
				 	right_y = y_

				 if bottom_y <= y_:
				 	bottom_y = y_ 
				 	bottom_x = x_
			
	np_point_left = []
	np_point_right = []
	pic_line = cv2.imread('aa.jpg')

	for x_ in range(left_x,bottom_x):
		for y_ in range(1,bottom_y):
			if pic_raw[480-y_,x_,0] != 255 and pic_raw[480-y_,x_,1] != 255:
				cv2.circle(pic_line,(x_,480-y_),1,(255,0,0))
				np_point_left.append( [x_, 480-y_] )
				break;
	
	for x_ in range(bottom_x,right_x):
		for y_ in range(1,bottom_y):
			if pic_raw[480-y_,x_,0] != 255 and pic_raw[480-y_,x_,1] != 255:
				cv2.circle(pic_line,(x_,480-y_),1,(255,0,0))
				np_point_right.append( [x_, 480-y_] )
				break;

	if len(np_point_left) > 10 and len(np_point_right) < 10:
		temp_left_vector = ransacLine(np_point_left)
		temp_right_vector = temp_left_vector

	elif len(np_point_left) < 10 and len(np_point_right) > 10:
		temp_right_vector = ransacLine(np_point_right)
		temp_left_vector = temp_right_vector
	else :
		temp_left_vector = ransacLine(np_point_left)
		temp_right_vector = ransacLine(np_point_right) 

	vector_left_t_left = ( left_x-temp_left_vector[2][0] )/temp_left_vector[0][0]
	vector_left_t_bottom =  ( bottom_x-temp_left_vector[2][0] )/temp_left_vector[0][0]
		
	left_y = temp_left_vector[3][0]+vector_left_t_left*temp_left_vector[1][0]
	left_y = round(left_y)
	
	bottom_y = temp_left_vector[3][0]+vector_left_t_bottom*temp_left_vector[1][0]
	bottom_y = round(bottom_y)

	vector_right_t_right = ( right_x-temp_right_vector[2][0] )/temp_right_vector[0][0]
	vector_right_t_bottom = ( bottom_x-temp_right_vector[2][0] )/temp_right_vector[0][0]
	right_y = temp_right_vector[3][0]+vector_right_t_right*temp_right_vector[1][0]
	right_y = round(right_y)

	bottom_y = temp_right_vector[3][0]+vector_right_t_bottom*temp_right_vector[1][0]
	bottom_y = round(bottom_y)
	
	cv2.circle(pic_raw,(left_x,left_y),2,(255,0,0))
	cv2.circle(pic_raw,(right_x,right_y),2,(255,0,0))
	cv2.circle(pic_raw,(bottom_x,bottom_y),2,(255,0,0))

	cv2.circle(pic_line,(left_x,left_y),2,(100,1,100))
	cv2.circle(pic_line,(right_x,right_y),2,(100,1,100))
	cv2.circle(pic_line,(bottom_x,bottom_y),2,(100,1,100))	

	if 0.2>abs( abs(temp_left_vector[1][0]/temp_left_vector[0][0])-abs(temp_right_vector[1][0]/temp_right_vector[0][0]) ):
		cv2.circle(pic_line,(right_x,right_y),5,(100,1,100))	
		cv2.circle(pic_line,(left_x,left_y),5,(100,1,100))
		cv2.circle(pic_raw,(right_x,right_y),5,(100,1,100))	
		cv2.circle(pic_raw,(left_x,left_y),5,(100,1,100))
		#return left_x,left_y,right_x,right_y
	
	else:	
		if abs(temp_left_vector[1][0]/temp_left_vector[0][0]) < abs(temp_right_vector[1][0]/temp_right_vector[0][0]):
			cv2.circle(pic_line,(bottom_x,bottom_y),5,(100,1,100))	
			cv2.circle(pic_line,(left_x,left_y),5,(100,1,100))	
			cv2.circle(pic_raw,(bottom_x,bottom_y),5,(100,1,100))	
			cv2.circle(pic_raw,(left_x,left_y),5,(100,1,100))	
			#return left_x,left_y,bottom_x,bottom_y

		elif abs(temp_left_vector[1][0]/temp_left_vector[0][0]) > abs(temp_right_vector[1][0]/temp_right_vector[0][0]):
			cv2.circle(pic_line,(bottom_x,bottom_y),5,(100,1,100))	
			cv2.circle(pic_line,(right_x,right_y),5,(100,1,100))	
			cv2.circle(pic_raw,(bottom_x,bottom_y),5,(100,1,100))	
			cv2.circle(pic_raw,(right_x,right_y),5,(100,1,100))	
			#return bottom_x,bottom_y,right_x,right_y##################################################################################################################

	#下面为得到3d坐标
	left_3d_front = 0
	right_3d_front = 0

	left_behind_x = 0
	left_behind_y = 0

	right_behind_x = 0
	right_behind_y = 0

	right_behind_3d = 0
	left_behind_3d = 0

	'''
	获得
	'''

	distance = 0
	x = left_x-1
	y = 0
	while x<639:
		y=0
		while y<479:
			distance = depth_frame.get_distance(x,y)
			if distance!=0.0:
				#left_x = x
				#left_y = y
				
				x = 10000
				break;
			y+=1
		x+=1
	#print(left_x,left_y,distance)
	left_3d_front = rs.rs2_deproject_pixel_to_point(depth_intrin,[left_x,left_y],distance)#二维转三维
 
	distance = 0
	x = right_x+1
	y = 0
	while x<639:
		y=0
		while y<479:
			distance = depth_frame.get_distance(x,y)
			if distance!=0.0:
				#right_x = x
				#right_y = y
				x = 10000
				break;
			y+=1
		x-=1
	right_3d_front = rs.rs2_deproject_pixel_to_point(depth_intrin,[right_x,right_y],distance)#二维转三维

	#print(right_x,right_y,distance)
	







	left_behind_3d,right_behind_3d = getBehindTwoPoints( MyPoint(left_3d_front),MyPoint(right_3d_front),MyPlane(a._a,a._b,a._c,a._d) )

	left_behind_x = rs.rs2_project_point_to_pixel( depth_intrin,[left_behind_3d.x,left_behind_3d.y,left_behind_3d.z] )[0]
	left_behind_y = rs.rs2_project_point_to_pixel( depth_intrin,[left_behind_3d.x,left_behind_3d.y,left_behind_3d.z] )[1]
			
			
	right_behind_x = rs.rs2_project_point_to_pixel( depth_intrin,[right_behind_3d.x,right_behind_3d.y,right_behind_3d.z] )[0]
	right_behind_y = rs.rs2_project_point_to_pixel( depth_intrin,[right_behind_3d.x,right_behind_3d.y,right_behind_3d.z] )[1]

	cv2.circle(pic_raw,(int(left_behind_x),int(left_behind_y) ),5,(0,0,255))	
	cv2.circle(pic_raw,(int(right_behind_x),int(right_behind_y) ),5,(0,0,255))

	
	# cv2.circle(imageColor,(int(left_behind_x),int(left_behind_y) ),2,(0,0,255))	
	# cv2.circle(imageColor,(int(right_behind_x),int(right_behind_y) ),2,(0,0,255))
	# cv2.circle(imageColor,(int(left_x),int(left_y) ),2,(0,0,255))	
	# cv2.circle(imageColor,(int(right_x),int(right_y) ),2,(0,0,255))
	# cv2.imshow("asa",imageColor)
	
	cv2.imshow("pic_raw",pic_raw)
	cv2.imshow("pic_line",pic_line)
	print("左右点")
	print(left_x,left_y)
	#while cv2.waitKey(1)!='a':
	cv2.waitKey(5000)

	#print(left_3d_front,right_3d_front,left_behind_3d,right_behind_3d)
	left_behind_3d = [left_behind_3d.x,left_behind_3d.y,left_behind_3d.z]
	right_behind_3d = [right_behind_3d.x,right_behind_3d.y,right_behind_3d.z]
	#return left_3d_front,
	
	four  = np.array([left_3d_front])
	four = np.append(four,[right_behind_3d],axis=0)
	four = np.append(four,[left_behind_3d],axis=0)
	four = np.append(four,[right_3d_front],axis=0)
	print("four")
	print(four)
	a.showStatic(four)
	'''
	@penway
	interface modified
	'''
	left_3d_front[0] *= 100
	left_3d_front[1] *= 100
	left_3d_front[2] *= 100
	left_behind_3d[0] *= 100
	left_behind_3d[1] *= 100
	left_behind_3d[2] *= 100
	right_3d_front[0] *= 100
	right_3d_front[1] *= 100
	right_3d_front[2] *= 100

	lfPoint = np.asanyarray(left_3d_front, np.int)
	lrPoint = np.asanyarray(left_behind_3d, np.int)
	rfPoint = np.asanyarray(right_3d_front, np.int)

	
	return lfPoint, lrPoint, rfPoint, [left_x, left_y], [right_x, right_y], [left_behind_x, left_behind_y], [right_behind_x, right_behind_y]
	#return left_3d_front,right_3d_front,left_behind_3d,right_behind_3d
	
	



#pipeline = rs.pipeline()
#config = rs.config()
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#config.enable_device_from_file(bag)

#profile = pipeline.start(config)
#aligner = rs.align(rs.stream.color)
#depth_sensor = profile.get_device().first_depth_sensor()
#color_sensor = profile.get_device().first_color_sensor()
'''
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file("20201101_170834.bag")

profile = pipeline.start(config)
aligner = rs.align(rs.stream.color)
depth_sensor = profile.get_device().first_depth_sensor()
color_sensor = profile.get_device().first_color_sensor()
'''


#depth_sensor.set_option(rs.option.motion_range, 29)
#depth_sensor.set_option(rs.option.laser_power, 16)
#depth_sensor.set_option(rs.option.accuracy, 3)
#depth_sensor.set_option(rs.option.confidence_threshold, 12)
#depth_sensor.set_option(rs.option.filter_option, 5)

#frames = pipeline.wait_for_frames()
#depth_frame = frames.get_depth_frame()
'''
frame = pipeline.wait_for_frames()  # 等两张图像同时出现
frame = aligner.process(frame)      # 深度图与彩色图对齐操作

frameColor = frame.get_color_frame()  # 获得两张图像
frameDepth = frame.get_depth_frame()

#intrinColor = frameColor.profile.as_video_stream_profile().intrinsics        # 获得图像的内参，后面会用到
intrinDepth = frameDepth.profile.as_video_stream_profile().intrinsics

#imageColor = np.asanyarray(frameColor.get_data())


fo = open('name.txt', "w")

str_total=""

for i in range(0,640):
    for j in range(0,480):
        distance = frameDepth.get_distance(i,j)
        if distance==0.0 or distance==-0.0:
        	continue
        t = rs.rs2_deproject_pixel_to_point(intrinDepth,[i,j],distance)
        str_total += str(t).strip('[').strip(']')+'\n'

fo.write(str_total)

dets = np.loadtxt('name.txt',delimiter=',')

distance = frameDepth.get_distance(320,240)
t = rs.rs2_deproject_pixel_to_point(intrinDepth,[320,240],distance)

print("二维点转三维点 320 240" )
print(t)


temp=rs.rs2_project_point_to_pixel( intrinDepth,[t[0],t[1],t[2]] )
print(temp)

getDesk3d(dets,frameDepth,intrinDepth) #传回四个list变量：左前三维点，右前三维点，左后三维点，右后三维点
'''
