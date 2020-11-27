import open3d 
import numpy as np
import pyrealsense2 as rs
import time
import cv2
import math

class Open3dCalc:
	def __init__(self, xyz):#初始化，传入一个n*3的矩阵（numpy类型 ）
		self.xyz=xyz#
		self.pcd=open3d.geometry.PointCloud()#生成点云
		self.pcd.points = open3d.utility.Vector3dVector(xyz)
		self.pcd = self.pcd.voxel_down_sample(voxel_size=0.0025 )
		self.axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
		self.vis = open3d.visualization.Visualizer()
		
	def __del__(self):
		pass

	def planeSeg(self):#平面分割，在初始化之后调用，返回一个n*3的矩阵
		self.plane_model, self.inliers = self.pcd.segment_plane(distance_threshold=0.005	,
                                         ransac_n=700,
                                         num_iterations=1000)
		[self._a, self._b, self._c, self._d] = self.plane_model
		print(f"Plane equation: {self._a:.2f}x + {self._b:.2f}y + {self._c:.2f}z + {self._d:.2f} = 0")

		self.inlier_cloud = self.pcd.select_by_index(self.inliers)
		self.outlier_cloud = self.pcd.select_by_index(self.inliers,invert=True)
		return np.asarray(self.inlier_cloud.points)

	def showStatic(self,_xyz=0):#可视化，传入一个n*3的矩阵,这个函数是阻塞的
		show_pcd=open3d.geometry.PointCloud()
		if(0):
			open3d.visualization.draw_geometries([self.inlier_cloud])
		else:
			show_pcd.points = open3d.utility.Vector3dVector(_xyz)
			self.inlier_cloud.paint_uniform_color([0, 1, 0])
			open3d.visualization.draw_geometries([show_pcd,self.inlier_cloud])

	def toSameZ(self):
		N =  math.sqrt(self._a**2+self._b**2+self._c**2)#A^2+B^2+C^2
		M =  math.sqrt(self._a**2+self._b**2)#A^2+B^2
		X = np.asarray([1,0,0])

		e3 = np.asarray([self._a/N,self._b/N,self._c/N])#法向量 Z	
		COS = np.dot(X,e3)
		SIN = math.sqrt( 1-COS**2 )
		e1 = X*SIN ## X轴
		e2 = np.cross(e3,e1)#y
		self.matrix_Z = np.asarray([e1,e2,e3])
		self.res = np.dot(  np.asarray(self.inlier_cloud.points), np.linalg.inv(self.matrix_Z) )
		sum = 0 
		for i in self.res:
			sum += i[2]
		sum = sum/np.size(self.res)*3
		self.z = sum
		return self.res

	def inverse_operation(self,point):#传入点
		return np.dot(point,self.matrix_Z)

	def GetFourDeskPoints(self,image_opencv=None):		
		
		t2 = np.delete(self.res,2, axis = 1)*300+400 #方便可视化
		t2 = t2.astype(np.float32)#改变格式，方便以后运行函数
		#print(type(image_opencv))
		if(type(image_opencv)!=None ):
			for i in t2:
		 		cv2.circle(image_opencv, tuple(i) ,  1, (100,100,100) )

		rect = cv2.minAreaRect(t2)#外接矩形
		box = cv2.boxPoints(rect)#获得四个点
		box = box[np.lexsort(box.T[1,None])]#y轴从小到大的

		if( box[2][0] < box[3][0] ):#如果
			self.left_f_2d  = box[2]
			self.right_f_2d = box[3]
		else:
			self.left_f_2d  = box[3]
			self.right_f_2d = box[2]


		self.left_f_2d[0] += 1
		self.left_f_2d[1] -= 2
		vec_lr = self.right_f_2d - self.left_f_2d #从左到右的向量
		vec_up = ( vec_lr[1],-vec_lr[0] )#从下到上的向量

		self.left_b_2d = self.left_f_2d + vec_up#
		self.right_b_2d = self.right_f_2d + vec_up
		
		# if( box[0][0] < box[1][0] ):
		# 	self.left_b_2d = box[0]
		# 	self.right_b_2d = box[1]
		# else:
		# 	self.left_b_2d = box[1]
		# 	self.right_b_2d = box[0]

		self.left_front_3d = np.asarray([self.left_f_2d[0],self.left_f_2d[1],self.z*300+400])
		self.left_front_3d = self.inverse_operation( (self.left_front_3d-400)/300 )

		self.right_front_3d = np.asarray([self.right_f_2d[0],self.right_f_2d[1],self.z*300+400])
		self.right_front_3d = self.inverse_operation( (self.right_front_3d-400)/300 )

		self.left_behind_3d = np.asarray([self.left_b_2d[0],self.left_b_2d[1],self.z*300+400])
		self.left_behind_3d = self.inverse_operation( (self.left_behind_3d-400)/300 )

		self.right_behind_3d = np.asarray([self.right_b_2d[0],self.right_b_2d[1],self.z*300+400])
		self.right_behind_3d = self.inverse_operation( (self.right_behind_3d-400)/300 )

		self.four  = np.array([self.left_front_3d])
		self.four = np.append(self.four,[self.right_behind_3d],axis=0)
		self.four = np.append(self.four,[self.left_behind_3d],axis=0)
		self.four = np.append(self.four,[self.right_front_3d],axis=0)

		return  self.left_b_2d,self.right_b_2d,self.right_f_2d,self.left_f_2d 
		        #返回左后，右后，右前，左前



	def Get2DPointFrom3D(self,ix,iy,iz):#从物体的3D点获得“俯视”的2D点


		Obj3DPoint = np.asarray([ix,iy,iz])
		C_Obj2DPoint = np.dot( Obj3DPoint,np.linalg.inv(self.matrix_Z) ) #矩阵运算

		return C_Obj2DPoint[0]*300+400,C_Obj2DPoint[1]*300+400 

	def Get2DPointFrom2D(self,ix,iy,depth_frame,depth_intrin):#从物体的2D点获得“俯视”的2D点
		Obj2DPoint = np.asarray([ix,iy])
	
		distance = depth_frame.get_distance(Obj2DPoint[0],Obj2DPoint[1])
		print("distance")
		print(distance)
		Obj3DPoint = rs.rs2_deproject_pixel_to_point(depth_intrin,[Obj2DPoint[0],Obj2DPoint[1]],distance)
		
		return self.Get2DPointFrom3D(Obj3DPoint[0],Obj3DPoint[1],Obj3DPoint[2])

	def GetFourDeskPoints3D_W(self):
		#return self.left_behind_3d,self.right_behind_3d,self.right_front_3d,self.left_front_3d
		
		return self.left_front_3d,self.left_behind_3d,self.right_front_3d,self.left_f_2d,self.left_b_2d,self.right_f_2d,self.right_b_2d

if __name__ == "__main__":
	pipeline = rs.pipeline()
	config = rs.config()
	#config.enable_device_from_file("/home/l/Documents/20201101_170810.bag")
	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

	profile = pipeline.start(config)
	aligner = rs.align(rs.stream.color)
	depth_sensor = profile.get_device().first_depth_sensor()
	color_sensor = profile.get_device().first_color_sensor()

	depth_sensor.set_option(rs.option.motion_range, 29)
	depth_sensor.set_option(rs.option.laser_power, 16)
	depth_sensor.set_option(rs.option.accuracy, 3)
	depth_sensor.set_option(rs.option.confidence_threshold, 12)
	depth_sensor.set_option(rs.option.filter_option, 5)


	while True:

		start = time.time()
		white = cv2.imread("aa.jpg")
		
		frame = pipeline.wait_for_frames()  # 等两张图像同时出现
		frame = aligner.process(frame)      # 深度图与彩色图对齐操作
		frameColor = frame.get_color_frame()  # 获得两张图像
		frameDepth = frame.get_depth_frame()

		#intrinColor = frameColor.profile.as_video_stream_profile().intrinsics        # 获得图像的内参，后面会用到
		intrinDepth = frameDepth.profile.as_video_stream_profile().intrinsics	

		pc = rs.pointcloud()
		points = pc.calculate(frameDepth)
		point_numpy = np.unique(points.get_vertices())
		
		a = Open3dCalc(point_numpy.tolist())
		a.planeSeg()
		a.toSameZ()

		left_b_2d,right_b_2d,right_f_2d,left_f_2d = a.GetFourDeskPoints(white)
		
		x,y = a.Get2DPointFrom2D(260,290,frameDepth,intrinDepth )	
		print(x,y)

		cv2.circle(white, tuple(left_b_2d ) ,  2, (0,0,255) )
		cv2.circle(white, tuple(right_b_2d) ,  2, (0,255,0) )
		cv2.circle(white, tuple(right_f_2d),  2, (255,0,0) )
		cv2.circle(white, tuple(left_f_2d ) ,  2, (0,0,0)   )


		cv2.circle(white,(round(x),round(y) ),3,(0,0,255) )
		cv2.waitKey(1)
		cv2.imshow("white",white)
		#print( a.GetFourDeskPoints3D() )
		left_behind_3d,right_behind_3d,right_front_3d,left_front_3d = a.GetFourDeskPoints()


		four = a.four

		print(four)

		#a.showStatic(four)


		print(time.time()-start)


