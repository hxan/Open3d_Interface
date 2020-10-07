import open3d 
import numpy as np
import time

class Open3dCalc:
	def __init__(self, xyz):#初始化，传入一个100*3的矩阵（numpy类型 ）
		self.xyz=xyz#
		self.pcd=open3d.geometry.PointCloud()#生成点云
		self.pcd.points = open3d.utility.Vector3dVector(xyz)
		self.pcd = self.pcd.voxel_down_sample(voxel_size=0.005)
		self.axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

	def __del__(self):
		pass

	def planeSeg(self):#平面分割，在初始化之后调用，返回一个n*3的矩阵
		self.plane_model, self.inliers = self.pcd.segment_plane(distance_threshold=0.015	,
                                         ransac_n=400,
                                         num_iterations=2000)


		[a, b, c, d] = self.plane_model
		print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

		self.inlier_cloud = self.pcd.select_by_index(self.inliers)
		self.inlier_cloud.paint_uniform_color([0, 1, 0])
		
		self.outlier_cloud = self.pcd.select_by_index(self.inliers, invert=True)
		self.outlier_cloud.paint_uniform_color([0, 0, 0])

		return np.asarray(self.inlier_cloud.points)


	def orienPlane(self):#返回平面接矩形的点集,返回八个点 8x3矩阵
		self.aabb_o = self.inlier_cloud.get_oriented_bounding_box()
		self.aabb_o.color=(0,0,0)

		return np.asarray( self.aabb_o.get_box_points() )
	def destPoints(self):# 返回桌角
		t=a.orienPlane()
		m,n,q,p=(t[0]+t[2])/2,(t[4]+t[2])/2,(t[1]+t[4])/2,(t[1]+t[0])/2
		np_point=[m]
		np_point=np.append(np_point,[n],axis=0)
		np_point=np.append(np_point,[q],axis=0)
		np_point=np.append(np_point,[p],axis=0)
		return np_point

	def axisPlane(self):#返回平面接矩形的点集,八个点，同上
		self.aabb_a = self.inlier_cloud.get_axis_aligned_bounding_box()
		self.aabb_a.color=(1,1,1)

		return np.asarray( self.aabb_a.get_box_points() )

	def showStatic(self,_xyz):#可视化，传入一个n*3的矩阵,这个函数是阻塞的
		show_pcd=open3d.geometry.PointCloud()

		show_pcd.points = open3d.utility.Vector3dVector(_xyz)
		open3d.visualization.draw_geometries([show_pcd,self.inlier_cloud,self.outlier_cloud])
    
	def hull(self):
		self.hull, _ = self.inlier_cloud.compute_convex_hull()
		self.hull_ls = open3d.geometry.LineSet.create_from_triangle_mesh(self.hull)
		self.hull_ls.paint_uniform_color((1, 0, 0))


		np_point=np.asarray(self.hull_ls.points)
		print(np_point[1]-np_point[0])
		len_np_point=len(np_point)-1

		for i in range(0,len_np_point ):
			unit=np_point[i+1]-np_point[i]
			j=0
			while j<1:
				t=np_point[i]+j*unit
				#print(t)
				np_point=np.append(np_point,[t],axis=0)
				#print(len(np_point))
				j+=0.01

		return np_point
	
dets= np.loadtxt('foo.txt',delimiter=',')


total=[None] * 102
pcd_file = open3d.io.read_point_cloud("E://15fuck.ply")

total[0]=Open3dCalc(dets )
total[0].planeSeg()
a=Open3dCalc(dets )
a.planeSeg()
a.showStatic(a.destPoints())


