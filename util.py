from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import pickle
import code,os,subprocess
import ctypes
import _ctypes
import pygame
import sys,cv2, torch,decimal    
import numpy as np
import open3d as o3d
from PIL import Image
from skimage.measure import compare_ssim
from torch.autograd import Variable
if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread
import pcl,random
import pcl.pcl_visualization
import struct
import time
from datetime import datetime


class Filter3D():
    def __init__(self, is_rgb_4byte = False):
        self.is_rgb_4byte = is_rgb_4byte
        

    def rgb2float( self,r, g, b, a = 0 ):
        return struct.unpack('f', struct.pack('i',r << 16 | g << 8 | b))[0]

    def draw_guild_lines(self,dic, density = 0.01):
        new_col = []
        new_pos = []
        x_start,x_end = dic["x"]
        y_start,y_end = dic["y"]
        z_start,z_end = dic["z"]

        x_points,y_points,z_points = np.asarray(np.arange(x_start,x_end,density)),np.asarray(np.arange(y_start,y_end,density)),np.asarray(np.arange(z_start,z_end,density))
        
        y_starts,y_ends = np.asarray(np.full((len(x_points)),y_start)),np.asarray(np.full((len(x_points)),y_end))
        z_starts,z_ends = np.asarray(np.full((len(x_points)),z_start)),np.asarray(np.full((len(x_points)),z_end))
        lines_x = np.concatenate((np.vstack((x_points,y_starts,z_starts)).T,np.vstack((x_points,y_ends,z_starts)).T,np.vstack((x_points,y_starts,z_ends)).T,np.vstack((x_points,y_ends,z_ends)).T))


        x_starts,x_ends = np.asarray(np.full((len(y_points)),x_start)),np.asarray(np.full((len(y_points)),x_end))
        z_starts,z_ends = np.asarray(np.full((len(y_points)),z_start)),np.asarray(np.full((len(y_points)),z_end))
        lines_y = np.concatenate((np.vstack((x_starts,y_points,z_starts)).T,np.vstack((x_ends,y_points,z_starts)).T,np.vstack((x_starts,y_points,z_ends)).T,np.vstack((x_ends,y_points,z_ends)).T))


        x_starts,x_ends = np.asarray(np.full((len(z_points)),x_start)),np.asarray(np.full((len(z_points)),x_end))
        y_starts,y_ends = np.asarray(np.full((len(z_points)),y_start)),np.asarray(np.full((len(z_points)),y_end))
        lines_z = np.concatenate((np.vstack((x_starts,y_starts,z_points)).T,np.vstack((x_ends,y_starts,z_points)).T,np.vstack((x_starts,y_ends,z_points)).T,np.vstack((x_ends,y_ends,z_points)).T))

        if (self.is_rgb_4byte):
            lines_x_color =  np.full((len(lines_x)),self.rgb2float(255,0,0))#blue for x
            lines_y_color =  np.full((len(lines_y)),self.rgb2float(0,255,0))#green for y
            lines_z_color =  np.full((len(lines_z)),self.rgb2float(0,0,255))#red for z
            return np.concatenate((lines_x,lines_y,lines_z)),np.asmatrix(np.concatenate((lines_x_color,lines_y_color,lines_z_color))).T
        else:
            lines_x_color = np.zeros((len(lines_x),3))
            lines_y_color = np.zeros((len(lines_y),3))
            lines_z_color = np.zeros((len(lines_z),3))

            lines_x_color[:,0] = 1.0 #red for x
            lines_y_color[:,1] = 1.0 #green for y
            lines_z_color[:,2] = 1.0 #blue for z
            return np.concatenate((lines_x,lines_y,lines_z)),np.asmatrix(np.concatenate((lines_x_color,lines_y_color,lines_z_color)))        

    def pass_through_filter(self, dic, pcd):

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        points[:,0]
        x_range = np.logical_and(points[:,0] >= dic["x"][0] ,points[:,0] <= dic["x"][1])
        y_range = np.logical_and(points[:,1] >= dic["y"][0] ,points[:,1] <= dic["y"][1])
        z_range = np.logical_and(points[:,2] >= dic["z"][0] ,points[:,2] <= dic["z"][1])

        pass_through_filter = np.logical_and(x_range,np.logical_and(y_range,z_range))

        pcd.points = o3d.utility.Vector3dVector(points[pass_through_filter])
        pcd.colors = o3d.utility.Vector3dVector(colors[pass_through_filter])

        return pcd


if __name__ == "__main__":

    pcd = o3d.io.read_point_cloud("point_cloud_0.pcd")

    dic = {"x":[-0.3,0.5],
            "y":[-0.3,0.5],
            "z":[0.5,1.0]}
    
    filter3D = Filter3D()

    #Drawing filter guidelines 
    new_pos, new_col = filter3D.draw_guild_lines(dic)
    new_data = np.concatenate((new_pos, new_col),axis = 1)
    guild_points = o3d.geometry.PointCloud()
    guild_points.points = o3d.utility.Vector3dVector(new_pos)
    guild_points.colors = o3d.utility.Vector3dVector(new_col)

    #Filtering
    pcd_filtered = filter3D.pass_through_filter(dic,pcd)


    #visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960,height=540)
    vis.add_geometry(pcd_filtered)
    #vis.add_geometry(guild_points)

    a  = vis.get_view_control()
    a.rotate(1050,0)
    vis.run()
    vis.destroy_window()

    code.interact(local = dict(globals(), **locals()))
