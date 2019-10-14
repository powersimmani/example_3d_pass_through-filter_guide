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

# colors for drawing different bodies 

class DepthRuntime(object):
    def __init__(self):
        pygame.init()
        self.save_flag_color = False
        self.save_flag_depth = False
        self.cnt =0 

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect_depth = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

        # back buffer surface for getting Kinect depth frames, 8bit grey, width and height equal to the Kinect color frame size
        self._frame_surface_depth = pygame.Surface((self._kinect_depth.depth_frame_desc.Width, self._kinect_depth.depth_frame_desc.Height), 0, 24)
        # here we will store skeleton data 
        self._bodies = None
        
        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen_depth = pygame.display.set_mode((self._kinect_depth.depth_frame_desc.Width, self._kinect_depth.depth_frame_desc.Height), 
                                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Depth")

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen_color = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect_color = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)


        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface_color = pygame.Surface((self._kinect_color.color_frame_desc.Width, self._kinect_color.color_frame_desc.Height), 0, 32)
        # here we will store skeleton data 




    def draw_color_frame(self, frame, out_path,frame_cnt):
        if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        a = np.reshape(frame,(1080,1920,4))
        a[:,:,0],a[:,:,1],a[:,:,2] = a[:,:,2].copy(),a[:,:,1].copy(),a[:,:,0].copy()
        im =  Image.fromarray(a).convert("RGB")
        im.save(out_path+"color_"+str(frame_cnt) + ".jpg")           

        

    def draw_depth_frame(self, frame, out_path,frame_cnt):
        if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        f8=np.uint8(frame.clip(1,4000)/16.)#clip을 이용해서 최소값을 0 최대 값을 250으로 바꾼다. 
        frame8bit=np.dstack((f8,f8,f8))# RGB값 전부 같은 값으로 바꾼다. 

        a = np.reshape(frame,(424,512))
        b = np.reshape(frame8bit,(424,512,3))

        im =  Image.fromarray(b).convert("RGB")
        im.save(out_path+"/"+"depth_"+str(frame_cnt) + ".jpg")

        with open(out_path+'depth_raw_'+str(frame_cnt)+".pck", 'wb') as f:
            pickle.dump(a, f)

    def stream_color_frame(self, frame, target_surface):
        if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        target_surface.lock()
        address = self._kinect_color.surface_as_array(target_surface.get_buffer())

        a = np.reshape(frame,(1080,1920,4))
        #a = cv2.resize(a, dsize=(512, 424), interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, dsize=(304, 228), interpolation=cv2.INTER_AREA)
        #a = cv2.flip(a,1)
        cv2.imshow("color",a)
        cv2.waitKey(1)

    def stream_depth_frame(self, frame, target_surface):
        if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        target_surface.lock()
        #f8=np.uint8(frame.clip(1,4000)/16.)#clip을 이용해서 최소값을 0 최대 값을 250으로 바꾼다. 
        #frame8bit=np.dstack((f8,f8,f8))# RGB값 전부 같은 값으로 바꾼다. 

        f82 = np.uint8((frame / frame.max())*255) 
        frame8bit2=np.dstack((f82,f82,f82))     
        a2 = np.reshape(frame8bit2,(424,512,3))   

        b = cv2.resize(a2, dsize=(304, 228), interpolation=cv2.INTER_AREA)
        b = cv2.applyColorMap(b, cv2.COLORMAP_JET)
        cv2.line(b,(152,0),(152,228),(0,0,0),1)
        cv2.line(b,(0,114),(304,114),(0,0,0),1)
        cv2.imshow("depth",b)
        cv2.waitKey(1)


    def stream_distance_frame(self,depth_frame, init_depth):
        init_depth = init_depth.astype(np.int16)
        b = np.reshape(init_depth-depth_frame,(424,512))
        b  = np.abs(b.astype(np.int16)).astype(np.uint16)
        b[b<10] =0
        b[b>1000] =0
        f82 = np.uint8((b / b.max())*255) 
        frame8bit2=np.dstack((f82,f82,f82))     
        b = np.reshape(frame8bit2,(424,512,3))   
        b = cv2.applyColorMap(b, cv2.COLORMAP_JET)
        cv2.imshow("diff",b)
        cv2.waitKey(1)


    def stream_background_frame(self,color_frame,back):
        if back is None:
            return
        color_frame = np.reshape(color_frame,(1080,1920,4))
        color_frame = cv2.resize(color_frame,(480, 270))
        # make a kernel of just 1s (averaging filter)
        kernel_size = np.ones((480, 270), np.float32) / (480*270)
        frame_gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        
        foreground, gamma = self.compare(frame_gray, back.astype(np.uint8))

        foreground = foreground.astype(np.uint8)
        background = back.astype(np.uint8)

        (score, diff) = compare_ssim(foreground, background, full=True)
        diff = (diff * 255).astype("uint8")


        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]       
        #final = cv2.filter2D(morph_img, -1, kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (32, 18))
        morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph_img = cv2.GaussianBlur(morph_img, (5,5) ,0)

        cv2.imshow('mask',morph_img)
        cv2.waitKey(1)
        
        #color_frame = self.adjust_gamma(color_frame, gamma = gamma)
        #fgmask = cv2.bitwise_and(color_frame, color_frame, mask = final)

        #cv2.imshow('mask',fgmask)
        #cv2.waitKey(1)

        return morph_img

    def rgb2float( self,r, g, b, a = 0 ):
        #code.interact(local = dict(globals(), **locals()))
        return struct.unpack('f', struct.pack('i',r << 16 | g << 8 | b))[0]

    def point_cloud(self,depth_frame,color_frame,RGB_type):
              
        # size must be what the kinect offers as depth frame
        L = depth_frame.size
        # create C-style Type
        TYPE_CameraSpacePoint_Array = PyKinectV2._CameraSpacePoint * L

        # instance of that type
        csps = TYPE_CameraSpacePoint_Array()
        # cstyle pointer to our 1d depth_frame data
        ptr_depth = np.ctypeslib.as_ctypes(depth_frame.flatten())
        # calculate cameraspace coordinates
        error_state = self._kinect_depth._mapper.MapDepthFrameToCameraSpace(L, ptr_depth,L, csps)
        # 0 means everythings ok, otherwise failed!
        if error_state:
            raise "Could not map depth frame to camera space! "
            + str(error_state)


        start_time = datetime.now()
        # convert back from ctype to numpy.ndarray
        pf_csps = ctypes.cast(csps, ctypes.POINTER(ctypes.c_float))
        position_data = np.copy(np.ctypeslib.as_array(pf_csps, shape=(L,3)))
        color_frame = np.reshape(color_frame,(1080,1920,4))


        color_data = np.zeros(shape=(L,3), dtype=np.int)
        color_position = np.zeros(shape=(L,2), dtype=np.float)

        null_cnt = [] 
        #위치상 무한대 제거 

        for index in range(L):
            a = self._kinect_depth._mapper.MapCameraPointToColorSpace(csps[index]) 
            color_position[index][0] = a.y;color_position[index][1] = a.x

        x_range = np.logical_and(color_position[:,1] >= 0, color_position[:,1] <=1799.4)
        y_range = np.logical_and(color_position[:,0] >= 0, color_position[:,0] <=1079.4)
        color_pos_range = np.logical_and(x_range,y_range)


        position_data = position_data[color_pos_range]
        color_mapper =np.rint(color_position[color_pos_range]).astype(int)

        #inf아니면서 color위치가 음수가 아닌 경우
        #code.interact(local = dict(globals(), **locals()))
        #color_data = np.asarray([color_frame[y][x][:3] for y,x in color_mapper])
        #color_data = np.asarray([color_frame[y][x][2] << 16 | color_frame[y][x][1] << 8 | color_frame[y][x][0] for y,x in color_mapper])
        #takes too much time
        color_data = np.asarray([self.rgb2float(color_frame[y][x][2],color_frame[y][x][1],color_frame[y][x][0]) for y,x in color_mapper])
        end_time = datetime.now()
        print('3d coloring: {}'.format(end_time - start_time))
        del pf_csps, csps, ptr_depth, TYPE_CameraSpacePoint_Array
        return position_data,color_data

    def put_boudnary_points(self,dic):
        new_col = []
        new_pos = []
        x_start,x_end = dic["x"]
        y_start,y_end = dic["y"]
        z_start,z_end = dic["z"]

        x_points,y_points,z_points = np.asarray(np.arange(x_start,x_end,0.01)),np.asarray(np.arange(y_start,y_end,0.01)),np.asarray(np.arange(z_start,z_end,0.01))
        
        y_starts,y_ends = np.asarray(np.full((len(x_points)),y_start)),np.asarray(np.full((len(x_points)),y_end))
        z_starts,z_ends = np.asarray(np.full((len(x_points)),z_start)),np.asarray(np.full((len(x_points)),z_end))
        lines_x = np.concatenate((np.vstack((x_points,y_starts,z_starts)).T,np.vstack((x_points,y_ends,z_starts)).T,np.vstack((x_points,y_starts,z_ends)).T,np.vstack((x_points,y_ends,z_ends)).T))


        x_starts,x_ends = np.asarray(np.full((len(y_points)),x_start)),np.asarray(np.full((len(y_points)),x_end))
        z_starts,z_ends = np.asarray(np.full((len(y_points)),z_start)),np.asarray(np.full((len(y_points)),z_end))
        lines_y = np.concatenate((np.vstack((x_starts,y_points,z_starts)).T,np.vstack((x_ends,y_points,z_starts)).T,np.vstack((x_starts,y_points,z_ends)).T,np.vstack((x_ends,y_points,z_ends)).T))


        x_starts,x_ends = np.asarray(np.full((len(z_points)),x_start)),np.asarray(np.full((len(z_points)),x_end))
        y_starts,y_ends = np.asarray(np.full((len(z_points)),y_start)),np.asarray(np.full((len(z_points)),y_end))
        lines_z = np.concatenate((np.vstack((x_starts,y_starts,z_points)).T,np.vstack((x_ends,y_starts,z_points)).T,np.vstack((x_starts,y_ends,z_points)).T,np.vstack((x_ends,y_ends,z_points)).T))

        lines_x_color =  np.full((len(lines_x)),self.rgb2float(255,0,0))
        lines_y_color =  np.full((len(lines_y)),self.rgb2float(0,255,0))
        lines_z_color =  np.full((len(lines_z)),self.rgb2float(0,0,255))

        return np.concatenate((lines_x,lines_y,lines_z)),np.asmatrix(np.concatenate((lines_x_color,lines_y_color,lines_z_color))).T

    def point_cloud_preprocessing(self, position_data,color_data):

        
        print('Cloud before filtering: ')        
        dic = {"x":[-0.3,0.5],
                "y":[-0.3,0.5],
                "z":[0.5,1.0]}
        
        #for concrete filtering 
        #new_pos, new_col = self.put_boudnary_points(dic)
        #new_data = np.concatenate((new_pos, new_col),axis = 1)


        #data[i][3] = 255 << 16 | 255 << 8 | 0

        for key in dic:
            start,end = dic[key]

        #code.interact(local = dict(globals(), **locals()))
        data = np.concatenate((position_data, np.expand_dims(color_data,axis = 1)),axis = 1)

        #data = np.asarray(np.concatenate((data,new_data),axis = 0))
        cloud = pcl.PointCloud_PointXYZRGB()
        cloud.from_list(data)
        name = ["x","y","z"]
        for i in range(3):
            print(name[i],"\t",data[:,i].min(),"\t",data[:,i].max())
        
        #전체영상
        #visual = pcl.pcl_visualization.CloudViewing()
        #visual.ShowColorCloud(cloud)



        def passT(cloud,dic):
            for key in dic:
                passthrough = cloud.make_passthrough_filter()            
                passthrough.set_filter_field_name(key)        
                passthrough.set_filter_limits(dic[key][0], dic[key][1])
                cloud = passthrough.filter()
            return cloud

        cloud_filtered = passT(cloud,dic)

        #PassT 후 영상 
        #visual = pcl.pcl_visualization.CloudViewing();visual.ShowColorCloud(cloud_filtered)
        #visual.ShowColorCloud(cloud_filtered)

        def segmentation(cloud_filtered):
            seg = cloud_filtered.make_segmenter()
            seg.set_optimize_coefficients(True)
            seg.set_model_type(pcl.SACMODEL_PLANE)
            seg.set_method_type(pcl.SAC_RANSAC)
            seg.set_distance_threshold(0.01)
        #seg.set_max_iterations(100)
            return seg.segment()

        data = cloud_filtered.to_list()
        inliers, model = segmentation(cloud_filtered)
        data = np.delete(data,inliers,axis = 0)
        cloud_filtered.from_list(data)
        a = model 
        data = cloud_filtered.to_list()
        inliers, plane = segmentation(cloud_filtered)
        data = np.delete(data,inliers,axis = 0)
        cloud_filtered.from_list(data)
        
        #visual = pcl.pcl_visualization.CloudViewing();visual.ShowColorCloud(cloud_filtered)
        #plane변수에서 평면을 coeff 곱해서 양수면 취하고 음수면 취하지 않는다.  
        
        ones = np.expand_dims(np.ones(len(data[:,:3])),axis = 1)
        
        xyzd = np.concatenate((data[:,:3],ones),axis =1)
        
        plane_pos = np.sum(xyzd*plane,axis= 1)
        np.sum(xyzd*plane,axis= 1)
        t = plane_pos<0
        t2 = plane_pos>=0
        data = data[t]
        cloud_filtered.from_list(data)
        #code.interact(local = dict(globals(), **locals()))
        #visual = pcl.pcl_visualization.CloudViewing();visual.ShowColorCloud(cloud_filtered)

        #plane 추출 영상 
        #visual = pcl.pcl_visualization.CloudViewing();visual.ShowColorCloud(cloud_filtered)
        #code.interact(local = dict(globals(), **locals()))    

        return cloud_filtered
        #data = cloud_filtered.to_list()
        #Create the filtering object - StatisticalOutlierRemoval filter
        fil = pcl.StatisticalOutlierRemovalFilter_PointXYZRGB(cloud_filtered)
        fil.set_mean_k(50)
        fil.set_std_dev_mul_thresh(1.0)
        cloud_filtered  = fil.filter()    
        

        #for i in inliers:
        #    data[i][3] = 255 << 16 | 255 << 8 | 0

        # get normal of the plane
        """
        kdtree = pcl.KdTree(cloud2)
        ne = pcl.NormalEstimation(cloud2)
        ne.set_RadiusSearch(0.05)
        ne.set_SearchMethod(kdtree)
        normal = ne.compute()
        """
        temp = np.asarray(cloud.to_list())[:,:3]
        cloud2 = pcl.PointCloud();cloud2.from_list(temp)
        ne = cloud2.make_NormalEstimation()
        tree = cloud2.make_kdtree()
        ne.set_SearchMethod(tree)
        ne.set_KSearch(50)
        cloud_normals = ne.compute()        
        #code.interact(local = dict(globals(), **locals()))


    def run(self):
        status = "init"
        out_path = "temp/"
        frame_cnt =0 
        object_cnt =0
        depth_frame = None
        color_frame = None
        init_depth = None
        init_color = None
        object_cnt = 0
        #pcd = o3d.geometry.PointCloud()
        pcd = o3d.io.read_point_cloud("temp.ply")
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=960,height=540)
        vis.add_geometry(pcd)
        a  = vis.get_view_control()
        a.rotate(900,0)
        start_time = time.time()
        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                elif event.type == pygame.KEYDOWN:
                    self.save_flag_color = True              
                    self.save_flag_depth = True              

            
            # --- Getting frames and drawing  
            self.cnt = self.cnt + 1;
            if self._kinect_depth.has_new_depth_frame():
                depth_frame = self._kinect_depth.get_last_depth_frame()
                self.stream_depth_frame(depth_frame, self._frame_surface_depth)
                #depth_frame = None

            if self._kinect_color.has_new_color_frame():
                color_frame = self._kinect_color.get_last_color_frame()
                self.stream_color_frame(color_frame, self._frame_surface_color)


            if depth_frame is not None and color_frame is not None:
                #position_data,color_data = self.point_cloud(depth_frame,color_frame,True)
                #pcd.points = o3d.utility.Vector3dVector(position_data)
                #pcd.colors = o3d.utility.Vector3dVector(np.flip(color_data.astype(np.float)/255,axis=1))
                

                
                position_data,color_data = self.point_cloud(depth_frame,color_frame,False)

                start_time = datetime.now()
                a = self.point_cloud_preprocessing(position_data,color_data)
                end_time = datetime.now()
                print('Duration pointcloud processing: {}'.format(end_time - start_time))


                object_out_path = out_path + str(object_cnt)+"/"
                ply_out_path = object_out_path + "point_cloud_"+str(frame_cnt) + ".pcd"
                os.makedirs(object_out_path,exist_ok=True)

                #code.interact(local = dict(globals(), **locals()))
                pcl.save(a,ply_out_path)
                start_time = time.time()


                #o3d for point streaming
                #vis.update_geometry()
                #vis.poll_events()
                #vis.update_renderer()                

                #pcl python
                #visual = pcl.pcl_visualization.CloudViewing()
                #visual.ShowColorCloud(a)

                #o3d.io.write_point_cloud(ply_out_path, pcd,write_ascii=True)
                #self.draw_depth_frame(depth_frame,object_out_path,frame_cnt)
                self.draw_color_frame(color_frame ,object_out_path,frame_cnt)
                frame_cnt += 1
        
            depth_frame = None
            color_frame = None

            if self.save_flag_depth ==True or self.save_flag_color ==True:
                if self._kinect_depth.has_new_depth_frame():
                    frame = self._kinect_depth.get_last_depth_frame()
                    self.draw_depth_frame(frame,out_path)
                    frame = None
                    self.save_flag_depth = False

                if self._kinect_color.has_new_color_frame():
                    frame = self._kinect_color.get_last_color_frame()
                    self.draw_color_frame(frame ,out_path)
                    frame = None      
                    self.save_flag_color = False
                
            # --- Limit to 60 frames per second
            self._clock.tick(60)
            pygame.display.update()
        # Close our Kinect sensor, close the window and quit.
        pygame.quit()


__main__ = "Kinect v2 Depth"
global cnt
cnt = 0
game = DepthRuntime();
game.run();