#!/usr/bin/env python3

import sys
import os
import numpy as np
import quaternion
import yaml
import pandas as pd


import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from std_msgs.msg import Float32
from builtin_interfaces.msg import Time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageOps 
from graphviz import Graph
from copy import copy, deepcopy
import time


# sys.path.append(os.path.dirname(os.path.abspath('src/task_4/task_4/classes.py')))
# from classes import Map, Queue, Node, Tree, MapProcessor, AStar


class Navigation(Node):
    """! Navigation node class.
    This class should serve as a template to implement the path planning and
    path follower components to move the turtlebot from position A to B.
    """

    def __init__(self, node_name='Navigation'):
        """! Class constructor.
        @param  None.
        @return An instance of the Navigation class.
        """
        super().__init__(node_name)
        # Path planner/follower related variables
        self.path = Path()
        self.path_node_frame = []
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0

        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/robot/amcl_pose', self.__ttbot_pose_cbk, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, '/global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/robot/cmd_vel', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time',10) #DO NOT MODIFY

        # Node rate
        # self.rate = self.create_rate(10)
        
        # read map
        # self.mp = MapProcessor('/home/albert/ros2_ws/src/task_4/maps/sync_classroom_map')
        self.mp = MapProcessor('/home/albert/ros2_ws/src/task_4/maps/classroom_map')
        # self.kr = self.mp.rect_kernel(10,1)      # can adjust the kernel size
        self.kr = self.mp.rect_kernel(5,1)      # can adjust the kernel size
        self.mp.inflate_map(self.kr,True)

        self.mp.get_graph_from_map()

        # fig, ax = plt.subplots(dpi=100)
        # plt.imshow(self.mp.inf_map_img_array)
        # plt.colorbar()
        # plt.show()
        
        # PID parameters
        # self.prev_error=0
        # self.integral=0
        # self.dt=0.1    
        self.prevHeadingError = 0
        self.prevIdx = 0
        

    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        # self.get_logger().info(
            # 'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))
        # print('goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))
        
    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose = data.pose
        # self.get_logger().info(
            # 'ttbot_pose: {:.4f}, {:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))
        # print('ttbot_pose: {:.4f}, {:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))

    
    def a_star_path_planner(self, start_pose, end_pose):        # should be in the Node frame
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        
        # start planning the path
        path_Rviz_frame = Path()
        self.get_logger().info(
            'A* planner.\n> start: {},\n> end: {}'.format(start_pose.pose.position, end_pose.pose.position))
        self.start_time = self.get_clock().now().nanoseconds*1e-9 #Do not edit this line (required for autograder)
        # TODO: IMPLEMENTATION OF THE A* ALGORITHM
        
        # First convert the start pose and goal pose from Rviz frame to Node frame 
        
        # (start_pose.pose.position.x, start_pose.pose.position.y) = self.convertFrames(start_pose.pose.position.x, start_pose.pose.position.y, 'RvizToNode')
        (start_pose_x, start_pose_y) = self.convertFrames(start_pose.pose.position.x, start_pose.pose.position.y, 'RvizToNode')
        (end_pose_x, end_pose_y) = self.convertFrames(end_pose.pose.position.x, end_pose.pose.position.y, 'RvizToNode')   

        # node name should be in the form of (y,x)
        
        self.mp.map_graph.root = f'{start_pose_y},{start_pose_x}'
        self.mp.map_graph.end = f'{end_pose_y},{end_pose_x}'
        start = float(self.get_clock().now().nanoseconds*1e-9)
        aStar = AStar(self.mp.map_graph)
        aStar.solve(self.mp.map_graph.g[self.mp.map_graph.root],self.mp.map_graph.g[self.mp.map_graph.end])
        path_node_frame = aStar.reconstruct_path(sn = self.mp.map_graph.g[self.mp.map_graph.root], en = self.mp.map_graph.g[self.mp.map_graph.end], Path = path_Rviz_frame)
        path_arr_as = self.mp.draw_path(path_node_frame)
        end = float(self.get_clock().now().nanoseconds*1e-9)
        print(f'Astar calculation time: {end - start}')
        fig, ax = plt.subplots()  # Create a single plot
        ax.imshow(path_arr_as)
        ax.set_title('Path A*')

        plt.show()
        
        # Do not edit below (required for autograder)
        self.astarTime = Float32()
        self.astarTime.data = float(self.get_clock().now().nanoseconds*1e-9-self.start_time)
        self.calc_time_pub.publish(self.astarTime)
        
        return path_Rviz_frame, path_node_frame

    def get_path_idx(self, path, vehicle_pose):         # should be in the Rviz frame
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  current_goal_pose     PoseStamped object containing the current vehicle position.
        @return idx                   Position in the path pointing to the next goal pose to follow.
        """
        idx = -1
        dist=np.Inf
        threshold = 0.5
        idxRange = 23
        idxRadius = 0.5
        # idxRadius = 1.9
        
        # convert the vehicle_pose from Rviz frame to Node frame
        # (vehicle_pose_x, vehicle_pose_y) = self.convertFrames(vehicle_pose.pose.position.x, vehicle_pose.pose.position.y, 'RvizToNode')

        # TODO: IMPLEMENT A MECHANISM TO DECIDE WHICH POINT IN THE PATH TO FOLLOW idx <= len(path)
        # for i in range(len(path.poses)):
        #     # poses in the path are also in the Node frame
        #     # convert them into Rviz frame
        #     # (temp_x, temp_y) = self.convertFrames(path.poses[i].pose.position.x, path.poses[i].pose.position.y, 'NodeToRviz')
        #     # temp_dist=np.sqrt((vehicle_pose.pose.position.x-temp_x)**2+(vehicle_pose.pose.position.y-temp_y)**2)         
            
        #     temp_dist=np.sqrt((vehicle_pose.pose.position.x-path.poses[i].pose.position.x)**2+(vehicle_pose.pose.position.y-path.poses[i].pose.position.y)**2)     
               
        #     if temp_dist < dist:
        #         idx = i
        #         dist = temp_dist                              
                
        # if idx == len(path.poses) - 1 or idx + idxRange > len(path.poses) - 1:
        #     idx = len(path.poses) - 1
            
        # elif idx < len(path.poses) - 1:
        #     if dist <= threshold:
        #         idx += idxRange
                
                
        for i in range(self.prevIdx,len(path.poses)):
            
            if (np.sqrt((vehicle_pose.pose.position.x-path.poses[i].pose.position.x)**2+(vehicle_pose.pose.position.y-path.poses[i].pose.position.y)**2)) >= idxRadius:
                idx = i
                self.prevIdx = i
                break
        
        if idx == -1:
            idx = len(path.poses) - 1
        
        return idx
    
    def quaternion_to_yaw(self, q): 
        q_length = np.sqrt( q.x ** 2 + q.y ** 2 + q.z ** 2 + q.w ** 2) 
        qx_norm = q.x / q_length
        qy_norm = q.y / q_length
        qz_norm = q.z / q_length
        qw_norm = q.w / q_length
        numerator = 2 * (qw_norm * qz_norm + qx_norm * qy_norm)
        denominator = 1 - 2 * (qy_norm ** 2 + qz_norm ** 2)
        yaw = np.arctan2( numerator ,denominator)
        return yaw


    def path_follower(self, vehicle_pose, current_goal_pose, prev_time):       # should be in the Rviz frame
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return path                   Path object containing the sequence of waypoints of the created path.
        """
        # TODO: IMPLEMENT PATH FOLLOWER
        
        # pid parameters
        # dont delete the following parameters
        kp_speed = 0.15
        kp_heading = 0.3
        kd_heading = 0.15
        
        # testing parameters
        # kp_speed = 0.12
        # kp_heading = 0.3
        # kd_heading = 0.4
        
        current_time = self.get_clock().now().nanoseconds*1e-9
        
        error=np.sqrt(( vehicle_pose.pose.position.x - current_goal_pose.pose.position.x) ** 2 + ( vehicle_pose.pose.position.y - current_goal_pose.pose.position.y) ** 2)
                 
        speed = kp_speed * error
        
        q_vehicle = vehicle_pose.pose.orientation
        
        # heading should be in radian
        # heading_vehicle_yaw = np.quaternion(q_vehicle)
        heading_vehicle_yaw = self.quaternion_to_yaw(q_vehicle)
        print(f'heading_vehicle_yaw: {heading_vehicle_yaw * 180 / np.pi }')
        
        heading_difference_vehicl_goal = np.arctan2(current_goal_pose.pose.position.y - vehicle_pose.pose.position.y , current_goal_pose.pose.position.x - vehicle_pose.pose.position.x )
        print(f'heading_difference_vehicl_goal: {heading_difference_vehicl_goal * 180 / np.pi }')

        heading_error = heading_difference_vehicl_goal - heading_vehicle_yaw
        
        print(f'heading_error: {heading_error * 180 / np.pi }')
        
        if heading_error > np.pi:
            heading_error = -2 * np.pi + heading_error
        elif heading_error < -np.pi:
            heading_error = 2 * np.pi + heading_error
        
        print(f'heading_error: {heading_error * 180 / np.pi }')
        

        
        print(f'speed: {speed}')


        
        heading = kp_heading * heading_error + kd_heading* (heading_error - self.prevHeadingError) / (current_time - prev_time)
        
        prev_time = current_time
        self.prevHeadingError = heading_error
        

        print(f'heading: {heading * 180 / np.pi}')
                
        return speed, heading

    def move_ttbot(self, speed, heading, prev_heading):           # should be in the Rviz frame
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired speed.
        @param  heading   Desired yaw angle.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()
        # TODO: IMPLEMENT YOUR LOW-LEVEL CONTROLLER
        # speed=min(0.15, max(-0.15,speed))

        if abs(prev_heading) >= (np.pi / 20) or abs(heading) >= (np.pi / 20):
            speed = 0.0
            prev_heading = heading
            # heading = heading / np.abs(heading) * np.pi / 20
            heading = heading / 5
        

        print(f'speed: {speed}')
        print(f'heading: {heading * 180 / np.pi}')

        # speed = 0.1
        
        cmd_vel.linear.x = speed
        cmd_vel.angular.z = heading

        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        @param none
        @return none
        """
        prev_heading = 0
        prev_time = 0
        while rclpy.ok():
            # Call the spin_once to handle callbacks
            # time.sleep(5)
            rclpy.spin_once(self, timeout_sec=0.1)  # Process callbacks without blocking
            print('launch successfully')
            
           
            # if self.ttbot_pose.pose.position == self.goal_pose.pose.position:
                # return
            
            
            # print(f'ttbot_pose: {self.ttbot_pose.pose.position}')
            # print(f'goal_pose: {self.goal_pose.pose.position}')
          
            
            # 1. Create the path to follow
            if (self.goal_pose.pose.position.x == 0.0 and self.goal_pose.pose.position.y == 0.0 and self.goal_pose.pose.position.z == 0.0):
                print('if')
                pass
            
            elif len(self.path.poses) <= 2:
                print('elif')
                # self.path should be in the Rviz frame
                self.path, self.path_node_frame = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)      
                # convert self.path into Rviz frame and also store the path (node frame) in self.path_node_frame
                for i in self.path.poses:
                    (i.pose.position.x, i.pose.position.y) = self.convertFrames(i.pose.position.x, i.pose.position.y, 'NodeToRviz')
                self.path_pub.publish(self.path)
            
            else:
                # 2. Loop through the path and move the robotq
                print('else')
                idx = self.get_path_idx(self.path, self.ttbot_pose)
                print('idx: {:}'.format(idx))
                current_goal = self.path.poses[idx]
                # current_goal_pose = self.convertFrames(current_goal.pose.position.x, current_goal.pose.position.y, 'NodeToRviz')
                print(f'current goal pose: {current_goal.pose.position}')
                # print(f'vehicle pose: {self.ttbot_pose}')
                
                speed, heading = self.path_follower(self.ttbot_pose, current_goal, prev_time)
                self.move_ttbot(speed, heading, prev_heading)
                
            print('before sleep') 
            # self.rate.sleep()
            time.sleep(1)
            print('after sleep')
            # Sleep for the rate to control loop timing
    # # for sync_class_map  
    # def convertFrames(self,x, y, args = None):
    #     # origin of node (0,0) top left 
    #     # origin of Rviz (-5.39,-6.29)
    #     # self.mp.map.map_im._size()
    #     # self.mp.map.map_df
    #     # self.mp.map.limits
    #     if args == 'RvizToNode':
    #         x = 0 + ((x + 5.39) / (299 * self.mp.map.map_df.resolution[0])) * self.mp.map.map_im._size[0]
    #         x = max( min(round(x), self.mp.map.map_im._size[0]-1), 0)
    #         y = 0 + (1 - (y + 6.29) / (210 * self.mp.map.map_df.resolution[0])) * self.mp.map.map_im._size[1]
    #         y = max( min(round(y), self.mp.map.map_im._size[1]-1), 0)
    #         return x, y
        
    #     elif args == 'NodeToRviz':
    #         x = -5.39 + ((x - 0) / self.mp.map.map_im._size[0]) * 299 * self.mp.map.map_df.resolution[0]
    #         x = max(min(round(x, 2), -5.39 + 299 * self.mp.map.map_df.resolution[0]), -5.39)
    #         y = -6.29 + (1- (y / self.mp.map.map_im._size[1])) * 210 * self.mp.map.map_df.resolution[0]
    #         y = max(min(round(y, 2), -6.29 + 210 * self.mp.map.map_df.resolution[0]), -6.29)
    #         return x, y
    #     else:
    #         pass
    
    # for class_map  
    def convertFrames(self,x, y, args = None):
        # origin of node (0,0) top left 
        # origin of Rviz (-5.39,-6.29)
        # self.mp.map.map_im._size()
        # self.mp.map.map_df
        # self.mp.map.limits
        if args == 'RvizToNode':
            x = 0 + ((x + 5.4) / (63 * self.mp.map.map_df.resolution[0])) * self.mp.map.map_im._size[0]
            x = max( min(round(x), self.mp.map.map_im._size[0]-1), 0)
            y = 0 + (1 - (y + 18.9) / (63 * self.mp.map.map_df.resolution[0])) * self.mp.map.map_im._size[1]
            y = max( min(round(y), self.mp.map.map_im._size[1]-1), 0)
            return x, y
        
        elif args == 'NodeToRviz':
            x = -5.4 + ((x - 0) / self.mp.map.map_im._size[0]) * 63 * self.mp.map.map_df.resolution[0]
            x = max(min(round(x, 2), -5.4 + 63 * self.mp.map.map_df.resolution[0]), -5.4)
            y = -18.9 + (1- (y / self.mp.map.map_im._size[1])) * 63 * self.mp.map.map_df.resolution[0]
            y = max(min(round(y, 2), -18.9 + 63 * self.mp.map.map_df.resolution[0]), -18.9)
            return x, y
        else:
            pass

class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)
    
    def __repr__(self):
        # fig, ax = plt.subplots(dpi=150)
        # ax.imshow(self.image_array,extent=self.limits, cmap=cm.gray)
        # ax.plot()
        return ""
        
    def __open_map(self,map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        # map_name = map_df.image[0]
        map_dir = os.path.dirname(map_name+'.pgm')
        image_path = os.path.join(map_dir, map_df.image[0])
        im = Image.open(image_path)
        size = 200, 200
        im.thumbnail(size)
        im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin,xmax,ymin,ymax]

    def __get_obstacle_map(self,map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i,j] = 0
        return img_array
    
    
class Queue():
    def __init__(self, init_queue = []):
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue)-1
    
    def __len__(self):
        numel = len(self.queue)
        return numel
    
    def __repr__(self):
        q = self.queue
        tmpstr = ""
        for i in range(len(self.queue)):
            flag = False
            if(i == self.start):
                tmpstr += "<"
                flag = True
            if(i == self.end):
                tmpstr += ">"
                flag = True
            
            if(flag):
                tmpstr += '| ' + str(q[i]) + '|\n'
            else:
                tmpstr += ' | ' + str(q[i]) + '|\n'
            
        return tmpstr
    
    def __call__(self):
        return self.queue
    
    def initialize_queue(self,init_queue = []):
        self.queue = copy(init_queue)
    
    def sort(self,key=str.lower):
        self.queue = sorted(self.queue,key=key)
        
    def push(self,data):
        self.queue.append(data)
        self.end += 1
    
    def pop(self):
        p = self.queue.pop(self.start)
        self.end = len(self.queue)-1
        return p
    
class Node():
    def __init__(self,name):
        self.name = name
        self.children = []
        self.weight = []
        
    def __repr__(self):
        return self.name
        
    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)
    
class Tree():
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}
        self.g_visual = Graph('G')
    
    def __call__(self):
        for name,node in self.g.items():
            if(self.root == name):
                self.g_visual.node(name,name,color='red')
            elif(self.end == name):
                self.g_visual.node(name,name,color='blue')
            else:
                self.g_visual.node(name,name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                #print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name,c.name)
                else:
                    self.g_visual.edge(name,c.name,label=str(w))
        return self.g_visual
    
    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name
            
    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False
    
    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True  

class AStar():
    def __init__(self,in_tree):
        self.in_tree = in_tree
        
        self.q = Queue()
        self.dist = {name:np.Inf for name,node in in_tree.g.items()}
        self.h = {name:0 for name,node in in_tree.g.items()}
        
        for name,node in in_tree.g.items():
            start = tuple(map(int, name.split(',')))
            end = tuple(map(int, self.in_tree.end.split(',')))
            self.h[name] = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        
        self.via = {name:0 for name,node in in_tree.g.items()}
        
        # for __,node in in_tree.g.items():
        #     self.q.push(node)
        
     
    def __get_f_score(self, node):
        # Place code here remove the pass 
        # statement once you start coding

        return self.dist[node.name] + self.h[node.name]
    
    def __get_g_score(self, currentNode, neighbor):
        start = tuple(map(int, currentNode.name.split(',')))
        end = tuple(map(int, neighbor.name.split(',')))
        return  abs(start[0] - end[0]) + abs(start[1] - end[1])
    
    def solve(self, sn, en):

        self.dist[sn.name] = 0
        self.q.push(sn)
        
        while len(self.q) > 0:
            self.q.sort(key=self.__get_f_score)
            u = self.q.pop()
            #print(u.name,self.q.queue)
            
            if u.name == en.name:
                break
            for c in u.children:
                
                g_score = self.dist[u.name] + self.__get_g_score(u, c) 
                
                if g_score < self.dist[c.name]:
                    self.dist[c.name] = g_score
                    self.via[c.name] = u.name
                    
                    if c in self.q.queue:
                        self.q.queue.remove(c)
                    self.q.push(c)
                    
        # return self.reconstruct_path(sn, en), self.dist[en.name]
                 
    
    def reconstruct_path(self,sn,en,Path):
        path = []
        current = en.name
        # dist = 0
        # Place code here
        Path.header.stamp = Time()
        Path.header.frame_id = "map"
        
        while current != 0:
            currentNode=PoseStamped()
            current_float = tuple(map(float, current.split(',')))
            currentNode.pose.position.x=current_float[1]
            currentNode.pose.position.y=current_float[0]
            currentNode.pose.position.z=0.0 
            
            currentNode.pose.orientation.x=0.0 
            currentNode.pose.orientation.y=0.0
            currentNode.pose.orientation.z=0.0
            currentNode.pose.orientation.w=1.0 
            currentNode.header.stamp=Time()      # change accordingly
            currentNode.header.frame_id='map'
            path.append(current)
            Path.poses.append(currentNode)
            current = self.via[current]

        Path.poses.reverse()  # Reverse the order of Path.poses
        # return path[::-1]
        path.reverse()
        dist = self.dist[en.name]
        return path,dist
        
        # return path[::-1], self.dist[en.name]

class MapProcessor():
    def __init__(self,name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)
    
    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and 
            (i < map_array.shape[0]) and 
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value 
    
    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute)
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy):
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)
        
    def inflate_map(self,kernel,absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r
                
    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = Node('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:                    
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left 
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left 
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[np.sqrt(2)])                    
        
    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm
    
    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size))
        return m
    
    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array



def main(args=None):
    rclpy.init(args=args)
    nav = Navigation(node_name='Navigation')
    # nav.convertWorldFrameToNodeFrame()

    try:
        nav.run()
    except KeyboardInterrupt:
        pass
    finally:
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
