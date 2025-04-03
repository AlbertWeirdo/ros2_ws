import rclpy
from rclpy.node import Node
import rclpy.timer
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class imageProcess():
    def __init__(self):
        self.ros2 = None
        self.cv2 = None
        self.hsv = None
        self.bridge =CvBridge()
        self.height = None
        self.width = None
    
    def cv2toros2(self):
        self.ros2 = self.bridge.cv2_to_imgmsg(self.cv2)    
    
    def ros2tocv2(self):
        self.cv2 = self.bridge.imgmsg_to_cv2(self.ros2, desired_encoding='8UC3')
        # self.cv2 = self.bridge.imgmsg_to_cv2(self.ros2, desired_encoding='rgb8')

class tracker(Node):
    def __init__(self):
        """
        subscribe to robot's camera and lidar
        publish robot's veleocity
        """
        super().__init__('tracker')
        self.subscriber1 = self.create_subscription(Image, '/camera/image_raw', self.objectDetect, 10)
        self.subscriber2 = self.create_subscription(LaserScan, '/scan', self.fetch_distance, 10)

        self.publisher = self.create_publisher(Twist, '/cmd_vel',10)
        self.bridge = CvBridge()
        self.msg = imageProcess()
        self.dist = None
        self.msg_update = False
        self.dist_update = False
        
        
    def objectDetect(self, msg):
        
        if msg != self.msg.ros2:
            self.get_logger().info('Received new image')
            self.msg_update = True
            self.msg.ros2 = msg
            self.msg.ros2tocv2()
            self.msg.height = msg.height
            self.msg.width = msg.width
            
            # cv2.imshow('frame', self.msg.cv2)
            # cv2.waitKey(25)
            # self.hsvMask()
    
    def fetch_distance(self, msg):
        if msg != self.dist:
            self.get_logger().info('Received new distance')
            self.dist_update = True
            self.dist = msg


            
    def hsvMask(self):
        self.msg.hsv = cv2.cvtColor(self.msg.cv2, cv2.COLOR_BGR2HSV)
        
        # upper_bound1 = np.array([10, 255, 255])
        # lower_bound1 = np.array([0, 70, 130])
        # upper_bound2 = np.array([180, 255, 255])
        # lower_bound2 = np.array([170, 70, 130])
        
        # mask1 = cv2.inRange(self.msg.hsv, lower_bound1, upper_bound1)
        # mask2 = cv2.inRange(self.msg.hsv, lower_bound2, upper_bound2)
        # mask =  mask1 + mask2
        
        upper_bound = np.array([120, 255, 255])
        lower_bound = np.array([105, 180, 100])
        mask = cv2.inRange(self.msg.hsv, lower_bound, upper_bound)
        masked = cv2.bitwise_and(self.msg.cv2, self.msg.cv2, mask = mask)
        
        # cv2.imshow('frame', self.msg.cv2)
        # cv2.imshow('after mask', masked)
        
        image, center, area = self.findContour(mask)
        
        cv2.namedWindow("after detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("after detection", 800, 600)
        if image is not None: 
            cv2.imshow('after detection', image)
            self.motionControl(center, area)
            
        else:
            cv2.imshow('after detection', self.msg.cv2)

            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = (1) * np.pi / 180
            self.publisher.publish(cmd_vel)
            
        cv2.waitKey(25)
    
    def findContour(self, mask):
        
        if mask is None:
            return None, None, None
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = list()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 400:
                continue
                                              
            error = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, error, True)
            
            if len(approx) >= 6:
                image = cv2.drawContours(self.msg.cv2, [approx], -1, (0, 255, 0), 3)
                center.append(float(np.mean(approx[:, 0, 0])))  #x
                center.append(float(np.mean(approx[:, 0, 1])))  #y
                
                self.get_logger().info(f'Found a red ball, area = {area}')
                return image, center, area
            
    
        self.get_logger().info('Didn\'t found a red ball')
        return None, None, None
            
    def motionControl(self, center, area):
        
        # if self.dist is None:
        #     return
        
        # the angle between the center of robot and the ball in rad
        ang = self.calculateBallAngle(center)
        self.get_logger().info(f'ang = {ang * 180 / np.pi } [degree]')
        self.get_logger().info(f'idx_min = {self.dist.angle_min}, idx_max = {self.dist.angle_max}, increment = {self.dist.angle_increment}')
        
        # distance between the robot and the ball's center
        # idx = -1 * int(((ang - 0) / self.dist.angle_increment) * 180 / np.pi)
        idx = int(-ang * 180 / np.pi)
        if idx < 0:
            idx += 360
        self.get_logger().info(f'idx = {idx}')
        idx = max(0, min(idx, len(self.dist.ranges) - 1))
        self.get_logger().info(f'idx = {idx}')
        dist = self.dist.ranges[idx]
        
        
        # pid parameters
        kp_speed = 0.1
        kp_ang_vel = 0.2
        
        # goal_dist = 1
        goal_area = 15000
        goad_ang = 0
        
        error_dist = -(area - goal_area) / goal_area
        # error_dist = dist - goal_dist
        
        error_ang = goad_ang - ang

        self.get_logger().info(f'error_dist = {error_dist}, error_ang = {error_ang}')
        
        max_vel = 1
        cmd_vel = Twist()
        cmd_vel.linear.x = kp_speed * error_dist
        cmd_vel.linear.x = max(min(cmd_vel.linear.x, 0.1 * max_vel), -0.1 * max_vel)
        cmd_vel.angular.z = kp_ang_vel * error_ang
        cmd_vel.angular.z = max(min(cmd_vel.angular.z, 0.15 * 1.02974 ), -0.15 * 1.02974)
        self.get_logger().info(f'vel = {cmd_vel.linear.x}, ang_vel = {cmd_vel.angular.z}')
        
        self.publisher.publish(cmd_vel)

    
    def calculateBallAngle(self, center):
        image_center = [0.5 * self.msg.width, 0.5 * self.msg.height]
        dist_ballcenter = (center[0] - image_center[0])
        
        
        # use [grep -r "horizontal_fov"] to find fov 
        # assume the robot's fov is 1.02974
        rad_per_pixel = self.msg.width / 1.02974
        
        ang = dist_ballcenter / rad_per_pixel
        self.get_logger().info(f'ang = {ang}')
        
        
        return ang
    
    def run(self):
        while rclpy.ok():
            self.msg_update = False
            self.dist_update = False
            rclpy.spin_once(self, timeout_sec=0.1)
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.msg_update and self.dist_update:
                self.hsvMask()
                
            else:
                self.get_logger().info('messages were not updated')
                
            time.sleep(1)                    
            
def main():
    rclpy.init()
    t = tracker()
    try:
        t.run()
    except KeyboardInterrupt:
        pass
    finally:
        t.destroy_node()
        rclpy.shutdown()
    
if __name__ == '__main__':
    main()
    