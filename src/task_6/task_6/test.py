import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class image():
    def __init__(self):
        self.ros2 = None
        self.cv2 = None
        self.hsv = None
        self.bridge =CvBridge()
        self.height = None
        self.width = None
        self.gray = None
    
    def cv2toros2(self):
        self.ros2 = self.bridge.cv2_to_imgmsg(self.cv2)    
    
    def ros2tocv2(self):
        self.cv2 = self.bridge.imgmsg_to_cv2(self.ros2, desired_encoding='8UC3')
        # self.cv2 = self.bridge.imgmsg_to_cv2(self.ros2, desired_encoding='rgb8')
        
class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.subscriber1 = self.create_subscription(Image, '/camera/image_raw', self.objectDetect, 10)
        self.msg = image()
        self.bridge = CvBridge()
        
    def objectDetect(self, msg):
        
        if msg != self.msg.ros2:
            self.get_logger().info('Received new image')
            self.msg_update = True
            self.msg.ros2 = msg
            self.msg.ros2tocv2()
            self.msg.height = msg.height
            self.msg.width = msg.width
            self.hsvMask()
            
    def hsvMask(self):
        self.msg.hsv = cv2.cvtColor(self.msg.cv2, cv2.COLOR_BGR2HSV)

        upper_bound = np.array([120, 255, 255])
        lower_bound = np.array([105, 180, 80])
        mask = cv2.inRange(self.msg.hsv, lower_bound, upper_bound)
        masked = cv2.bitwise_and(self.msg.cv2, self.msg.cv2, mask = mask)
        
        
        
        cv2.imshow('frame', self.msg.cv2)
        cv2.imshow('after mask', masked)
        
        image, center = self.findContour(mask)

        if image is not None: 
            cv2.imshow('after detection', image)
            
        cv2.waitKey(25)
    
    def findContour(self, mask):
        
        if mask is None:
            return None, None
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = list()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 400:
                continue
                                              
            """
            To-do: write a curvature check
            """
            error = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, error, True)
            
            if len(approx) >= 5:
                image = cv2.drawContours(self.msg.cv2, [approx], -1, (0, 255, 0), 3)
                center.append(float(np.mean(approx[:, 0, 0])))  #x
                center.append(float(np.mean(approx[:, 0, 1])))  #y
                
                self.get_logger().info(f'Found a red ball, area = {area}')
                return image, center
            
    
        self.get_logger().info('Didn\'t found a red ball')
        return None, None
    
    
def main():
    rclpy.init()
    ob = ObjectDetector()
    rclpy.spin(ob)
    ob.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
