import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D
from cv_bridge import CvBridge
import cv2
from matplotlib import pyplot as plt
import numpy as np


class image():
    def __init__(self):
        self.ros2 = None
        self.cv2 = None
        self.hsv = None        
        
class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.subscriber = self.create_subscription(Image, 'video_data', self.objectDetect, 10)
        self.publisher = self.create_publisher(BoundingBox2D, 'bbox', 10)
        self.image = image()
        self.bridge = CvBridge()
        
    def objectDetect(self,msg):
        
        # step 1: Detect red color objects using hsv mask
        # step 2: Find contour and identify triangle
        # step 3: Draw box
        
        # When target's features include color, it'd better to use hsv masking
        # Otherwise, gray scale + edge detection would be better.
        
        if msg != self.image.ros2:
        
            self.get_logger().info(f'Received new image : {msg}')
            self.image.ros2 = msg
            self.image.cv2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='8UC3')
            
            # cv2.imshow('Frame', self.image.cv2)
            # cv2.waitKey(25)

            # convert the image from RGB into hue
            # https://answers.opencv.org/question/184711/select-hsv-hue-from-30-to-30-in-python/
            # https://docs.wpilib.org/en/stable/docs/software/vision-processing/wpilibpi/image-thresholding.html
            # https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
            self.image.hsv = cv2.cvtColor(self.image.cv2, cv2.COLOR_BGR2HSV)
            # create color masks (hue [0, 180], saturation [0, 255], brightness [0, 255])
            
            # red from 0 to 20 degree
            lower_bound1 = np.array([0, 70, 130])
            upper_bound1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(self.image.hsv, lower_bound1, upper_bound1)
            # red from 160 to 180 degree
            lower_bound2 = np.array([170, 70, 130])
            upper_bound2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(self.image.hsv, lower_bound2, upper_bound2)
            
            mask = mask1 + mask2
            masked = cv2.bitwise_and(self.image.cv2, self.image.cv2, mask = mask)
            
            cv2.imshow('Before mask', self.image.cv2)
            cv2.imshow('Mask', mask)
            cv2.imshow('After mask', masked)
            cv2.waitKey(500)
            
        else:
            cv2.destroyAllWindows()
            
            
            
        
        
def main():
    rclpy.init()
    od = ObjectDetector()
    rclpy.spin(od)
    od.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()