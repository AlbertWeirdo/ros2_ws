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
            self.hsvMask()
        else:
            cv2.destroyAllWindows()    
        
    
    def hsvMask(self):
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
        imageWithBoundingBox = self.findContour(mask)
        if imageWithBoundingBox is not None:
            cv2.imshow("Triangle Detection", imageWithBoundingBox)
        cv2.waitKey(25)
        
    def findContour(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            
            # filter out noise
            area = cv2.contourArea(contour)
            if area < 1000:
               continue
           
            # check if the contour is a triangle
            # acceptable error range will be 2 percent of the contour's perimeter  
            error = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, error, True)
            
            if len(approx) == 3:
                # draw a green triangle outside the object 
                cv2.drawContours(self.image.cv2, [approx], -1, (0, 255, 0), 3)
                # draw a bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                image = cv2.rectangle(self.image.cv2, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # compute triangle's centroid
                triangle_center_x = np.mean(approx[:, 0, 0])
                triangle_center_y = np.mean(approx[:, 0, 1])
                print(f'triangle\'s centroid is {triangle_center_x - x, triangle_center_y - y} relative to the top left corner of the frame')
                
                pub_msg = BoundingBox2D()
                # pub_msg.center = triangle_center_x
                # pub_msg.center = triangle_center_y
                pub_msg.size_x = float(w)
                pub_msg.size_y = float(h)
                
                self.publisher.publish(pub_msg)
                
                # cv2.imshow("Triangle Detection", self.image.cv2)
                # cv2.waitKey(25)
                return image
            
                       
        
        
def main():
    rclpy.init()
    od = ObjectDetector()
    rclpy.spin(od)
    od.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()