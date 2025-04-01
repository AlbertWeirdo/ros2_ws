import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D
from cv_bridge import CvBridge
import cv2 as cv

class image():
    def __init__(self):
        self.ros2 = None
        self.cv2 = None
        self.gray = None
        
class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.subscriber = self.create_subscription(Image, 'video_data', self.objectDetect, 10)
        self.publisher = self.create_publisher(BoundingBox2D, 'bbox', 10)
        self.image = image()
        self.bridge = CvBridge()
        
    def objectDetect(self,msg):
        if msg != None:
            self.get_logger.info(f'Received new image : {msg}')
            self.image.ros2 = msg
            self.image.cv2 = self.bridge.cv2_to_imgmsg(msg)
            self.image.gray = cv.cvtColor(self.image.cv2, cv.COLOR_BGR2GRAY)
            cv.imshow(self.image.gray)
            
            
        
        
def main():
    rclpy.init()
    od = ObjectDetector()
    rclpy.spin(od)
    od.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()