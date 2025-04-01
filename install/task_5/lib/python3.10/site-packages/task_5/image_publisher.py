import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv

class ImagePublisher(Node):
    def __init__(self, address_to_video):
        super().__init__('image_publisher')
        
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image ,'video_data', 10)
        self.address_to_video = address_to_video
        self.processImage()
        
    def processImage(self):
        # read video
        cap = cv.VideoCapture(self.address_to_video)
        
        while cap.isOpened():
            success, frame = cap.read()
            
            # if the video is not being reading correctly
            if not success:
                print('Can\'t receive frame. Exiting')
                break
            
            cv.imshow('frame', frame)

            # convert the frame from opencv image to ros2 image
            ros2_image = self.bridge.cv2_to_imgmsg(frame)
            self.publisher.publish(ros2_image)

            # if the user press esc in the 25 milliseconds, the loop will stop
            if cv.waitKey(25) == 27:
                break
        
        cap.release()
        cv.destroyAllWindows()
        
def main():
    rclpy.init()
    ip = ImagePublisher('/home/albert/ros2_ws/src/task_5/resource/lab3_video.avi')
    ip.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
    
    