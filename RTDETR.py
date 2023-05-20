#!/usr/bin/env python
import cv2
import torch
import random

import rospy

from cv_bridge import CvBridge

from ultralytics import RTDETR


from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D
from vision_msgs.msg import ObjectHypothesisWithPose
from vision_msgs.msg import Detection2DArray
from std_srvs.srv import SetBool


class RT_DETR():

    def __init__(self) -> None:

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        # params
        # self.model = rospy.get_param("model", "rtdetr-l.pt")
       
        # self.img_topic = rospy.get_param("img_topic", "/airsim_node/Drone_1/camera_1/Scene")
       

        # self.threshold = rospy.get_param("threshold", 0.5)

        # self.enable = rospy.get_param("enable", True)
        
        # # params
        self.model1 = "rtdetr-l.pt"
        self.model2 = "rtdetr-x.pt"
        self.bool_rtdetr_l = True

        self.img_topic = "/airsim_node/Drone_1/camera_1/Scene"
       
        self.threshold =  0.5

        self.enable =  True

        self._class_to_color = {}
        self.cv_bridge = CvBridge()
        if self.bool_rtdetr_l:
            self.rtdetr = RTDETR(self.model1)
            
        else:
            self.rtdetr = RTDETR(self.model2)
            
        rospy.sleep(1)
        

        # topics
        self._pub = rospy.Publisher("detections",Detection2DArray, queue_size= 10)
        self._detection_image_pub = rospy.Publisher("detection_image", Image, queue_size= 10)
        rospy.sleep(1)
        self._sub = rospy.Subscriber(self.img_topic, Image, self.image_cb)

        self.coco_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 
                           8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 
                           15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 
                           24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
                           32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 
                           38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
                           46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 
                           54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
                           62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
                           70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 
                           78: 'hair drier', 79: 'toothbrush'}


    def image_cb(self, msg: Image) -> None:

        if self.enable:

            # convert image + predict
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            
            results = self.rtdetr.predict(source = cv_image, verbose = False, show = False, save = False)
            

            # create detections msg
            detections_msg = Detection2DArray()
            detections_msg.header = msg.header

            results = results[0].cpu()
            

            for b in results.boxes:

                label = self.coco_names[int(b.cls)]
                score = float(b.conf)

                if score < self.threshold:
                    continue

                detection = Detection2D()

                detection.header = msg.header

                detection.source_img = msg

                box = b.xywh[0]

                # get boxes values
                detection.bbox.center.x = float(box[0])
                detection.bbox.center.y = float(box[1])
                detection.bbox.size_x = float(box[2])
                detection.bbox.size_y = float(box[3])

                # get track id
                track_id = 0
                if not b.id is None:
                    track_id = int(b.id)
                #detection.id = track_id


                # get hypothesis
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = int(b.cls)
                hypothesis.score = score
                detection.results.append(hypothesis)

                # draw boxes for debug
                if label not in self._class_to_color:
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b1 = random.randint(0, 255)
                    self._class_to_color[label] = (r, g, b1)
                color = self._class_to_color[label]

                min_pt = (round(detection.bbox.center.x - detection.bbox.size_x / 2.0),
                          round(detection.bbox.center.y - detection.bbox.size_y / 2.0))
                max_pt = (round(detection.bbox.center.x + detection.bbox.size_x / 2.0),
                          round(detection.bbox.center.y + detection.bbox.size_y / 2.0))
                cv_image = cv2.rectangle(cv_image, min_pt, max_pt, color, 2)

                label = "{}:({}) {:.3f}".format(label, str(track_id), score)
                pos = (min_pt[0], max(15, int(min_pt[1] - 10)))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv_image = cv2.putText(cv_image, label, pos, font,
                            0.5, color, 1, cv2.LINE_AA)


                # append msg
                detections_msg.detections.append(detection)

            # publish detections and dbg image
            self._pub.publish(detections_msg)

               
            self._detection_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image,
                                                            encoding=msg.encoding))
            cv2.imshow("Real-Time Detection with RTDETR",cv_image)
            cv2.waitKey(1)

            if rospy.is_shutdown():
                cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node("RTDETR_Sea", anonymous= True)
    RT_DETR()
    rospy.spin()