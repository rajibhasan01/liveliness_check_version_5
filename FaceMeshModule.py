import cv2;
import mediapipe as mp;
import time;


count = 0;
face_orientation = 'front'

class FaceMeshDetector():

    def __init__(self,
               static_image_mode=False,
               max_num_faces=1,
               refine_landmarks=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):

        self.staticMode = static_image_mode;
        self.maxFaces = max_num_faces;
        self.refine_landmarks = refine_landmarks;
        self.minDetectionCon = min_detection_confidence;
        self.minTrackCon = min_tracking_confidence;

        self.mpDraw = mp.solutions.drawing_utils;
        self.mpFaceMesh = mp.solutions.face_mesh;
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refine_landmarks,
                                                 self.minDetectionCon, self.minTrackCon);
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2);

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
        self.results = self.faceMesh.process(self.imgRGB);
        faces = [];
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                           self.drawSpec, self.drawSpec);
      
                for id,lm in enumerate(faceLms.landmark):
                  
                    ih, iw, ic = img.shape;
                    x,y = int(lm.x*iw), int(lm.y*ih);
                    
                    if id == 93:
                      x1 = x - 10;
                    if id == 10:
                      y1 = y - 20;
                    
                    if id == 152:
                      y2 = y ;
                    if id == 454:
                      x2 = x + 10;
                      
                      # img = cv2.rectangle(img, (x1 , y1 ), (x2 ,y2 ), (0,0,255), 2);
                      image = cv2.line(img, (x1 , y1 ), (x1+20 ,y1 ), (0,255,0), 2);
                      image = cv2.line(img, (x1 , y1 ), (x1 ,y1+20 ), (0,255,0), 2);
                      
                      image = cv2.line(img, (x2 - 20 , y1 ), (x2 ,y1 ), (0,255,0), 2);
                      image = cv2.line(img, (x2 , y1 ), (x2 ,y1+20 ), (0,255,0), 2);
                      
                      image = cv2.line(img, (x1 , y2 ), (x1 ,y2-20 ), (0,255,0), 2);
                      image = cv2.line(img, (x1, y2 ), (x1 + 20 ,y2 ), (0,255,0), 2);
                      
                      image = cv2.line(img, (x2 - 20 , y2 ), (x2 ,y2 ), (0,255,0), 2);
                      image = cv2.line(img, (x2 , y2 -20 ), (x2 ,y2 ), (0,255,0), 2);

        return img
      
    def find_Orientation(self, img, draw=True):
      
      global count;
      global face_orientation;

      if self.results.multi_face_landmarks:
          for faceLms in self.results.multi_face_landmarks:
            # print('right',faceLms.landmark[323].x)
            # print('middle',faceLms.landmark[1].x)
            # print('left', faceLms.landmark[93].x)
            
            if faceLms.landmark[1].x >= faceLms.landmark[323].x:
              count += 1;
              face_orientation = "right";
              # print("Right ", count);
              
            elif faceLms.landmark[1].x <= faceLms.landmark[93].x:
              count += 1;
              face_orientation = "left";
              # print("Left ", count);
            
            elif faceLms.landmark[175].y >= faceLms.landmark[152].y:
              count += 1;
              face_orientation = "down";
              # print("Down ", count);
            
            elif faceLms.landmark[134].y <= faceLms.landmark[127].y:
              count += 1;
              face_orientation = "up";
              # print("Up ", count);
            
            elif faceLms.landmark[14].y - faceLms.landmark[13].y >= 0.015:
              count += 1;
              face_orientation = "open";
              # print("Open ", count);
            
            else:
              face_orientation = "font"
              
            
              
              
              
            # print('open', faceLms.landmark[159].y)
            # print('close', faceLms.landmark[145].y)
            
      return img, face_orientation;
              


def main():
    cap = cv2.VideoCapture(0);
    pTime = 0;
    detector = FaceMeshDetector(max_num_faces=1);
    
    while True:
        success, img = cap.read();
        # img = cv2.flip(img,1)
        img, face = detector.findFaceMesh(img);
        # if len(face)!= 0:
        #     # print(face[0])
        cTime = time.time();
        fps = 1 / (cTime - pTime);
        pTime = cTime;

        cv2.imshow("Image", img);
        cv2.waitKey(1);


if __name__ == "__main__":
    main();