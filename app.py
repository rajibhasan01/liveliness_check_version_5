from flask import Flask, render_template, Response
import cv2

import FaceMeshModule as fm;
import face_question as fq;

app = Flask(__name__)
video_capture = cv2.VideoCapture(0)
detector = fm.FaceMeshDetector(max_num_faces=1);

def generate_frames():
    fq.generate_status();
    while True:
        # read the camera frame
        success, frame = video_capture.read()
        
        if not success:
            break
        else:
            frame = cv2.flip(frame,1);
            frame = detector.findFaceMesh(frame, False);
            frame, face_orientation = detector.find_Orientation(frame, False);
            
            # Generating new question
            new_question = fq.generate_qstn(frame);
    
            # Matching buffer ans with current question
            fq.match_q_a(face_orientation);
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/check_liveliness')
def check_liveliness():
    return render_template('check_liveliness.html')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run()