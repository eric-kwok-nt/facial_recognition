from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from .runner import runner
from peekingduck.pipeline.nodes.input import live, recorded


global capture,rec_frame, switch, face, rec, out 
capture=0
face=0
switch=1
rec=0

try:
    os.mkdir('./uploads')
except OSError as error:
    pass

app = Flask(__name__, template_folder='./templates')
camera = cv2.VideoCapture(1)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['uploads', "IMG_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/fr", methods=["POST"])
def fr():
    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return render_template("predict.html")

    logging.info(f"Received input image file: {uploaded_file.filename}")
    # create temporary file with random name
    temp_filename_stem = str(uuid.uuid4())
    _, file_extension = os.path.splitext(uploaded_file.filename)
    temp_filename = temp_filename_stem + file_extension
    # print(f"os.getcwd(): {os.getcwd()}")
    temp_filepath = os.path.join(os.getcwd(), api_dir, temp_filename)

    # save image to temp file
    uploaded_file.save(temp_filepath)
    logging.info(f"Saving image file to temp file: {temp_filename}")

    logging.info(f"Running facial recognition...")
    img_array, _, bbox_labels = runner(type='api', input_filepath=api_dir)

    predicted_name = str(bbox_labels[0])

    # Delete temp image file after using it for prediction.
    os.remove(temp_filepath)

    logging.info(f"Temp file deleted: {temp_filename}")

    img = Image.fromarray(img_array)

    #Convert Pillow Image to bytes and then to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_byte = buffered.getvalue() # bytes
    img_base64 = base64.b64encode(img_byte) #Base64-encoded bytes * not str

    #It's still bytes so json.Convert to str to dumps(Because the json element does not support bytes type)
    img_str = img_base64.decode('utf-8') # str

    resp_dict = {
        "img": img_str,
        "name": predicted_name,
    }
    headers = {"Content-Type": "application/json"}
    return make_response(jsonify(resp_dict), 200, headers)


@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('stop') == 'Stop/Start':
            if(switch==1):
                input_node = live.Node()
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                camera = cv2.VideoCapture(0)
            else:
                camera = cv2.VideoCapture(1)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                camera = cv2.VideoCapture(1)
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=8000)
    # For production mode, comment the line above and uncomment below. Then run at the terminal: waitress-serve --port=8000 app:app
    # serve(app, host="0.0.0.0", port=8000)   # using waitress
    # app.run(host="0.0.0.0")
    
camera.release()
cv2.destroyAllWindows()     