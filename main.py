# main.py

import os
from flask import Flask, render_template, Response, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
from camera import VideoCamera

htmlfile = open('./templates/index.html', 'r')
html = htmlfile.read()

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

video_camera = VideoCamera()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            pic = VideoCamera()
            pic.get_pic(filename)
            filename = 'hat_' + filename
            file_url = url_for('uploaded_file', filename=filename)
            # SOME BUGS HERE
            return html + '<br><img src=' + file_url + '>'
    return html

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
            b'Content-type: image/jpg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    video_camera.open_camera()
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_close')
def video_close():
    video_camera.close_camera()
    frame = video_camera.get_singal('./templates/signal.png')
    return Response(frame, mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # app.run(host='0,0,0,0', debug=True)
    app.run()

