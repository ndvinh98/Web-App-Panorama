import os
from flask import Flask, render_template, request, redirect
from flask_dropzone import Dropzone
from stitch import multiStitching
from utils import loadImages
import cv2
import glob
import timeit

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=10,
    DROPZONE_MAX_FILES=30,
    DROPZONE_IN_FORM=True,
    DROPZONE_UPLOAD_ON_CLICK=True,
    DROPZONE_UPLOAD_ACTION='handle_upload',
    DROPZONE_UPLOAD_BTN_ID='submit',
)

dropzone = Dropzone(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    for key, f in request.files.items():
        if key.startswith('file'):
            f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
    return '', 204

@app.route('/form', methods=['POST'])
def handle_form():
    start = timeit.default_timer()
    resize = request.form.get('resize')
    print(resize)
    if resize=='0':
       list_images=loadImages(os.path.join(app.config['UPLOADED_PATH']),resize=0)
       for k in glob.glob(os.path.join(app.config['UPLOADED_PATH']+'/*.*')):
            os.remove(k)
       panorama=multiStitching(list_images,option='SURF',ratio=0.75)
    if resize=='1':
       list_images=loadImages(os.path.join(app.config['UPLOADED_PATH']),resize=0)
       for k in glob.glob(os.path.join(app.config['UPLOADED_PATH']+'/*.*')):
            os.remove(k)
       panorama=multiStitching(list_images,option='ORB',ratio=0.75)
    if resize=='2':
        list_images=loadImages(os.path.join(app.config['UPLOADED_PATH']),resize=1)
        for k in glob.glob(os.path.join(app.config['UPLOADED_PATH']+'/*.*')):
            os.remove(k)
        panorama=multiStitching(list_images,option='SURF',ratio=0.75)
    #list_images=loadImages(os.path.join(app.config['UPLOADED_PATH']),resize=1)
    
    
    cv2.imwrite('static/panorama.jpg',panorama)
    stop = timeit.default_timer()
    return render_template('result.html',timee=stop-start)
    

@app.route('/result')
def viewResult():
    return render_template('result.html')

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == '__main__':
    app.run(debug=True)
    TEMPLATES_AUTO_RELOAD = True
