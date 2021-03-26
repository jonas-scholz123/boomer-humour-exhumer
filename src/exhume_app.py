import os
from flask import Flask, flash, request, redirect, url_for
# from flask import Flask, request
from werkzeug.utils import secure_filename

from exhume import ExhumerContainer

# app = Flask(__name__)
# 


UPLOAD_FOLDER = '../data/user_uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


base_html =  '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=image>
      <input type=submit value=Upload>
    </form>
    '''

exhumer = ExhumerContainer()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/exhumepath', methods=["POST"])
def exhume_image_at_path():
    im_path = request.form["im_path"]
    return str(exhumer.exhume(im_path))

@app.route('/api/exhume', methods=['POST'])
def upload_file():
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No image part')
            return redirect(request.url)
        file = request.files['image']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fpath)
            
            prob = round(exhumer.exhume(fpath) * 100, 2)

            boomer_string = f"The image you just uploaded exhudes {str(prob)} % boomer energy."
            return boomer_string