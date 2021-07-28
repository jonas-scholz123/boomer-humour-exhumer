import os
from flask import Flask, flash, request, redirect, url_for, jsonify
# from flask import Flask, request
from werkzeug.utils import secure_filename

from exhume import ExhumerContainer

UPLOAD_FOLDER = '../data/user_uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}


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
def exhume_image():
        if 'image' not in request.files:
            # 415 => Unsupported Media Type
            return jsonify({"error": "no image found"}), 415

        file = request.files['image']
        # if user does not select file, browser also
        # submit an empty part without filename

        if file.filename == '':
            return jsonify({"error": "no filename"}), 415

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fpath)
            prob, engine, text = exhumer.exhume_with_meta(fpath)
            prob = round(prob * 100, 2)
            return jsonify({
                'boomerness': prob,
                'ocr_engine': engine.__class__.__name__,
                'text': text.replace('\n', " "),
                }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)