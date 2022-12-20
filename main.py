import os
from flask import (
    Flask,
    flash,
    request,
    redirect,
    render_template,
    url_for,
    send_from_directory,
)
from werkzeug.utils import secure_filename
from pathlib import Path
from prepare_video import run

uploads = Path(__file__).parent / "videos"
UPLOAD_FOLDER = uploads.resolve()
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # here replace orginal vide with the new one
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            filepath = Path(UPLOAD_FOLDER) / filename
            run(filepath.resolve())
            return redirect(url_for("download_file", name=filename))
    return render_template("index.html", uploaded_videos=os.listdir(UPLOAD_FOLDER))


@app.route("/uploads/<name>")
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


if __name__ == "__main__":
    app.run()
