import os
from app import app
from flask import flash, request, redirect, url_for, render_template
from prepare_video import ModelRunner
from pathlib import Path
from miskibin import get_logger

logger = get_logger(lvl=20)
runner = ModelRunner()


@app.route("/")
def upload_form():
    return render_template("upload.html")


@app.route("/", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        flash("No image selected for uploading")
        return redirect(request.url)
    else:
        filename = file.filename
        logger.info(f"File {filename} is selected for uploading")
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        runner.run(Path(os.path.join(app.config["UPLOAD_FOLDER"], filename)))
        logger.info(f"File {filename} successfully uploaded")
        # print('upload_video filename: ' + filename)
        flash("Video successfully uploaded and displayed below")
        return render_template(
            "upload.html", filename=str(filename).replace(".avi", ".mp4")
        )


@app.route("/display/<filename>")
def display_video(filename):
    # print('display_video filename: ' + filename)
    return redirect(url_for("static", filename="uploads/" + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)
