from flask import Flask, request, jsonify, send_file
from model_test_script import final_video

app = Flask(__name__)

def main_function(gender, audio_path, specific_avatar):
    re = final_video(gender, audio_path, specific_avatar)
    return re

@app.route('/')
def index():
    return "Hello world"


@app.route('/video', methods=['GET'])
def get_video():
    video_path = 'final_output.mp4'
    return send_file(video_path, as_attachment=True)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    # Save the file to a secure location
    file.save('data/audio.wav')
    
    return jsonify({'message': 'File uploaded successfully'}), 200


@app.route('/retrive', methods=['POST'])
def retrive():
 

    specific_avatar = int(request.form['specific_avatar'])
    gender = request.form['gender']

    audio_path = 'data/audio.wav'

    result = main_function(gender, audio_path, specific_avatar)

    return jsonify({
        'audio_path': audio_path,
        'specific_avatar': specific_avatar,
        'gender': gender,
        'video_path': result
    })

if __name__ == '__main__':
    app.run(host='192.168.1.6', port=5000, debug=True)