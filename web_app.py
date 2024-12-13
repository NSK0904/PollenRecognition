from flask import Flask, request, render_template, jsonify, Response
import os
import pandas as pd
from modules.inference import run_inference
import threading
from queue import Queue

app = Flask(__name__)

CONFIDENCE_LOG_PATH = r"ENTER_PATH_HERE"

log_queue = Queue()

status = {"completed": False}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    global status
    status["completed"] = False 
    log_queue.queue.clear() 

    test_folder_path = request.form.get("test_folder_path")
    model_file = request.files.get("model_file")

    if not test_folder_path or not os.path.exists(test_folder_path):
        return "Test folder path is invalid or missing", 400

    if not model_file or not model_file.filename.endswith('.onnx'):
        return "Only ONNX files are allowed for the model", 400

    model_file_path = os.path.join(os.getcwd(), model_file.filename)
    model_file.save(model_file_path)

    def inference_task():
        try:
            log_queue.put("Starting inference task...")
            def log_callback(message):
                log_queue.put(message)

            run_inference(test_folder_path, model_file_path, CONFIDENCE_LOG_PATH, log_callback)
            log_queue.put(f"Confidence log saved at: {CONFIDENCE_LOG_PATH}")
            log_queue.put("Inference task completed.")
            status["completed"] = True
        except Exception as e:
            log_queue.put(f"Inference error: {e}")
            status["completed"] = False

    threading.Thread(target=inference_task).start()
    return jsonify({"message": "Inference started"})

@app.route('/logs')
def stream_logs():
    def generate_logs():
        while True:
            log = log_queue.get()
            yield f"data: {log}\n\n"
            if "completed" in log.lower() or "error" in log.lower():
                break

    return Response(generate_logs(), mimetype="text/event-stream")

@app.route('/results')
def show_results():
    try:
        if not os.path.exists(CONFIDENCE_LOG_PATH):
            return f"Confidence log not found at {CONFIDENCE_LOG_PATH}", 404

        df = pd.read_csv(CONFIDENCE_LOG_PATH)

        df = df.replace(r'\n', '', regex=True)
        df = df.replace(r'\s+', ' ', regex=True) 
        df = df.fillna('') 

        records = df.to_dict(orient='records')
        columns = df.columns.tolist()

        return render_template(
            'results.html',
            records=records,
            columns=columns
        )
    except Exception as e:
        return f"An error occurred while loading results: {str(e)}", 500

if __name__ == '__main__':
    os.makedirs(os.path.dirname(CONFIDENCE_LOG_PATH), exist_ok=True)
    app.run(debug=True)