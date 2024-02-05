import subprocess
from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    output = ""  # Initialize an empty string to capture the output
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files[]')


        if uploaded_files:
            temp_dir = 'uploads'
            os.makedirs(temp_dir, exist_ok=True)

            for uploaded_file in uploaded_files:
                uploaded_file_path = os.path.join(temp_dir, uploaded_file.filename)
                uploaded_file.save(uploaded_file_path)

                # Capture the output of the backend script using subprocess
                result = subprocess.Popen(["python", "backend_code.py", uploaded_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Read the stdout and stderr to capture the output
                out, err = result.communicate()
                output += f"Output for {uploaded_file.filename}:\n{out + err}\n\n"  # Combine both stdout and stderr

    return render_template('index.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)
# app.py
# import subprocess
# from flask import Flask, render_template, request
# import os
#
# app = Flask(__name__)
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     output = ""
#
#     if request.method == 'POST':
#         uploaded_files = request.files.getlist('files[]')
#         year = request.form.get('year')
#         print(year)
#         if uploaded_files and year:
#             temp_dir = 'uploads'
#             os.makedirs(temp_dir, exist_ok=True)
#
#             for uploaded_file in uploaded_files:
#                 uploaded_file_path = os.path.join(temp_dir, uploaded_file.filename)
#                 uploaded_file.save(uploaded_file_path)
#
#                 result = subprocess.Popen(["python", "backend_code.py", uploaded_file_path, "--year", str(year)] , stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
#
#                 out, err = result.communicate()
#                 output += f"Output for {uploaded_file.filename}:\n{out + err}\n\n"
#
#     return render_template('index.html', output=output)
#
# if __name__ == '__main__':
#     app.run(debug=True)
