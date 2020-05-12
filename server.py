from flask import Flask, render_template, request, make_response
import werkzeug
import os
from Main import Model
app = Flask(__name__)

ALLOWED_EXTENSIONS = ['csv', 'xlsx']
TEST_DIRECTORY = 'test_data'
TABLE_NAME = {'xlsx': 'test.xlsx', 'csv': 'test.csv'}
HTML_FILE = 'main.html'

error = {'format': 'Недопустимый формат файла!', 'file': 'Где-то произошла ошибка. Cкорее всего, файл не загружен.'}

model = None

def extension_check(filename):
    return filename.split('.')[-1] if '.' in filename else 'error'



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print('POST')
        global model
        df = request.files['table']
        date = request.form.get('date')
        print(date)
        filename = werkzeug.secure_filename(df.filename)
        extension = extension_check(filename)
        if extension in ALLOWED_EXTENSIONS:
            path_to_test_data = os.path.join(TEST_DIRECTORY, TABLE_NAME[extension])
            df.save(path_to_test_data)
            data = model.test(path_to_test_data, dates=[date]).sort_values(by='Churn', ascending=False)\
                .rename(columns={'Churn': 'Уверенность модели'}).reset_index().to_html()
            print('READY')
            return render_template(HTML_FILE, data=data)
        else:
            return render_template(HTML_FILE, error=error['format'])


    if request.method == 'GET':
        print('GET')
        return render_template('main.html')


if __name__ == '__main__':
    model = Model()
    #model.load()
    model.train()
    app.run(debug=False, port='8810')