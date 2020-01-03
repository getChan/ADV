from flask import Flask, jsonify, request, render_template
import demo_run
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def demo():
    if request.method == "POST":
        query = request.form.get('query')
        # model predict
        print(query)
        result = demo_run.translate(query)
        return render_template("demo.html", result=result, query=query)
    else:
        return render_template('demo.html')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0') 