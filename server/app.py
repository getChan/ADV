from flask import Flask, request, render_template
import demo_run

def create_app():
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
        
    return app