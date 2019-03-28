from flask import Flask,render_template
app=Flask(__name__)


@app.route('/')
def index():
    return render_template('C09_first_app.html')


if __name__=="__main__":
#     print("hii")
    app.run()

# main()