from flask import render_template, Flask, request
from model import give_rec

app = Flask(__name__)

@app.route('/<movie>')
def main_page(movie):
    return give_rec(movie.title())

if __name__ == "__main__":
    app.run(debug=True)