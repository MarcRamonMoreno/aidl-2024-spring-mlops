import logging
import pathlib

import torch
from flask import Flask, render_template, request
from torchtext.data.utils import get_tokenizer, ngrams_iterator

from model import SentimentAnalysis


VOCAB = None
MODEL = None
NGRAMS = None
TOKENIZER = None

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)

# The code in this function will be executed before we recieve any request
@app.before_first_request
def _load_model():
    # Assuming the rest of your setup code remains unchanged
    checkpoint_path = pathlib.Path(__file__).parent.absolute() / "state_dict.pt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    global VOCAB, MODEL, NGRAMS, TOKENIZER, MAP_TOKEN2IDX
    VOCAB = checkpoint["vocab"]
    # Loading the model with parameters from the checkpoint
    embed_dim = checkpoint['embed_dim']
    num_class = checkpoint['num_class']
    vocab_size = len(VOCAB)
    MODEL = SentimentAnalysis(vocab_size=vocab_size, embed_dim=embed_dim, num_class=num_class)
    MODEL.load_state_dict(checkpoint['model_state_dict'])

    NGRAMS = checkpoint["ngrams"]
    TOKENIZER = get_tokenizer("basic_english")
    MAP_TOKEN2IDX = VOCAB.get_stoi()
    MODEL.eval()  # Set the model to evaluation mode



# Disable gradients
@torch.no_grad()
def predict_review_sentiment(text):
    tokenized = [MAP_TOKEN2IDX.get(token, 0) for token in ngrams_iterator(TOKENIZER(text), NGRAMS)]
    if not tokenized:  # Check if the list is empty
        return 0  # You may choose a different way to handle empty inputs
    text_tensor = torch.tensor([tokenized], dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)
    output = MODEL(text_tensor, offsets)
    confidences = torch.softmax(output, dim=1)
    return confidences.squeeze()[1].item()  # Assuming class 1 is positive sentiment

@app.route("/predict", methods=["POST"])
def predict():
    """The input parameter is `review`"""
    review = request.form["review"]
    print(f"Prediction for review:\n {review}")

    result = predict_review_sentiment(review)
    return render_template("result.html", result=result)


@app.route("/", methods=["GET"])
def hello():
    """ Return an HTML. """
    return render_template("hello.html")


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == "__main__":
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host="localhost", port=8080, debug=True)
