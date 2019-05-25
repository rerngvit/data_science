import flask
import io, os
import tensorflow as tf
import tensorflow_hub as hub
from text_preprocessor import TextPreprocessor

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None
session = None
text_preprocessor = TextPreprocessor()
graph = tf.Graph()

def load_sentence_encoder():
    global model, session, graph
    print(" Start initializing Tensorflow hub")
    os.environ["TFHUB_CACHE_DIR"] = '/tf_hub_cache'
    # Create and intialize the Tensorflow session
    with graph.as_default():    
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        model = hub.Module(module_url)    
        session = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    print(" Model Sentence Encoder is loaded")

@app.route("/sentence_encoder", methods=["POST"])
def sentence_encoder():
    global graph
    data = {"success": False}
    if flask.request.is_json:
        content = flask.request.get_json()
        input_txt = content["input_txt"]
        preprocessed_txt = text_preprocessor.preprocess_text(input_txt)
        data["input_text"] = input_txt
        data["preprocessed_text"] = preprocessed_txt

        with graph.as_default():
            sentence_vector = session.run(model([preprocessed_txt]))[0]
            array_float_str = "{%s}" % ",".join(["{:f}".format(x) for x in list(sentence_vector)])
        
        data["encoded_vector"] = array_float_str
        data["success"] = True
    else:
        data["Error"] = "the input data type is invalid (not JSON)"
    return flask.jsonify(data)

if __name__ == "__main__":
	print(" Loading model and start the server...")
	load_sentence_encoder()
	app.run(host='0.0.0.0', port=9090)