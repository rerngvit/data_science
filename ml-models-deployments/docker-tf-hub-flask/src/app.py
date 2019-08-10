import flask
import io, os
import tensorflow as tf
import tensorflow_hub as hub
from text_preprocessor import TextPreprocessor

def init_sentence_encoder():
    global session, graph, text_preprocessor, encoding_ops, messages_plh
    graph = tf.Graph()
    text_preprocessor = TextPreprocessor()

    print(" Start initializing Tensorflow hub")
    os.environ["TFHUB_CACHE_DIR"] = '/tf_hub_cache'
    # Create and intialize the Tensorflow session
    with graph.as_default():    
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        embed = hub.Module(module_url)    
        session = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        messages_plh = tf.placeholder(dtype=tf.string, shape=[None])
        encoding_ops = embed(messages_plh)
    print(" Model Sentence Encoder is loaded")

# initialize our Flask application and the model
app = flask.Flask(__name__)
print(" Loading model and start the server...")
init_sentence_encoder()
	
@app.route("/sentence_encoder", methods=["POST"])
def sentence_encoder():
    data = {"success": False}
    if flask.request.is_json:
        content = flask.request.get_json()
        input_txt = content["input_txt"]
        # Disable preprocessing
        preprocessed_txt = text_preprocessor.preprocess_text(input_txt)
        data["input_text"] = input_txt
        data["preprocessed_text"] = preprocessed_txt
        with graph.as_default():
            sentence_vector = session.run(encoding_ops, feed_dict={messages_plh: [preprocessed_txt]})[0]
            array_float_str = "{%s}" % ",".join(["{:f}".format(x) for x in list(sentence_vector)])
        
        data["encoded_vector"] = array_float_str
        data["success"] = True
    else:
        data["Error"] = "the input data type is invalid (not JSON)"
    return flask.jsonify(data)

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=9090)