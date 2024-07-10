from flask import Flask, request, jsonify
from flask_cors import CORS
from core import Document, Core
from dotenv import load_dotenv
load_dotenv()
import redis
import pickle
import os


app = Flask(__name__)
cors=CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
status = "active"

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

redis_host = os.getenv('REDIS_HOST')
redis_port = os.getenv('REDIS_PORT')

pool = redis.ConnectionPool(host=redis_host, port=redis_port, db=0)
redis_client = redis.Redis(connection_pool=pool)


@app.route('/')
def hello_world():
    return jsonify({"status":status,"Value":'surfmind server running successfully',"Version":1.0})

@app.route('/ingest/<sid>', methods=['POST'])
def ingest(sid):
    data = request.json
    print('got data from server')
    docs=[]
    for x in data['dataa']:
        doc = Document(x["content"],{'source': x['url'], 'date':x['date']})
        docs.append(doc)
    print('making docs\n')
    core = Core()
    process_docs = core.makeDocs(docs)
    print('saving in redis')
    redis_client.set(sid, pickle.dumps(process_docs))
    return jsonify({"status": "success"}), 200

@app.route('/search/<sid>', methods=['POST'])
def search(sid):
    data = request.json
    ques = data.get('query')
    core=Core()
    print('inside search')
    chain = core.LLMResponse()
    print('getting from redis')
    content = redis_client.get(sid)
    process_docs = pickle.loads(content)
    print('getting result')
    docs = process_docs.invoke(ques)
    context = docs[0].page_content
    url = docs[0].metadata['source']
    date= docs[0].metadata['date']
    print('getting AI response')
    result = chain.invoke([context,date,url])
    res = jsonify({"success":True,"result":result})
    return (res)


if __name__ == '__main__':
    app.run(debug=True, port=8000)