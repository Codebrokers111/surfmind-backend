from flask import Flask, request, jsonify
from flask_cors import CORS
from core import Document, Core
from dotenv import load_dotenv
load_dotenv()
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


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    history = data.get('data')
    ques = data.get('query')
    core=Core()
    print('got data from server')
    docs=[]
    for x in history:
        doc = Document(x["content"],{'source': x['url'], 'date':x['date']})
        docs.append(doc)
    print('making docs\n')
    process_docs = core.makeDocs(docs)
    print('inside search')
    chain = core.LLMResponse()
    print('getting result')
    docs = process_docs.invoke(ques)
    context = docs[0].page_content
    url = docs[0].metadata['source']
    date= docs[0].metadata['date']
    print('getting AI response')
    result = chain.invoke([context,date,url])
    pchain = core.structure()
    finalOutput = pchain.invoke({"content":result})
    res = jsonify({"success":True,"result":result,"format":finalOutput})
    return res


if __name__ == '__main__':
    app.run(debug=True, port=8000)