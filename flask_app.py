from PIL import Image
import base64
import io
from flask import Flask, request, jsonify
from backend.predict import preprocess_and_load, predict_and_postprocess, DetectedInfo
from pathlib import Path

# 传入__name__实例化Flask
app = Flask(__name__)

# 预处理及加载网络模型
model, exp, trt_file, decoder, args = preprocess_and_load()  

@app.route('/predict/', methods=['POST'])
# 响应POST消息的预测函数
def get_prediction():
    response = request.get_json()
    data_str = response['image']
    point = data_str.find(',')
    base64_str = data_str[point:]  # remove unused part like this: "data:image/jpeg;base64,"
    image = base64.b64decode(base64_str) # base64图像解码
    img = Image.open(io.BytesIO(image)) # 打开文件
    if (img.mode != 'RGB'):
        img = img.convert("RGB")
    save_path = str(Path(args['source']) / Path("img4predict.jpg")) # 保存路径
    img.save(save_path) # 保存文件
    # img.save("./frontend/static/images/img4predict.jpg")  

    # 预测图像及后处理
    predict_and_postprocess(model, exp, trt_file, decoder, args)
    results = {"results": DetectedInfo.boxes_detected}
    print(results)

    return jsonify(results)

@app.after_request
def add_headers(response):
    # 允许跨域
    response.headers.add('Access-Control-Allow-Origin', '*') 
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')




