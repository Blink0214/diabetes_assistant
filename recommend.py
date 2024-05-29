import os
from flask import Flask, request, jsonify
import pandas as pd
from experiments.expsimu import Expsimu
from experiments.expknn import literals

app = Flask(__name__)

# 自定义数据结构体
class ResponseData:
    def __init__(self, code, message, recommendation):
        self.code = code
        self.message = message
        self.recommendation = recommendation

@app.route('/')
def hello():
    return 'Hello, Flask!'

# @app.route('/greet')
# def greet_route():
#     name = request.args.get('name', 'World')
#     return greet(name)

# @app.route('/greeter')
# def greeter_route():
#     name = request.args.get('name', 'World')
#     greeter = Greeter(name)
#     return greeter.greet()


@app.route('/api/data', methods=['POST'])
def get_else_data():
    data = request.get_json()
    name = data.get('name')
    age = data.get('age')
    return jsonify({'name': name, 'age': age})


@app.route('/api/recommend', methods=['POST'])
def get_recommend_data():
    data = request.get_json()

    # print("数据：",data)

    # 解析JSON数据并转换为DataFrame
    df = pd.DataFrame(data['data'])

    # 将column1列转换为日期时间类型
    df['date'] = pd.to_datetime(df['date'])

    # 将column2列转换为浮点数类型
    df['OT'] = df['OT'].astype(float)

    # print("dataframe:",df.head())


    conv_kernel = [48, 32, 24]
    isometric_kernel = []
    # seq_len = 480 # 为什么是480？
    seq_len = 96 # 一天的数据个数

    # TODO
    # 确保卷积核的大小在进行分解操作时是奇数。在卷积操作中，通常使用奇数大小的卷积核可以确保卷积操作的中心位置是存在的，从而更好地捕捉到数据的局部信息。
    # //是整数除法，取结果的整数部分A
    for ii in conv_kernel:
        if ii % 2 == 0:  # the kernel of decomposition operation must be odd
            isometric_kernel.append((seq_len + ii) // ii)
        else:
            isometric_kernel.append((seq_len + ii - 1) // ii)
    
    exp = Expsimu(name='adjust', num_embed=64, num_hidden=8, learning_rate=0.01, seq_len=seq_len, in_features=1,
                  freq='min', classes=93,
                  lgf_layers=1, conv_kernel=conv_kernel, isometric_kernel=isometric_kernel,
                  seed=2024, subject='adolescent', dataset_dir=os.path.join('.', 'datasets', 'trainset'))


    # exp.train()
    result = exp.test_knn(input_data=df)
    # exp.next_phase()

    print("模型结果：",result)

    # 将 PyTorch 张量转换为 Python 列表
    result_list = result.flatten().tolist()
    therapy_list = []
    for item in result_list:
        print(item)
        therapy_list.append(literals[item])
    

    # 创建 ResponseData 对象
    response_data = ResponseData(code=200, message='请求成功', recommendation = therapy_list)

    response_dict = {'code': response_data.code, 'message': {'text': response_data.message, 'recommendation': response_data.recommendation}}

    

    # # 如果有 recommendation 字段，则将其添加到 message 字段中
    # if response_data.recommendation:
    #     response_dict['message']['recommendation'] = response_data.recommendation
    
    
    # 返回 JSON 格式的数据
    return jsonify(response_dict)




# localhost
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3300, debug=True)
