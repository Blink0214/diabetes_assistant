import pandas as pd
from sanic import Sanic, response
import torch
import logging as log
from model.micn import MICN
from utils.timefeature import time_features

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device(f"cuda:0")
    log.info("GPU Available. Using GPU[%d]: %s", 0, torch.cuda.get_device_name(0))
else:
    log.info("No GPU available. Using CPU.")


def eval_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    a = checkpoint['params']['args']
    model = MICN(in_features=a.in_features, out_features=a.out_features,
                 seq_len=a.seq_len, pred_len=a.pred_len, num_hidden=a.num_hidden,
                 mic_layers=a.mic_layers, dropout=a.dropout, freq=a.freq, device=device,
                 decomp_kernel=a.decomp_kernel, conv_kernel=a.conv_kernel,
                 isometric_kernel=a.isometric_kernel).float().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['params']['mean'], checkpoint['params']['std'], model.eval()


app = Sanic("seg-rec-app")
app.ctx.mean, app.ctx.std, app.ctx.model = eval_model('micn.pth')


@app.post('/predict')
async def predict(request):
    seq = request.json['seq']
    seq = [[pd.to_datetime(s[0], format='%Y-%m-%d %X'), s[1]] for s in seq]

    rst = []
    date = []
    ot = []
    for elem in seq:
        date.append(elem[0])
        ot.append(elem[1])
    ot = (ot - app.ctx.mean) / app.ctx.std
    for i in range(32):
        date.append(date[-1] + pd.offsets.Minute(15))
    ipt = pd.DataFrame(date, columns=['date'])
    time_stamp = time_features(ipt, freq='min')
    data = torch.Tensor(ot).reshape([1, 96, 1]).float().to(device)
    mark = torch.from_numpy(time_stamp).reshape([1, 128, 5]).float().to(device)
    pred = app.ctx.model(data, mark)
    rst = pred.reshape(-1, ).tolist()
    rst = rst * app.ctx.std + app.ctx.mean

    results = [[i.strftime('%Y-%m-%d %X'), j] for i, j in zip(date[-32:], rst)]
    return response.json({'results': results})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8989)
