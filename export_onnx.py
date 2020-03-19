import torch
import torch.onnx as onnx
import yolact
import onnxruntime
import os
from torch.autograd import Variable

__all__ = ['onnx_export']

def onnx_export(model, dataset, export):
    batch_data = dataset.pull_item(0)
    batch_data = torch.unsqueeze(batch_data[0].cuda(), 0)
    cur_path = os.getcwd()
    onnx_path = os.path.join(cur_path, export)
    onnx.export(model, batch_data, onnx_path, verbose=True, opset_version=10)

    # forward pytorch
    img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(0)
    torch_out = {}
    torch_out['loc'], torch_out['conf'], torch_out['mask'],\
        torch_out['priors'], torch_out['proto']= \
            model(Variable(img.unsqueeze(0)).cuda())
    import pdb; pdb.set_trace()
    ort_session = onnxruntime.InferenceSession(onnx_path)
