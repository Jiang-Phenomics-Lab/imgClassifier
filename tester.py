from abc import ABCMeta, abstractclassmethod
from collections.abc import Mapping, Sequence
from tqdm import tqdm
import torch

class BaseTester(metaclass=ABCMeta):
    @abstractclassmethod
    def __init__(self):
        pass

    @abstractclassmethod
    def test(self):
        pass
class Tester(BaseTester):
    def __init__(self, model, dataloader, resume_dict=None, name=None) -> None:
        self.model = model
        self.dataloader = dataloader
        self.vectors = None
        self.labels = None
        if resume_dict is not None:
            if type(resume_dict) is str:
                resume_dict = torch.load(resume_dict)
            self.model.load_state_dict(resume_dict['model_state_dict'])
    
    def test(self):
        tbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        vector_list = []
        label_list = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in tbar:
                if type(data) is tuple:
                    data = tuple(d.cuda() for d in data)
                    model_output = [self.model(d) for d in data]
                elif type(data) is torch.Tensor:
                    model_output = self.model(data.cuda())
                else:
                    model_output = self.model(data)
                vector_list.append(model_output)
                label_list.append(target)
        self.vectors = torch.cat(vector_list, 0)
        self.labels = self._collate_labels(label_list)
        return self.vectors, self.labels

    def _collate_labels(self, label_list):
        element_0 = label_list[0]
        if isinstance(element_0, torch.Tensor):
            return torch.cat(label_list, 0).numpy()
        elif isinstance(element_0, Mapping):
            result = {}
            for key in element_0.keys():
                inner_val = element_0[key]
                if isinstance(inner_val, Sequence):
                    result[key] = []
                    for row in label_list:
                        result[key].extend(row[key])
                elif isinstance(inner_val, torch.Tensor):
                    tensor_list = [i[key] for i in label_list]
                    result[key] = torch.cat(tensor_list, 0)
                else:
                    raise TypeError(f"Unkonwn type of label in key{key}: {type(inner_val)}")
            return result
        else:
            raise TypeError(f"Unkonwn type of labels: {type(element_0)}")

            