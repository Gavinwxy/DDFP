import logging

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, d_list, **kwargs):
        # parse the input list
        self.parse_input_list(d_list, **kwargs)

    def parse_input_list(self, d_list, max_sample=-1, start_idx=-1, end_idx=-1):
        logger = logging.getLogger("global")
        assert isinstance(d_list, str)
        if "cityscapes" in d_list:
            self.list_sample = []
            for line in open(d_list, "r"):
                total_index = line.strip().split('\t')[0][8:-4]
                mode, index = total_index.split('/')
                city = index.split('_')[0]
                comp_path = mode + '/' + city + '/' + index
                
                self.list_sample.append([
                    'leftImg8bit/' + comp_path + '_leftImg8bit.png',
                    "gtFine/" + comp_path + "_gtFine_labelTrainIds.png"
                    ])
                
        elif "pascal" in d_list or "VOC" in d_list:
            self.list_sample = [
                [
                    "JPEGImages/{}.jpg".format(line.strip()),
                    "SegmentationClassAug/{}.png".format(line.strip()),
                ]
                for line in open(d_list, "r")
            ]
        else:
            raise "unknown dataset!"

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        logger.info("# samples: {}".format(self.num_sample))

    def img_loader(self, path, mode):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert(mode)

    def __len__(self):
        return self.num_sample
