import gzip
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class Moving_MNIST(Dataset):

    def __init__(self, args, split):

        super(Moving_MNIST, self).__init__()

        with gzip.open(args.train_data_dir, 'rb') as f:
            self.datas = np.frombuffer(f.read(), np.uint8, offset=16)
            self.datas = self.datas.reshape(-1, *args.image_size)

        if split == 'train':
            self.datas = self.datas[args.train_samples[0]: args.train_samples[1]]
        else:
            self.datas = self.datas[args.valid_samples[0]: args.valid_samples[1]]

        self.image_size = args.image_size
        self.input_size = args.input_size
        self.step_length = args.step_length
        self.num_objects = args.num_objects

        self.num_frames_input = args.num_frames_input
        self.num_frames_output = args.num_frames_output
        self.num_frames_total = args.num_frames_input + args.num_frames_output

        print('Loaded {} {} samples'.format(self.__len__(), split))

    def _get_random_trajectory(self, seq_length):

        assert self.input_size[0] == self.input_size[1]
        assert self.image_size[0] == self.image_size[1]

        canvas_size = self.input_size[0] - self.image_size[0]

        x = random.random()
        y = random.random()

        theta = random.random() * 2 * np.pi

        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)

        for i in range(seq_length):

            y += v_y * self.step_length
            x += v_x * self.step_length

            if x <= 0.: x = 0.; v_x = -v_x;
            if x >= 1.: x = 1.; v_x = -v_x
            if y <= 0.: y = 0.; v_y = -v_y;
            if y >= 1.: y = 1.; v_y = -v_y

            start_y[i] = y
            start_x[i] = x

        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)

        return start_y, start_x

    def _generate_moving_mnist(self, num_digits=2):

        data = np.zeros((self.num_frames_total, *self.input_size), dtype=np.float32)

        for n in range(num_digits):

            start_y, start_x = self._get_random_trajectory(self.num_frames_total)
            ind = np.random.randint(0, self.__len__())
            digit_image = self.datas[ind]

            for i in range(self.num_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.image_size[0]
                right = left + self.image_size[1]
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]

        return data

    def __getitem__(self, item):

        num_digits = random.choice(self.num_objects)
        images = self._generate_moving_mnist(num_digits)

        inputs = torch.from_numpy(images[:self.num_frames_input]).permute(0, 3, 1, 2).contiguous()
        targets = torch.from_numpy(images[self.num_frames_output:]).permute(0, 3, 1, 2).contiguous()
        # print(inputs.shape)
        # print(targets.shape)
        return inputs / 255., targets / 255.

    def __len__(self):
        return self.datas.shape[0]


class Moving_MNIST_Test(Dataset):
    def __init__(self, args):
        super(Moving_MNIST_Test, self).__init__()

        self.num_frames_input = args.num_frames_input
        self.num_frames_output = args.num_frames_output
        self.num_frames_total = args.num_frames_input + args.num_frames_output

        self.dataset = np.load(args.test_data_dir)
        self.dataset = self.dataset[..., np.newaxis]
        
        print('Loaded {} {} samples'.format(self.__len__(), 'test'))
        
    def __getitem__(self, index):
        images =  self.dataset[:, index, ...]

        inputs = torch.from_numpy(images[:self.num_frames_input]).permute(0, 3, 1, 2).contiguous()
        targets = torch.from_numpy(images[self.num_frames_output:]).permute(0, 3, 1, 2).contiguous()
        return inputs / 255., targets / 255.

    def __len__(self):
        return len(self.dataset[1])
        
class TaxiBJDataset(Dataset):
    def __init__(self, args, mode='train'):
        super(TaxiBJDataset, self).__init__()
        self.num_frames_input = args.num_frames_input
        self.num_frames_output = args.num_frames_output
        self.num_frames = self.num_frames_input + self.num_frames_output
        self.root = args.root_path.format(mode)
        self.samples_list = self._prepare_samples_list()

        print(f'Loaded {len(self.samples_list)} samples')

    def _prepare_samples_list(self):
        samples_list = []
        for folder in os.listdir(self.root):
            folder_path = os.path.join(self.root, folder)
            image_list = os.listdir(folder_path)
            for i in range(len(image_list) - self.num_frames + 1):
                samples_list.append([
                    os.path.join(folder_path, image_list[j]) for j in range(i, i + self.num_frames)
                ])
        return samples_list

    def _load_data(self, paths):
        return np.stack([np.fromfile(path, dtype=np.float32).reshape(2, 32, 32) for path in paths])

    def __getitem__(self, index):
        paths = self.samples_list[index]
        data = self._load_data(paths)
        inputs = torch.from_numpy(data[:self.num_frames_input])
        outputs = torch.from_numpy(data[self.num_frames_output:])
        return inputs, outputs

    def __len__(self):
        return len(self.samples_list)


class HumanDataset(Dataset):
    def __init__(self, args, mode='train'):
        super(HumanDataset, self).__init__()

        self.train_person = ['s_01', 's_05', 's_06', 's_07', 's_08']
        self.test_person = ['s_09', 's_11']
        self.path = args.root_path
        self.num_frames_input = args.num_frames_input
        self.num_frames_output = args.num_frames_output
        self.num_frames = self.num_frames_input + self.num_frames_output

        self.samples_list = self._prepare_samples_list(mode)

        print(f'Loaded {len(self.samples_list)} samples ({mode})')

    def _prepare_samples_list(self, split):
        persons = self.train_person if split == 'train' else self.test_person
        file_list = [f for f in os.listdir(self.path) if any(p in f for p in persons)]
        samples_list = []

        for file in file_list:
            image_list = os.listdir(os.path.join(self.path, file))
            for i in range(len(image_list) - self.num_frames + 1):
                dataname_list = [
                    os.path.join(self.path, file, image_list[j])
                    for j in range(i, i + self.num_frames)
                ]
                samples_list.append(dataname_list)

        return samples_list

    def _load_data(self, data_path_list):
        images = [np.asarray(Image.open(path)) for path in data_path_list]
        return np.stack(images)

    def __getitem__(self, index):
        images = self._load_data(self.samples_list[index])

        inputs = torch.from_numpy(images[:self.num_frames_input]).float().permute(0, 3, 1, 2) / 255.
        outputs = torch.from_numpy(images[self.num_frames_input:]).float().permute(0, 3, 1, 2) / 255.

        return inputs, outputs

    def __len__(self):
        return len(self.samples_list)


class KTHDataSet(Dataset):
    def __init__(self, args, mode='train'):
        super(KTHDataSet, self).__init__()
        self.mode = mode
        self.elements = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
        self.train_person =  ['person01', 'person02', 'person03', 'person04', 'person05', 'person06', 'person07',
                             'person08', 'person09', 'person10', 'person11', 'person12', 'person13', 'person14',
                             'person15', 'person16']
        self.test_person = ['person17', 'person18', 'person19', 'person20', 'person21', 'person22', 'person23',
                            'person24', 'person25']
        self.samples_list = self._prepare_samples_list(args)

        print(f'Loaded {len(self.samples_list)} samples ({mode})')

    def _prepare_samples_list(self, args):
        samples_list = []
        persons = self.train_person if self.mode == 'train' else self.test_person
        total_frames = 20 if self.mode == 'train' else 30  # test:10 -> 20
        # total_frames = 20 if self.mode == 'train' else 50   # test:10 -> 40

        for element in self.elements:
            root_path = args.root_path.format(element)
            file_list = [f for f in os.listdir(root_path) if any(p in f for p in persons)]

            for file in file_list:
                path = os.path.join(root_path, file)
                image_list = os.listdir(path)
                end_ = len(image_list) - total_frames + 1
                step = 1 if self.mode == 'train' else 20 if element not in ['jogging', 'running'] else 3

                for i in range(0, end_, step):
                    data_list = [os.path.join(path, image_list[j]) for j in range(i, i + total_frames)]
                    samples_list.append(data_list)

        return samples_list

    def _load_data(self, data_path_list):
        images = [np.asarray(Image.open(path))[:, :, np.newaxis] for path in data_path_list]
        return np.stack(images)

    def __getitem__(self, index):
        images = self._load_data(self.samples_list[index])
        # print(images.shape)

        inputs = torch.from_numpy(images[:10]).float().permute(0, 3, 1, 2) / 255.
        targets = torch.from_numpy(images[10:]).float().permute(0, 3, 1, 2) / 255.

        return inputs, targets

    def __len__(self):
        return len(self.samples_list)