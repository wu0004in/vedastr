import copy
import math
import random

from torch.utils.data import DistributedSampler


class DistributedBalanceSampler(DistributedSampler):

    def __init__(self, dataset, oversample, downsample, batch_size, **kwargs):
        assert hasattr(dataset, 'data_range')
        assert hasattr(dataset, 'batch_ratio')
        self.oversample = oversample
        self.downsample = downsample
        self.samples_range = dataset.data_range
        self.batch_ratio = dataset.batch_ratio
        self.batch_size = batch_size

        super(DistributedBalanceSampler, self).__init__(dataset=dataset, **kwargs)
        self._generate_indices_()
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self._num_samples % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self._num_samples - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(self._num_samples / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def _generate_indices_(self):
        self._num_samples = len(self.dataset)
        indices_ = []
        # TODO, elegant
        for idx, v in enumerate(self.samples_range):
            if idx == 0:
                temp = list(range(v))
                if self.shuffle:
                    random.shuffle(temp)
                indices_.append(temp)
            else:
                temp = list(range(self.samples_range[idx - 1], v))
                if self.shuffle:
                    random.shuffle(temp)
                indices_.append(temp)
        if self.oversample:
            indices_ = self._oversample(indices_)
        if self.downsample:
            indices_ = self._downsample(indices_)
        return indices_

    def _oversample(self, indices):
        max_len = max([len(index) for index in indices])
        result_indices = []
        for idx, index in enumerate(indices):
            current_nums = len(index)
            need_num = max_len - current_nums
            total_nums = need_num // current_nums
            mod_nums = need_num // current_nums
            init_index = copy.copy(index)
            for _ in range(max(0, total_nums - 1)):
                new_index = copy.copy(init_index)
                if self.shuffle:
                    random.shuffle(new_index)
                index += new_index
            index += random.sample(index, mod_nums)
            result_indices.append(index)
        self._num_samples = max_len * len(indices)
        return result_indices

    def _downsample(self, indices):
        min_len = min([len(index) for index in indices])
        result_indices = []
        for idx, index in enumerate(indices):
            index = random.sample(index, min_len)
            result_indices.append(index)
        self._num_samples = min_len * len(indices)
        return result_indices

    def __iter__(self):
        indices_ = self._generate_indices_()
        total_nums = len(self) // self.batch_size
        sizes = [int(self.batch_size * br) for br in self.batch_ratio]
        final_index = [total_nums * size for size in sizes]
        indices = []
        for idx2 in range(total_nums):
            for idx3, size in enumerate(sizes):
                indices += indices_[idx3][idx2 * size:(idx2 + 1) * size]
        for idx4, index in enumerate(final_index):
            indices += indices_[idx4][index:]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
