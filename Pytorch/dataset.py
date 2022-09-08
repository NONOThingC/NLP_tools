class MultiFileDataset(Dataset):
    r"""Dataset as a concatenation of multiple file datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
        datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]
    """
    @staticmethod
    def cumsum_dataset(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @staticmethod
    def cumsum_num(nums):
        r, s = [], 0
        for num in nums:
            s+=num
            r.append(s)
        return r


    def read_file(self,filename):
        """
        intelligent read
        """
        if self.current_filename=="":
            self.file_io=open("Trunk" + str(filename) + ".pkl", 'rb')
        if self.current_filename != filename:
            self.file_io.close()
            self.file_io=open("Trunk" + str(filename) + ".pkl", 'rb')
        self.current_filename = filename
        return pickle.load(self.file_io)

    def __init__(self, trunk_sizes) -> None:

        self.cumulative_sizes = self.cumsum_num(trunk_sizes)
        self.trunk_count=len(self.cumulative_sizes)
        self.current_filename=''

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        import bisect
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.read_file(dataset_idx)[sample_idx]


    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)