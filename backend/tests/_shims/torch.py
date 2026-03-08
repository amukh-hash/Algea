class _DummyDevice:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name


def device(name):
    return _DummyDevice(name)


class _Cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def get_device_name(_idx):
        return "cpu"

    @staticmethod
    def get_device_properties(_idx):
        class P:
            total_mem = 0
        return P()


cuda = _Cuda()


class _NN:
    class Module:
        pass

    class DataParallel:
        def __init__(self, *args, **kwargs):
            pass


nn = _NN()
