class TensorKeys(list):
    def __init__(self, refs):
        super().__init__(refs)

    def __contains__(self, item):
        try:
            return super().__contains__(item.ref())
        except AttributeError:
            return super().__contains__(item.experimental_ref())


class TensorDict(dict):
    def __init__(self):
        super().__init__()
        # self.d = {}

    def __setitem__(self, key, value):
        try:
            super().__setitem__(key.ref(), value)
        except AttributeError:
            super().__setitem__(key.experimental_ref(), value)

    def __getitem__(self, item):
        try:
            return super().__getitem__(item.ref())
        except AttributeError:
            return super().__getitem__(item.experimental_ref())

    def keys(self):
        return TensorKeys(super().keys())
