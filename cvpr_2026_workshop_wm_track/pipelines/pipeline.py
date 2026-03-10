
class BasePipeline:
    def to(self, device):
        return self

    def __call__(self, *args, **kwargs):
        raise NotImplementedError