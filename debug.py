from easydict import EasyDict
from materials.datasets import FlyingChairs


def debug_FlyingChairs():
    args = EasyDict(
        crop_size = [256, 256],
        inference_size = [-1,-1],
    )
    flying_chairs = FlyingChairs(args, is_cropped = True, root="/root/autodl-tmp/data/FlyingChairs_release/data")
    [images], [flow] = flying_chairs[0]
    print(type(images))
    print(type(flow))

    print(images.shape)
    print(flow.shape)


if __name__ == "__main__":
    debug_FlyingChairs()
