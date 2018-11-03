from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

Estimator = TfPoseEstimator


def get_estimator(model='cmu', resize='0x0', resize_out_ratio=4.0):
    w, h = model_wh(resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    return e


def infer(image, model='cmu', resize='0x0', resize_out_ratio=4.0):
    """

    :param image:
    :param model:
    :param resize:
    :param resize_out_ratio:
    :return:
    """
    w, h = model_wh(resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(image, None, None)
    if image is None:
        raise Exception('Image can not be read, path=%s' % image)
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

    return humans


if __name__ == '__main__':
    import time

    start = time.time()
    print infer('golf.jpg', model='mobilenet_thin')
    end = time.time()
    print(end - start)
