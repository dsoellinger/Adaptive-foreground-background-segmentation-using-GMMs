from segmentizer import Segmentizer
from segmentizer.data_loader import LASIESTADataLoader
from sklearn.metrics import f1_score
from itertools import chain
from multiprocessing import Pool


spec = [
    ('I_SI_01', '/Users/dsoellinger/Downloads/I_SI_01', '/Users/dsoellinger/Downloads/I_SI_01-GT')
]


def _background_map_conversion(rgb):

    # Background
    if rgb == [0,0,0]:
        return True

    # Moving object with label 1 along the sequence.
    if rgb == [255, 0, 0]:
        return False

    # Moving object with label 2 along the sequence.
    if rgb == [0, 255, 0]:
        return False

    # Moving object with label 3 along the sequence.
    if rgb == [255, 255, 0]:
        return False

    # Moving objects remaining static.
    if rgb == [255, 255, 255]:
        return False

    # Uncertainty pixels.
    if rgb == [128, 128, 128]:
        return False

    return None


def evaluate_performance(params):

    name, groundtruth_path, label_path = params

    y_true = []
    y_pred = []

    video_segmentizer = Segmentizer(352, 288)

    data_loader = LASIESTADataLoader(groundtruth_path, label_path)

    for i, (groundtruth_frame, label_frame) in enumerate(data_loader):

        if i == 1:
            break

        predicted_background = video_segmentizer.fit_and_predict(groundtruth_frame)

        label_frame = label_frame.tolist()
        label_frame = [[_background_map_conversion(rgb) for rgb in row ] for row in label_frame]
        y_true += list(chain.from_iterable(label_frame))

        y_pred += list(chain.from_iterable(predicted_background))

        score = f1_score(y_true, y_pred)

    print('Finished evaluation of dataset ' + name)

    return name, score


pool = Pool(processes=10)
scores = list(pool.map(evaluate_performance, spec))
print(scores)
f = open('scores.txt', 'wt')
for name, score in scores:
    f.write(name + ';' + str(score))
    f.flush()


