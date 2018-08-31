from segmentizer import Segmentizer
from segmentizer.data_loader import LASIESTADataLoader
from sklearn.metrics import f1_score
from itertools import chain
from multiprocessing import Pool

DB_DIR = '../datasets/'

spec = [
    ('I_BS_01', DB_DIR + 'I_BS_01', DB_DIR + 'I_BS_01-GT'),
    ('I_CA_01', DB_DIR + 'I_CA_01', DB_DIR + 'I_CA_01-GT'),
    ('I_IL_01', DB_DIR + 'I_IL_01', DB_DIR + 'I_IL_01-GT'),
    ('I_MB_01', DB_DIR + 'I_MB_01', DB_DIR + 'I_MB_01-GT'),
    ('I_MC_01', DB_DIR + 'I_MC_01', DB_DIR + 'I_MC_01-GT'),
    ('I_OC_01', DB_DIR + 'I_OC_01', DB_DIR + 'I_OC_01-GT'),
    ('I_SI_01', DB_DIR + 'I_SI_01', DB_DIR + 'I_SI_01-GT'),
    ('I_SM_01', DB_DIR + 'I_SM_01', DB_DIR + 'I_SM_01-GT')
]


def background_map_conversion(rgb):

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

    # Uncertain pixels
    if rgb == [128, 128, 128]:
        return [128, 128, 128]

    return None


def remove_uncertain_pixels(predicted_background, groundtruth):

    new_predicted_background = []
    new_groundtruth = []

    for i, val_groundtruth in enumerate(groundtruth):
        if val_groundtruth != [128, 128, 128]:
            new_groundtruth.append(val_groundtruth)
            new_predicted_background.append(predicted_background[i])

    return new_predicted_background, new_groundtruth


def evaluate_performance(params):

    name, original_path, label_path = params

    y_true = []
    y_pred = []

    video_segmentizer = Segmentizer(352, 288)

    data_loader = LASIESTADataLoader(original_path, label_path)

    for i, (original_frame, label_frame) in enumerate(data_loader):

        predicted_background = video_segmentizer.fit_and_predict(original_frame)

        label_frame = label_frame.tolist()
        label_frame = [[background_map_conversion(rgb) for rgb in row] for row in label_frame]
        y_true += list(chain.from_iterable(label_frame))
        y_pred += list(chain.from_iterable(predicted_background))

        y_pred, y_true = remove_uncertain_pixels(y_pred, y_true)

        score = f1_score(y_true, y_pred)

    print('Finished evaluation of dataset ' + name)

    return name, score


pool = Pool(processes=2)
scores = list(pool.map(evaluate_performance, spec))
print(scores)

f = open('scores.txt', 'wt')
for name, score in scores:
    f.write(name + ';' + str(score) + '\n')
    f.flush()
