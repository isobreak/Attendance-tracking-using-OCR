import numpy as np
from Levenshtein import distance


def get_present_student_ids(acceptable_names: list[list[str]], recognised_clusters: list[list[str]], thresh):
    """
    Finds present students based on recognised clusters and acceptable names
    Args:
        acceptable_names: list of acceptable names
        recognised_clusters: list of recognised clusters

    Returns:
        Dictionary with 'ids', 'scores' and 'full_names' that have been found
    """
    # preprocessing
    acceptable_names = [[name_part.lower() for name_part in name] for name in acceptable_names]
    recognised_clusters = [[token.lower() for elem in cluster for token in elem.split()] for cluster in recognised_clusters]

    mat_outer = np.zeros((len(acceptable_names), len(recognised_clusters)), dtype=np.float32)
    for i, name in enumerate(acceptable_names):
        for j, cluster in enumerate(recognised_clusters):
            mat_inner = np.zeros((len(name), len(cluster)), dtype=np.float32)
            at_least_one = False
            for m, name_part in enumerate(name):
                for n, token in enumerate(cluster):
                    d = distance(token, name_part) / len(name_part)
                    if 1 - d > thresh:
                        mat_inner[m, n] = 1 - d
                        at_least_one = True
            if at_least_one:
                while mat_inner.shape[1] and mat_inner.shape[0] and np.max(mat_inner):
                    i_0, i_1 = np.unravel_index(np.argmax(mat_inner), mat_inner.shape)
                    mat_outer[i, j] += mat_inner[i_0, i_1]
                    mat_inner = np.delete(mat_inner, i_0, 0)
                    mat_inner = np.delete(mat_inner, i_1, 1)
            else:
                continue

    # get recognised ids
    recognised_ids = []
    scores = []
    rec_ids2 = []
    while mat_outer.shape[1] and mat_outer.shape[0] and np.max(mat_outer):
        i, j = np.unravel_index(np.argmax(mat_outer), mat_outer.shape)
        recognised_ids.append(i.item())
        rec_ids2.append(j)
        scores.append(mat_outer[i, j].item())
        mat_outer[:, j] = np.zeros(mat_outer.shape[0], dtype=np.float32)
        mat_outer[i, :] = np.zeros(mat_outer.shape[1], dtype=np.float32)

    for k in range(len(rec_ids2)):
        print(recognised_clusters[rec_ids2[k]])
        print(acceptable_names[recognised_ids[k]])
        print(scores[k])
        print()

    return {
        'ids': recognised_ids,
        'scores': scores,
        'full_names': [acceptable_names[k] for k in recognised_ids],
    }
