import pandas as pd

from functions import get_present_student_ids

if __name__ == "__main__":
    with open('../../data/results/recognition/res.txt', 'r', encoding='utf-8') as f:
        pages = f.read()
        pages = [[cluster.split(';;;') for cluster in image.split(';;;;')] for image in
                               pages.split(';;;;;')]

    acceptable_names = pd.read_csv('../../data/test_data/students.csv').values.tolist()

    for recognised_clusters in pages:
        res = get_present_student_ids(acceptable_names, recognised_clusters, 0.7)
        print(res)
