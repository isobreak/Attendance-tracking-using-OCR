# Attendance-tracking-using-OCR

The repository contains the service's code for tracking student attendance using OCR. Based on the image and the list of possible student names, the system is able to recognize the list of those present. Model's weights and all image examples could be found [here](https://drive.google.com/file/d/1-mWXj8_PRp6vd7_lxeKN3GHaEbkIIOw6/view?usp=sharing). Some of the examples are available in [data/results](data/results).

## System's Purpose
The use of sheets for self-recording of students present at lectures is a common practice at MEPhI. To automate the process of entering attendance data into the database, this service is used, which provides the opportunity to extract information about students present based on a worksheet.

## Approach used for solving the Task
The recognition pipeline is implemented in the [processing.py](src/app/src/processing.py) while each pipeline step is adjusted using the code in corresponding directory [in src](src/), and it consists of 4 stages:

1. **Word Detection**
   - There are 3 detection approaches implemented, which are based on:
     - [**DBSCAN**](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html): It clusters contour pixels based on their location - each cluster is considered a detected object. It requires further development (an optimized custom distance metric that takes Y into account to a greater extent than X). It is sensitive to the quality of image binarization. It can be used on images with low text density.
     - [**Morphological Transformations**](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html): This is the fastest and least accurate of the methods. It is sensitive to the quality of image binarization. It can be used on images with low text density and strictly horizontal text arrangement.
     - [**Faster R-CNN**](https://arxiv.org/pdf/1506.01497): This has the best accuracy on complex data among the algorithms used, and the lowest speed. It is used in the final pipeline (inference.py).

2. **Word Clustering**
   - Clustering is carried out in order to separate all identified objects into groups containing information about one student. Assuming that students sign their names in separate lines, clustering is performed on the average Y value using [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

3. **Word Recognition**
   - [**TrOCR**](https://huggingface.co/raxtemur/trocr-base-ru) is used to recognize text within each bounding box.

4. **Full Name Matching**
   - The comparison is made using [Levenshtein distances](https://ru.wikipedia.org/wiki/Расстояние_Левенштейна), normalized by the length of the corresponding words.

## Pipeline examples
More examples of each pipeline's stage could be found in [data/results/](data/results) directory. <br>

### Example of Faster R-CNN detection and clustering with DBSCAN
&emsp;Different clusters correspond to different colors:<br>
<img src="data\results\clustering\5440470954456245678.jpg" alt="Example of Faster R-CNN detection and clustering with DBSCAN" width="50%"><br>
&emsp;DBSCAN's eps=8, image resolution: 800x800
### Example of recognised texts for each cluster
<img src="data\results\recognition\5440470954456245678.jpg" alt="Example of recognised texts for each cluster" width="50%"><br>
## Inference speed
Average inference speed: **5 sec/image** (on RTX 4070 Super, 12 Gb)

## Project's structure

data<br />
&emsp;└── results&emsp;&emsp;&emsp;#Image examples of each pipeline's stage **(TRUNCATED IN REPO due to large size)**<br />
&emsp;&emsp;├── clustering/&emsp;&emsp;#Clustering results for different eps values.<br />
&emsp;&emsp;├── detection/&emsp;&emsp;#Detection and postprocessing results<br />
&emsp;&emsp;└── recognition/&emsp;&emsp;#Recognition results<br />
src<br />
&emsp;├── app&emsp;&emsp;# Current service realization<br />
&emsp;&emsp;├── data/&emsp;&emsp; Weights and default students' database (MISSED IN REPO)<br />
&emsp;&emsp;└── huggingface/&emsp;&emsp;#Cached models (MISSED IN REPO)<br />
&emsp;&emsp;└── src<br />
&emsp;&emsp;&emsp;├── constants<br />
&emsp;&emsp;&emsp;├── main&emsp;&emsp;&emsp;&emsp;#FastAPI backend<br />
&emsp;&emsp;&emsp;└── processing&emsp;&emsp;#Core logic of the service<br />
&emsp;#Training / exploration source codes<br>
&emsp;├── comparison/&emsp;&emsp;&emsp;#Codes for Full Names Matching algorithms' investigation<br/>
&emsp;├── detection&emsp;# Implementation of detection using Faster R-CNN and the final pipeline<br />
&emsp;&emsp;├── constants.py&emsp;#Constants used in the detection module<br />
&emsp;&emsp;├── EDA.py&emsp;&emsp;&emsp;# Dataset visualization<br />
&emsp;&emsp;├── models.py&emsp;&emsp;# Used models<br />
&emsp;&emsp;└── train.py&emsp;&emsp;&emsp;# Training Faster R-CNN on the [ai-forever/school_notebooks_RU](https://huggingface.co/datasets/ai-forever/school_notebooks_RU) dataset<br />
&emsp;├── detection_classic&emsp;# Implementation of classic detection approaches (implemented [trackbars](https://docs.opencv.org/3.4/da/d6a/tutorial_trackbar.html) for adjusting processing parameters)<br />
&emsp;&emsp;├── center_clustering.py&emsp;#Using DBSCAN on contour centers: speeds up processing<br />
&emsp;&emsp;├── dbscan_approach.py&emsp;#Using DBSCAN for detection<br />
&emsp;&emsp;└── dilation_approach.py&emsp;#Using morphological transformations for detection<br />
&emsp;├── embeddings/&emsp;&emsp;&emsp;&emsp;&emsp;#Codes for metric learning approach investigation<br/>
&emsp;└── optimized/&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;#ONNX optimization codes<br/>
## Installation via Docker
This way you can get the most up-to-date version of the service.<br>
### Install and Run Docker image: [kiryaz/asa_ocr](https://hub.docker.com/r/kiryaz/asa_ocr)
&emsp;To have access to CUDA inference run container with **--gpus all** flag. **If you have less than 4 GB**, inference will still use the **CPU** instead.<br>
&emsp;Default port: 8000.
## Installation via PIP
Pull repo and run **pip install -r requirements.txt**<br>
Python version: 3.9