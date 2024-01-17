# Tuberculosis Detection by Convolutional Neural Networks
This project is an attempt to create a neural network that can detect tuberculosis at a reasonable accuracy from purely x-ray images.

Overall results are about 70-80% accuracy and 80-85% recall score.

# Dataset
**Dataset not included on Github because it's 4gb** 
Download from the Kaggle link.

Pulmonary Chest X-Ray Abnormalities from K Scott Mader
Available on Kaggle at https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities

# Installation/Execution
1. Make sure you have all the libraries installed.
2. Download the dataset from the Kaggle link.
3. Create a folder in the same folder as cp468.ipynb named "input"
4. Move all the files from \archive\ChinaSet_AllFiles\ChinaSet_AllFiles\CXR_png\ to \input\china_xrays\
5. Move all files from \archive\Montgomery\MontgomerySet\CXR_png\ to \input\montgomery_xrays\

File structure should look like this:

-cp468.ipynb

-input

--china_xrays

--\-xray images from the china set

--montgomery_xrays

--\-xray images from the montgomery set

# Performance Parameters
Most weight was put in recall, to measure how accurate the model was when the patient truly has tuberculosis.
Loss & Accuracy was used to make sure the model was not faulty. 
i.e If the model says every patient has tuberculosis, it has a perfect recall score.

# About Developers
Nicholas Sam, 3rd year, Wilfrid Laurier University

James Puppin, 4th year, Wilfrid Laurier University

# License
Copyright (c) 2022, Nicholas Sam, James Puppin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# Acknowledgements
Jaeger S, Karargyris A, Candemir S, Folio L, Siegelman J, Callaghan F, Xue Z, Palaniappan K, Singh RK, Antani S, Thoma G, Wang YX, Lu PX, McDonald CJ. Automatic tuberculosis screening using chest radiographs. IEEE Trans Med Imaging. 2014 Feb;33(2):233-45. doi: 10.1109/TMI.2013.2284099. PMID: 24108713

Candemir S, Jaeger S, Palaniappan K, Musco JP, Singh RK, Xue Z, Karargyris A, Antani S, Thoma G, McDonald CJ. Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. IEEE Trans Med Imaging. 2014 Feb;33(2):577-90. doi: 10.1109/TMI.2013.2290491. PMID: 24239990