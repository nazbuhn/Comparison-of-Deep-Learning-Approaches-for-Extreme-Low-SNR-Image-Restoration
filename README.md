<h1>Comparison of deep learning approaches for extreme low-SNR image restoration software and analytics code</h1>

<h2>Overview:</h2>
<h3>Title: Comparison of Deep Learning Approaches for Extreme Low-SNR Image Restoration</h3>
<h4>Authors:</h4> Nasreen Elizabeth Buhn<sup>1</sup>, Sriya Reddy Adunur<sup>2</sup>, Joseph Hamilton<sup>3</sup>, Summer Levis<sup>3</sup>, Guy M. Hagen<sup>3</sup>, Jonathan D. Ventura<sup>2</sup>
<br>
<br>
<sup>1</sup>Biological Sciences Department, California Polytechnic State University, San Luis Obispo, California, 93407 <br>
<sup>2</sup>Department of Computer Science and Software Engineering, California Polytechnic State University, San Luis Obispo, California, 93407<br>
<sup>3</sup>UCCS BioFrontiers Center, University of Colorado at Colorado Springs, 1420 Austin Bluffs Parkway, Colorado Springs, Colorado, 80918

<h4>Abstract:</h4>
<h5>Background: </h5>
Live-cell fluorescence microscopy enables the study of dynamic cellular processes. However, fluorescence microscopy can damage cells and disrupt biological processes through photobleaching and phototoxicity. Reducing a sample’s light exposure mitigates these effects but results in low signal-to-noise ratio (SNR) images. Deep learning provides a solution for restoring these low-SNR images. However, these deep learning methods require large, representative datasets for training, testing, and benchmarking, as well as substantial GPU memory, particularly for denoising large images.
<h5>Results:</h5>
We present a new fluorescence microscopy dataset designed to expand the range of imaging conditions and specimens currently available for evaluating denoising methods. The dataset contains 324 paired high/low-SNR images spanning 4.19-282.22 megapixels across 12 sub-datasets that vary in specimen, objective, staining type, excitation wavelength, and exposure time. The dataset also includes spinning disk confocal microscopy examples and extreme-noise cases. We evaluated three state-of-the-art deep learning denoising models on the dataset: a supervised transformer-based model, a supervised CNN model, and an unsupervised single image model. We also developed an image stitching method that enables large images to be processed in smaller crops and reconstructed. 
<h5>Conclusions:</h5>
Our dataset provides a diverse benchmark for evaluating deep learning denoising methods, and our stitching method provides a solution to GPU memory constraints encountered when processing large images. Among the evaluated deep learning models, the supervised transformer-based model had the top denoising performance but required the longest training time. 


<h2>File descriptions:</h2>
<br><b>External/Restormer: </b>Forked modified Restormer model. Edited code description is included in UPDATE.md.

<br><b>CARE: </b>Contains training (train_care_generator.py) and testing (test_care.py) script used for CARE implementation.(https://github.com/CSBDeep/CSBDeep/commits/main/ commit:282664fc294e8ba5d00a6ea82fcadcf9198a24b9)

<br><b>N2F: </b>Contains N2F (N2F.py) and test (run.py) script used in Noise2Fast impletmentation. (https://github.com/jason-lequyer/Noise2Fast/commits/main/ commit: 8b244699ee84eb057b77cbcedc9dc6f5a4a434b8)

<br><b>Adaptive_image_stitching.py:</b> Implementation of the adaptive image-stitching logic described in paper (see the “Adaptive image stitching” section).
