# Furniture Style Classifier
Final project for the Building AI course

<img src="https://github.com/gasingh/furniture-style-classifier/blob/main/project_cover_image.JPG" alt="project_cover_img" width="1000">

## Story and Motivation

I was wandering around in the streets of London, and stumbled across a book outside a bookshop in a sale. It was a pictorial guide from the 1990s teaching people how to identify furniture styles. The book is meant as a concise resource for curators, furniture collectors, and connoisseurs, as well as students of history and decorative arts. The book provide one page descriptions and images for 36 styles ranging from 1600s to 1900s. Almost instantly, I was fascinated by the idea that if a person can be trained to read a book and learn to identify furniture styles, can an AI do it as well? 

<p align="center" width="100%">
  <img src="https://github.com/gasingh/furniture-style-classifier/blob/main/Capture1%20-%20Copy-fotor-20241013163213_%5Bfotor%5D.png" width="500"> <img src="https://github.com/gasingh/furniture-style-classifier/blob/main/Capture2%20-%20Copy-fotor-20241013163419_%5Bfotor%5D.jpg" width="500"> <br>
</p>

<p align="center" width="100%">
  <img src = "https://github.com/gasingh/furniture-style-classifier/blob/main/Screenshot_20241007_005720.jpg" width= "300"> <img src = "https://github.com/gasingh/furniture-style-classifier/blob/main/Screenshot_20241007_005734.jpg" width= "300"> <img src = "https://github.com/gasingh/furniture-style-classifier/blob/main/Screenshot_20241007_005837.jpg" width= "300"> <br>
</p>

(_Above 5 Photographs taken by the author_).

## Aim of the Project

The project is meant to provide an AI trained eye for furniture collectors, which can predict the best/ nearest possible visual style match for a piece of furniture! The project is inspired by a book which teaches people how to identify furniture types by understanding characteristic portions of a furniture type and attribute them to specific historical furniture styles. So the aim of the project is to learn the appropriate AI means to help find a solution that matches the way humans understand, perceive and identify images, and stylizations! 

<!-- Learn how to translate a visual guide into a textual reference for a computer program (probably a database?)
* The book has images with portions of furniture objects tagged and named, so they become clearly identifiable to the viewer. How can a set of images and related textual data/ labels be associated in a digital format, without too much manual processing? Maybe some kind of an image-OCR solution could help? Or another mini AI that can read a scan of a page, and make sense of it? -->  

## How is it used?

The idea will be to be able to send a picture to the application online onto a website, and the AI application should provide you the nearest match to the provided furniture image, and tell you why it showed you what it showed you. It should be able to label the various elements of an image based on learning typical visual features of a furniture style from an era. 

<!--
## Future Development

In the future, the developed AI tool should able to read descriptive page scans about furniture classification and typology texts and learn from it:

## Challenges

Limitations: The project doesnot look to solve 
Ethical Considerations: Ethically any images which are still in copyright should be utilized only after procuring permissions/  a solution like this? -->

## AI methods: Skills, Knowledge Acquisition and Learning

_What kind of skills, what kind of assistance would I need to move on?_

I would need to learn how to classify images using a **CNN**. I have learned that these are the kind of neural networks which can be trained to classify images. Below are two sections which assimilate some visual and textual searches and some preliminary research conducted on the subject. This should serve as a ready reference for me to start working on my project further. 

<details> <summary> Literature Review 01: Understanding CNNs </summary>  

  <br>**Google Search 01**: _cnn image classification diagram_
  [cnn image classification diagram - Google Search](https://www.google.com/search?sca_esv=81b62b241f3359c7&rlz=1C1GCEB_enGB1045GB1045&sxsrf=ADLYWIJA4sPq0NGHOn2zctn4R8i_ehBPTQ:1728608543687&q=cnn+image+classification+diagram&udm=2&fbs=AEQNm0Aa4sjWe7Rqy32pFwRj0UkWd8nbOJfsBGGB5IQQO6L3J_86uWOeqwdnV0yaSF-x2jon2iao6KWCaVjfn7ahz_sfz4kQc-hbvsXJ2gNx0RnV2nl305mvoek0YK94ylYY2a4b3Q-OEwW5lKppi2wujywZWmdIJVp8wrsv_g-eh5sWEDXx8JNpgmjsiKj2mZMvftPlZJZz&sa=X&ved=2ahUKEwjR3M7RkIWJAxUo3wIHHRD1KZMQtKgLegQIDhAB&biw=1383&bih=747&dpr=2.5#vhid=H2hl3BjvD5pswM&vssid=mosaic) <br>
   
  * **i. Understanding CNNs for Image Classification**
    <br><img src="https://media.licdn.com/dms/image/D5612AQFnSLkzkkXIqg/article-cover_image-shrink_720_1280/0/1687713196637?e=2147483647&v=beta&t=2pX0mP5aH9rt-_IDYh6S3fRioMwcq30Oo9MBjkhhxlU" width="600"> <br>
  
    [Understanding Convolutional Neural Networks (CNNs): Best For Image Classification](https://www.linkedin.com/pulse/understanding-convolutional-neural-networks-cnns-best-al-ameen/)
  
  * **ii. Basic Convolutional Network Architecture Explained**
    <br><img src="https://github.com/gasingh/furniture-style-classifier/blob/main/Schematic-diagram-of-a-basic-convolutional-neural-network-CNN-architecture-26_%5Bconvertio.co%5D.jpg" width="600"> <br>
  
    [A High-Accuracy Model Average Ensemble of Convolutional Neural Networks for Classification of Cloud Image Patches on Small Datasets](https://www.researchgate.net/publication/336805909_A_High-Accuracy_Model_Average_Ensemble_of_Convolutional_Neural_Networks_for_Classification_of_Cloud_Image_Patches_on_Small_Datasets)
  
  * **iii. Image Recognition with CNNs**
    <br><img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*S6fxo5MZG-Jy2DwAbxC8xQ.jpeg" width="600"> <br>
 
    CNNs are a type of deep learning algorithm that are particularly good at recognizing patterns and objects in images. With the right training and enough data, CNNs can even outperform humans in tasks like image recognition. So grab your favorite caffeinated beverage, put on your learning cap, and let’s dive into the world of CNNs!
    
    Convolutional neural networks, or CNNs, are a type of deep learning algorithm that is widely used in computer vision tasks. CNNs are specifically designed to process data that has a grid-like structure, such as images, and are capable of automatically learning and extracting features from this data. This allows CNNs to perform tasks such as image classification and object detection with high accuracy.
    
    Convolutional neural networks (CNNs) are commonly used for image recognition and other tasks that involve visual data. For example, you could use a CNN to classify images of different types of animals. To do this, you would first need to create a dataset of images of animals, with labels indicating the type of animal in each image (e.g. cat, dog, horse, etc.).
    
    To train the CNN, you would feed the dataset into the model and use an optimization algorithm to adjust the weights of the network so that it can accurately classify the images. Once the model is trained, you can use it to make predictions on new images. For example, you could feed an image of a cat into the model and it would predict that the image contains a cat.
    
    The key advantage of using a CNN for this task is that it can automatically learn the features that are important for distinguishing between different types of animals. This means that you don’t need to manually design features to use as inputs to the model — the CNN will learn these features automatically from the data. This makes CNNs very effective for tasks that involve visual data.
 
    The formula for a convolutional neural network (CNN) involves several components, including the input layer, convolutional layers, pooling layers, and fully connected layers.
    
    The input layer receives the raw data, such as an image or audio signal. The convolutional layers apply filters to the input data, extracting important features and reducing the dimensionality. The pooling layers then downsample the feature maps, reducing the spatial dimensions while retaining the most important information.
    
    Finally, the fully connected layers use the extracted features to make predictions or perform other tasks, such as classification or regression. The output of the CNN is determined by the specific goals of the model and the design of the network architecture.
        
    [Img Source](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*S6fxo5MZG-Jy2DwAbxC8xQ.jpeg)
    [Image recognition with CNNs: improving accuracy and efficiency | by Mitch Huang | Medium](https://medium.com/@mitchhuang777/image-recognition-with-cnns-improving-accuracy-and-efficiency-dd347b636e0c)
  
  * **iv. Deep CNN for Image Recognition**
    <br><img src="https://media.licdn.com/dms/image/C5612AQG39VNw5iWz3A/article-cover_image-shrink_720_1280/0/1529972698892?e=2147483647&v=beta&t=n4Nam1QvboO8Awhf3b_T00n5_esV7SO202fTpZ5MHEI" width="600"> <br>
  
    [Hierarchical Deep CNN for Image Recognition | LinkedIn](https://www.linkedin.com/pulse/hierarchical-deep-cnn-image-recognition-satya-vasanth-reddy-tumati/)
  
  * **v. CNN Architecture Illustrated**
    <br><img src="https://github.com/gasingh/furniture-style-classifier/blob/main/The-typical-CNN-architecture-for-image-classification-tasks-a-The-typical-CNN%5BDeep%20learning%20in%20optical%20metrology%20a%20review%5D.png" width="600"> <br>
  
    The typical CNN architecture for image-classification tasks. a The typical CNN architecture for image classification tasks consists of the input layer, convolutional layers, fully connected layers, and output prediction. b Convolution operation. c Pooling operation.

    <img src="https://github.com/gasingh/furniture-style-classifier/blob/main/Three-deep-learning-based-upsampling-methods-typically-used-in-CNN-a-Unpooling-b%20%5BDeep%20learning%20in%20optical%20metrology%20a%20review%5D.png" width="600"> <br>
    
    Three deep-learning-based upsampling methods typically used in CNN. a Unpooling. b Transposed convolution. c Sub pixel convolution.

    [Img Source 1](https://www.researchgate.net/figure/The-typical-CNN-architecture-for-image-classification-tasks-a-The-typical-CNN_fig2_358833373)
,[Img Source 2](https://www.researchgate.net/figure/Three-deep-learning-based-upsampling-methods-typically-used-in-CNN-a-Unpooling-b_fig4_358833373)
  
    [(PDF) Deep learning in optical metrology: a review](https://www.researchgate.net/publication/358833373_Deep_learning_in_optical_metrology_a_review)

</details> 

<details> <summary> Literature Review 02: CNN Classification & Logic </summary>  

  <br> **Google Search 02** : _cnn image classification logic explained_
  <br>[cnn image classification logic explained - Google Search](https://www.google.com/search?sca_esv=0ddce2aac02c1925&rlz=1C1GCEB_enGB1045GB1045&sxsrf=ADLYWIL7FDPMsDphMXEHum_LcnsV1SZhfQ:1728830101591&q=cnn+image+classification+logic+explained&udm=2&fbs=AEQNm0Aa4sjWe7Rqy32pFwRj0UkWfbQph1uib-VfD_izZO2Y5sC3UdQE5x8XNnxUO1qJLaQWdrk7tnb4cmEQUUbePQeEPt1o3RbA2PBoOOMJ3T4YmNHjUWP9fTrmSj74dymHzutB84oF5TSmO6C32fnEW8r36y461mhVKj_KdcRWfsjRkNsZzVwY13qpaW5CEVFIaryiYRoM&sa=X&ved=2ahUKEwjcg9SAyouJAxVQUUEAHcgGKI4QtKgLegQIEBAB&biw=1383&bih=747&dpr=2.5) <br>

  * **i. Review of deep learning: concepts, CNN architectures, challenges, applications, future directions | SpringerOpen 2021**
 
    In the last few years, the deep learning (DL) computing paradigm has been deemed the Gold Standard in the machine learning (ML) community. Moreover, it has gradually become the most widely used computational approach in the field of ML, thus achieving outstanding results on several complex cognitive tasks, matching or even beating those provided by human performance. One of the benefits of DL is the ability to learn massive amounts of data. The DL field has grown fast in the last few years and it has been extensively used to successfully address a wide range of traditional applications. More importantly, DL has outperformed well-known ML techniques in many domains, e.g., cybersecurity, natural language processing, bioinformatics, robotics and control, and medical information processing, among many others. Despite it has been contributed several works reviewing the State-of-the-Art on DL, all of them only tackled one aspect of the DL, which leads to an overall lack of knowledge about it. Therefore, in this contribution, we propose using a more holistic approach in order to provide a more suitable starting point from which to develop a full understanding of DL. Specifically, this review attempts to provide a more comprehensive survey of the most important aspects of DL and including those enhancements recently added to the field. In particular, this paper outlines the importance of DL, presents the types of DL techniques and networks. It then presents convolutional neural networks (CNNs) which the most utilized DL network type and describes the development of CNNs architectures together with their main features, e.g., starting with the AlexNet network and closing with the High-Resolution network (HR.Net). Finally, we further present the challenges and suggested solutions to help researchers understand the existing research gaps. It is followed by a list of the major DL applications. Computational tools including FPGA, GPU, and CPU are summarized along with a description of their influence on DL. The paper ends with the evolution matrix, benchmark datasets, and summary and conclusion.

    <br><img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs40537-021-00444-8/MediaObjects/40537_2021_444_Fig3_HTML.png" width="600" > <br>    
    The difference between deep learning and traditional machine learning

    Achieving the classification task using conventional ML techniques requires several sequential steps, specifically pre-processing, feature extraction, wise feature selection, learning, and classification. Furthermore, feature selection has a great impact on the performance of ML techniques. Biased feature selection may lead to incorrect discrimination between classes. Conversely, DL has the ability to automate the learning of feature sets for several tasks, unlike conventional ML methods [18, 26]. DL enables learning and classification to be achieved in a single shot (Fig. 3: Figure above). 

    <br><img src= "https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs40537-021-00444-8/MediaObjects/40537_2021_444_Fig4_HTML.png?as=webp" width="600"> <br>
    Deep learning performance compared to human

     DL has become an incredibly popular type of ML algorithm in recent years due to the huge growth and evolution of the field of big data [27, 28]. It is still in continuous development regarding novel performance for several ML tasks [22, 29,30,31] and has simplified the improvement of many learning fields [32, 33], such as image super-resolution [34], object detection [35, 36], and image recognition [30, 37]. Recently, DL performance has come to exceed human performance on tasks such as image classification (Fig. 4: Figure above).

    <br><img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs40537-021-00444-8/MediaObjects/40537_2021_444_Fig7_HTML.png" width="600"> <br>
    An example of CNN architecture for image classification

    In the field of DL, the CNN is the most famous and commonly employed algorithm [30, 71,72,73,74,75]. The main benefit of CNN compared to its predecessors is that it automatically identifies the relevant features without any human supervision [76]. CNNs have been extensively applied in a range of different fields, including computer vision [77], speech processing [78], Face Recognition [79], etc. The structure of CNNs was inspired by neurons in human and animal brains, similar to a conventional neural network. More specifically, in a cat’s brain, a complex sequence of cells forms the visual cortex; this sequence is simulated by the CNN [80]. Goodfellow et al. [28] identified three key benefits of the CNN: equivalent representations, sparse interactions, and parameter sharing. Unlike conventional fully connected (FC) networks, shared weights and local connections in the CNN are employed to make full use of 2D input-data structures like image signals. This operation utilizes an extremely small number of parameters, which both simplifies the training process and speeds up the network. This is the same as in the visual cortex cells. Notably, only small regions of a scene are sensed by these cells rather than the whole scene (i.e., these cells spatially extract the local correlation available in the input, like local filters over the input).
    
    A commonly used type of CNN, which is similar to the multi-layer perceptron (MLP), consists of numerous convolution layers preceding sub-sampling (pooling) layers, while the ending layers are FC layers. An example of CNN architecture for image classification is illustrated in Fig. 7.
    
    [Review of deep learning: concepts, CNN architectures, challenges, applications, future directions | Journal of Big Data | Full Text](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00444-8)
    
  * **ii. Math behind CNNs**
    <br><img src= "https://svitla.com/wp-content/webp-express/webp-images/uploads/2024/07/36717-undestanding_cnn_for_image_processing_expert_voices-1872x1054.jpg.webp" width = "500">
    <br><img src= "https://svitla.com/wp-content/uploads/2024/07/36718-cnn_for_image_processing_part_2.jpg" width = "500"><br>
    [Understanding CNN for Image Processing | Svitla Systems](https://svitla.com/blog/cnn-for-image-processing/) <br>
    [Math Behind CNNs for Image Processing | Svitla Systems](https://svitla.com/blog/math-at-the-heart-of-cnn/) <br>
  
  * **iii. Face Recognition using CNNs**
    <br><img src= "https://i.ytimg.com/vi/rv_RctF10DY/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLB6mX4cFV8qk1Y6xDl2KRiHDq2yVg" width = "500"> <br>
    [Face Recognition using CNN (GoogleNet) - YouTube](https://www.youtube.com/watch?app=desktop&v=rv_RctF10DY)

</details>

<details> <summary> Literature Review 03: CNN Code Samples and Mathematics Explained </summary>  

  <br> **Google Search 03** : _cnn image classification logic explained_

  * **i. Image Similarity using Cosine Distance** 
    <br><img src= "https://miro.medium.com/v2/resize:fit:1100/format:webp/0*sAw6h-uU0o9xsG8_.png" width = "200"> <br>
    [Image Similarity using CNN feature embeddings | by fareid | Medium](https://medium.com/@f.a.reid/image-similarity-using-feature-embeddings-357dc01514f8) <br>
    Code Reference for an Image2Vec Class:<br>
    [ImageSimilarity/src/ImgSim/image_similarity.py at main · totogot/ImageSimilarity](https://github.com/totogot/ImageSimilarity/blob/main/src/ImgSim/image_similarity.py)
  
  * **ii. MNIST DataSet identification using a CNN**
    <br> <img src= "https://cdn.analyticsvidhya.com/wp-content/uploads/2024/06/image-29.png" width = "500"><br> 
    [Image Classification Using CNN](https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/)

  * **iii. Introduction to Deep Learning | Mathworks UK** 
    <br> <img src="https://www.mathworks.com/videos/introduction-to-deep-learning-what-are-convolutional-neural-networks--1489512765771/_jcr_content/thumbnail.adapt.1200.medium.jpg/1692945517181.jpg" width = "500"> <br>
    [What Are Convolutional Neural Networks? | Introduction to Deep Learning - MATLAB](https://uk.mathworks.com/videos/introduction-to-deep-learning-what-are-convolutional-neural-networks--1489512765771.html?gclid=CjwKCAjw9p24BhB_EiwA8ID5BgM-Z1P_irHJVX-sgf7yURAMJa6FLDN24UporL4_hAPeeRACNPGdZBoCvqMQAvD_BwE&ef_id=CjwKCAjw9p24BhB_EiwA8ID5BgM-Z1P_irHJVX-sgf7yURAMJa6FLDN24UporL4_hAPeeRACNPGdZBoCvqMQAvD_BwE:G:s&s_kwcid=AL!8664!3!604136203113!b!!g!!%2Bconvolutional%20%2Bneural%20%2Bnetworks&s_eid=psn_52858618944&q=+convolutional++neural++networks&gad_source=1)
  
  * **iv. What is a Convolutional Neural Network? | Mathworks UK**
    <br><img src = "https://uk.mathworks.com/discovery/convolutional-neural-network/_jcr_content/mainParsys/band_copy_copy/mainParsys/lockedsubnav/mainParsys/columns/a32c7d5d-8012-4de1-bc76-8bd092f97db8/image_2109075398_cop.adapt.full.medium.jpg/1726854213002.jpg" width = "800" > <br>
    Example of a network with many convolutional layers. Filters are applied to each training image at different resolutions, and the output of each convolved image is used as the input to the next layer. <br>
    [What Is a Convolutional Neural Network? | 3 things you need to know - MATLAB & Simulink](https://uk.mathworks.com/discovery/convolutional-neural-network.html)
  
  * **v. Understanding of the CNN Matrix Math | Miro: Raghav Prabhu**
    <br><img src= "https://miro.medium.com/v2/resize:fit:1100/format:webp/1*4yv0yIH0nVhSOv3AkLUIiw.png" width = "300"> <br>
    Consider a 5 x 5 whose image pixel values are 0, 1 and filter matrix 3 x 3
    <br><img src= "https://miro.medium.com/v2/resize:fit:640/format:webp/1*MrGSULUtkXc0Ou07QouV8A.gif" width = "300"> <br>
    Then the convolution of 5 x 5 image matrix multiplies with 3 x 3 filter matrix which is called “Feature Map” as output. Annotated above as the Convolved Feature.
    <br><img src= "https://miro.medium.com/v2/resize:fit:828/format:webp/1*uJpkfkm2Lr72mJtRaqoKZg.png" width = "300"> <br>
    Convolution of an image with different filters can perform operations such as edge detection, blur and sharpen by applying filters. The below example shows various convolution image after applying different types of filters (Kernels). <br>
    <br> <img src = "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*4GLv7_4BbKXnpc6BRb0Aew.png" width = "500" ><br>
    Summary
        - Provide input image into convolution layer
        - Choose parameters, apply filters with strides, padding if requires. Perform convolution on the image and apply ReLU activation to the matrix.
        - Perform pooling to reduce dimensionality size
        - Add as many convolutional layers until satisfied
        - Flatten the output and feed into a fully connected layer (FC Layer)
        - Output the class using an activation function (Logistic Regression with cost functions) and classifies images.
    <br>
    [Understanding of Convolutional Neural Network (CNN) — Deep Learning | by Prabhu Raghav | Medium](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148) <br>

  * **vi. CNN Architectures | Miro: Raghav Prabhu**
    <br><img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*HyaPKtxU07iVzZ4RjJJlUQ.png" width = "500"> <br>
    [CNN Architectures — LeNet, AlexNet, VGG, GoogLeNet and ResNet | by Prabhu Raghav | Medium](https://medium.com/@RaghavPrabhu/cnn-architectures-lenet-alexnet-vgg-googlenet-and-resnet-7c81c017b848) <br>
</details>

<details> <summary> Literature Review 04.0: CNN Details and various mathematical data extraction layers of the algorithm </summary>  

<br> **Google Search 04** : _pooling to reduce dimensions in CNN_
  <br>[pooling to reduce dimensions in CNN - Google Search](https://www.google.com/search?sca_esv=778fd8ccaca240ed&rlz=1C1GCEB_enGB1045GB1045&sxsrf=ADLYWILPtdUmBQRzcTnLjhp_uTU-VzsQLg:1728858943390&q=pooling+to+reduce+dimensions+in+CNN&udm=2&fbs=AEQNm0Aa4sjWe7Rqy32pFwRj0UkWd8nbOJfsBGGB5IQQO6L3J_86uWOeqwdnV0yaSF-x2jon2iao6KWCaVjfn7ahz_sfz4kQc-hbvsXJ2gNx0RnV2nl305mvoek0YK94ylYY2a4b3Q-OEwW5lKppi2wujywZWmdIJVp8wrsv_g-eh5sWEDXx8JNpgmjsiKj2mZMvftPlZJZz&sa=X&ved=2ahUKEwik5L-5tYyJAxWq8LsIHd6UFVkQtKgLegQIFBAB&biw=1383&bih=747&dpr=2.5#vhid=BGrXRMr3gYeZXM&vssid=mosaic) <br>
  
  * **i. Convolutional Neural Network: A Complete Guide | OpenCV | PyTorch** 
    <br><img src= "https://learnopencv.com/wp-content/uploads/2023/01/Convolutional-Neural-Networks.png" width = "800" > <br>
    <br><img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/Capture_OpenCV_PyTorch%20CNN%20Tutorial.JPG" width = "800" > <br>
    <br>
    OpenCV University has CNN tutorials in PyTorch. This is great!
    <br><br>
    Convolutional Neural Network (CNN) forms the basis of computer vision and image processing. In this post, we will learn about Convolutional Neural Networks in the context of an image classification problem. We first cover the basic structure of CNNs and then go into the detailed operations of the various layer types commonly used. The above diagram shows the network architecture of a well-known CNN called VGG-16 for illustration purposes. It also shows the general structure of a CNN, which typically includes a series of convolutional blocks followed by a number of fully connected layers.
    <br><br>
    The convolutional blocks extract meaningful features from the input image, passing through the fully connected layers for the classification task. <br> <br>
    [Convolutional Neural Network: A Complete Guide](https://learnopencv.com/understanding-convolutional-neural-networks-cnn/)
    <br>    
  * <mark>**ii. Guide to CNNs | Saturn Cloud**</mark>
    <br><img src= "https://saturncloud.io/images/blog/a-cnn-sequence-to-classify-handwritten-digits.webp" width = "500"> <br>
    A CNN sequence to classify handwritten digits <br> <br>
    A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm that can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image, and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.
    <br><br>
    The architecture of a ConvNet is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. A collection of such fields overlap to cover the entire visual area.
    <br><br>
    A ConvNet is able to successfully capture the Spatial and Temporal dependencies in an image through the application of relevant filters. The architecture performs a better fitting to the image dataset due to the reduction in the number of parameters involved and the reusability of weights. In other words, the network can be trained to understand the sophistication of the image better. <br> <br>
    <br> <img src = "https://saturncloud.io/images/blog/types-of-pooling.webp" width = "500"> <br>
    [A Guide to Convolutional Neural Networks — the ELI5 way | Saturn Cloud Blog](https://saturncloud.io/blog/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way/)
    
  * iii. <mark>**A Comprehensive Guide: What are Convolutional Neural Networks | BasicAI's Blog** </mark>
    <br><img src= "https://static.wixstatic.com/media/4c4fd6_ca524a49ba784a208a4326f20eadb73d~mv2.jpg/v1/fill/w_1600,h_900,al_c,q_85/4c4fd6_ca524a49ba784a208a4326f20eadb73d~mv2.jpg" width = "500"> <br>
    While browsing through images on social media or using facial recognition to unlock your smartphone, have you ever wondered what technology makes these seemingly simple everyday actions possible? Behind all this is the powerful technology of Convolutional Neural Networks (CNNs). CNNs are not only the cornerstone of modern computer vision but also a key driver in advancing artificial intelligence. <br>
    <br>Applications of CNNs: <br><br>
    Image Recognition and Classification: This is the most traditional and widespread application of CNNs. They have shown exceptional performance in tasks like recognizing faces on social media and identifying abnormalities in medical imaging. For instance, Google Photos uses CNNs for image content recognition and classification, distinguishing thousands of objects and scenes with over 90% accuracy. In healthcare, CNNs are used for assisting diagnoses, such as identifying skin cancer, matching the accuracy of professional dermatologists.
    <br><br>
    Object Detection and Segmentation: CNNs can not only recognize objects within images but also determine their location and size (object detection) and even precisely segment each object within an image (image segmentation). In autonomous vehicle technology, such as Tesla's Autopilot system, CNNs are used for real-time detection of vehicles, pedestrians, and various obstacles on the road, with an accuracy rate exceeding 95%. In retail, CNNs enhance efficiency and accuracy in object recognition and inventory management. <br> <br>
    [A Comprehensive Guide: What are Convolutional Neural Networks | BasicAI's Blog](https://www.basic.ai/blog-post/a-comprehensive-guide-to-convolutional-neural-networks)
    <br>
  * **iv. CNN Guide | Microsoft | PyTorch | CIFAR10 dataset** <br>
    ![img](https://learn.microsoft.com/en-us/training/achievements/pytorch-computer-vision.svg)
    <br>Here, you'll build a basic convolution neural network (CNN) to classify the images from the CIFAR10 dataset.
    <br>
    [Introduction to Computer Vision with PyTorch - Training | Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/intro-computer-vision-pytorch/)
    [Use PyTorch to train your image classification model | Microsoft Learn](https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model)

  



  * vii.
    

[CNN in combination with KNN - Deep Learning - fast.ai Course Forums | Specific Algorithm tweak for Furniture Classification Tasks](https://forums.fast.ai/t/cnn-in-combination-with-knn/4280)

</details>

<details> <summary> <mark> Literature Review 04.1_EXTRAs: CNN Explainer & 2d Convolutions </mark> </summary>

_**Super Useful Links from OpenCV CNN Webpage!!**_
  
  * <mark>**CNN Explainer** </mark>
    <br><img src = "https://poloclub.github.io/cnn-explainer/assets/figures/preview.png" width = "500"><br>
    [**CNN Explainer**](https://poloclub.github.io/cnn-explainer/)
    [**Demo Video "CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization" - YouTube**](https://www.youtube.com/watch?v=HnWIHWFbuUQ)
    <br>
  * **2D Convolution Explained**
    <br> <img src = "https://i.ytimg.com/vi/yb2tPt0QVPY/maxresdefault.jpg" width = "500"> <br>
    [2D Convolution Explained: Fundamental Operation in Computer Vision - YouTube](https://www.youtube.com/watch?v=yb2tPt0QVPY) <br>
  * <mark>**6 basic things to know about Convolution | 2D Convolution explained by applying Convolution Matrices in Numpy ** </mark> 
    <br> <img src = "https://miro.medium.com/v2/resize:fit:928/0*e-SMFTzO8r7skkpc" width = "500"> <br>
    Convolution Operation on a 7x7 matrix with a 3x3 kernel <br>
      * 1. In mathematics, convolution is an operation performed on two functions (f and g) to produce a third function. Convolution is one of the most important operations in signal and image processing. It could operate in 1D (e.g. speech processing), 2D (e.g. image processing) or 3D (video processing). <br>
      * 2. In image processing, convolution is the process of transforming an image by applying a kernel over each pixel and its local neighbors across the entire image. The kernel is a matrix of values whose size and values determine the transformation effect of the convolution process. <br>
      * 3. The Convolution Process involves these steps. (1)It places the Kernel Matrix over each pixel of the image (ensuring that the full Kernel is within the image), multiplies each value of the Kernel with the corresponding pixel it is over. (2)Then, sums the resulting multiplied values and returns the resulting value as the new value of the center pixel. (3) This process is repeated across the entire image. <br>
      * 4. As we see in the picture, a 3x3 kernel is convoluted over a 7x7 source image. Center Element of the kernel is placed over the source pixel. The source pixel is then replaced with a weighted sum of itself and surrounding pixels. The output is placed in the destination pixel value. In this example, at the first position, we have 0 in source pixel and 4 in the kernel. 4x0 is 0, then moving to the next pixel we have 0 and 0 in both places. 0x0 is 0. Then again 0x0 is 0. Next at the center there is 1 in the source image and 0 in the corresponding position of kernel. 0x1 is 0. Then again 0x1 is 0. Then 0x0 is 0 and 0x1 is 0 and at the last position it is -4x2 which is -8. Now summing up all these results we get -8 as the answer so the output of this convolution operation is -8. This result is updated in the Destination image. <br>
      * 5. The output of the convolution process changes with the changing kernel values.
          * IDENTITY KERNEL: For example, an Identity Kernel shown below, when applied to an image through convolution, will have no effect on the resulting image. Every pixel will retain its original value as shown in the following figure. <br>
          <img src = "https://miro.medium.com/v2/resize:fit:186/format:webp/0*r5ARjKpVERojnPFu" width = "100"><br>
          Identity Kernel <br>
          <img src = "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ukrsCZSyKUYsX9hR2ItJog.png" width = "300" > <br>
          Original Image(Left) and Image after applying Identity Filter of size 3x3(Right) <br>
          * SHARPEN KERNEL: A Sharpen Kernel like this when applied to an image through convolution, will have an image sharpening effect to the resulting image. The precise values can be customized for varying levels of sharpness as shown in the following figure. <br>
          <br><img src = "https://miro.medium.com/v2/resize:fit:1100/format:webp/1*DZyIk0Gx2K174hkZym0mmg.png" width = "300"> <br>
          Sharpen Kernel <br>
          <img src = "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*VWCiLmwKi-EEeUYQ-tA7gQ.png" width = "500"> <br>
          Original Image(Left) and Image after applying Sharpen Filter of size 3x3 (Right) <br>
          * GAUSSIAN BLUR KERNEL: The Gaussian Blur Kernel like this when applied to an image through convolution, will apply a Gaussian Blurring effect to the resulting image.
          <br><img src = "https://miro.medium.com/v2/resize:fit:640/format:webp/0*ovFlOqxpw8NVOPEC" width = "300"> <br>
          <br><img src = "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*eOlu2cCsALcbWn0tqANDaA.png" width = "500"> <br>
          Just as how the values of the Kernel can be varied for different levels of effects, the size of the Kernel can also be altered to shape the effect of the convolution.By increasing the size of the Kernel Matrix, the spatial locality influencing each pixel’s resulting value is increased as pixels from further away are being pulled into the equation. There are many more Kernels that are used in image processing such as edge detection, embossing, rotation, etc. <br>
      * 6. Convolution is the key concept in Convolutional Neural Networks. Convolutional Neural Networks (CNN) are a type of Deep Neural Network. A CNN comprises of Convolutional Layer, Pooling Layer, and Fully-Connected Layer. At the Convolution layer, a CNN applies convolution on to its inputs using a Kernel Matrix that it calibrates through training. For this reason, CNNs are very good at feature matching in images and object classification. The convolution layer parameters consist of a set of learnable kernels. Every kernel is small matrix that extends through the full depth of the input volume. During the forward pass, we convolve each kernel across the width and height of the input image and compute dot products between the pixel values of the source and kernel at corresponding positions.
      <br>
    [6 basic things to know about Convolution | by Madhushree Basavarajaiah | Medium](https://medium.com/@bdhuma/6-basic-things-to-know-about-convolution-daef5e1bc411) <br>
    Numpy Code <br>
    [Convolution](https://gist.github.com/MadhushreeB/620a27aceb3088885546bc5b5a88245f#file-convolution-py) <br>
  * <mark>**What Is a Convolution? How To Teach Machines To See Images** </mark>
    <br> <img src= "https://assets.8thlight.com/images/insights/posts/2022-03-25-what-is-a-convolution/padding_example.png" width = "500" > <br>
    [Image Classification: An Introduction to Artificial Intelligence | 8th Light](https://8thlight.com/insights/image-classification-an-introduction-to-artificial-intelligence)
    [What Is a Convolution? How To Teach Machines To See Images | 8th Light](https://8thlight.com/insights/what-is-a-convolution-how-to-teach-machines-to-see-images)
  * **Above 4 links based on a subsequent Google Search** : _2D Convolution Explained: Fundamental Operation in Computer Vision_
    [2D Convolution Explained: Fundamental Operation in Computer Vision - Google Search](https://www.google.com/search?sca_esv=778fd8ccaca240ed&rlz=1C1GCEB_enGB1045GB1045&sxsrf=ADLYWIJxNef7cUzatRZ3gqBHF2ASFG3p6w:1728862557474&q=2D+Convolution+Explained:+Fundamental+Operation+in+Computer+Vision&udm=2&fbs=AEQNm0Aa4sjWe7Rqy32pFwRj0UkWfbQph1uib-VfD_izZO2Y5sC3UdQE5x8XNnxUO1qJLaTV4c8WUZIVpqotNJgZT-nG6tsHwc3BxYJ7SJsoRXGGNwEIXpA7OZG7ZaHhjSESTEISdWI9c-9iMZIuDuLLcLeXPAcrvS050xiLHdT_XEfYYSw3K4Imoc20bsa6YY0FtgSBNRMi&sa=X&ved=2ahUKEwjg9-n0woyJAxVZnf0HHWJiAGIQtKgLegQIDRAB&biw=1383&bih=747&dpr=2.5#vhid=cr7CSy_DCtOU5M&vssid=mosaic)
  * 01:02 14/10/2024
</details>

<details> <summary> <mark>  Literature Review 04.2_EXTRAs: Google Photos using CNNs & the ImageNet Competition and Paper by Google Research | 2012 </mark> </summary>

_**Excellent Articles**_

* **A Comprehensive Guide: What are Convolutional Neural Networks | BasicAI's Blog**

    <br><img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/Capture_BasicAI_Google%20uses%20CNNs%20for%20Image%20Recognition.JPG" width = "500">
    
    [A Comprehensive Guide: What are Convolutional Neural Networks | BasicAI's Blog](https://www.basic.ai/blog-post/a-comprehensive-guide-to-convolutional-neural-networks)

* **Revolutionizing Vision: The Rise and Impact of Image Recognition Technology | BasicAI's Blog**

    <br><img src = "https://static.wixstatic.com/media/4c4fd6_997a209ead334aa7a5b019564856b2d2~mv2.jpg/v1/fill/w_1120,h_630,al_c,q_90,enc_auto/4c4fd6_997a209ead334aa7a5b019564856b2d2~mv2.jpg" width = "500">
    <br><img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/Capture_BasicAI_Image%20Recognition.JPG" width = "500"> <br>
    
    [Revolutionizing Vision: The Rise and Impact of Image Recognition Technology | BasicAI's Blog](https://www.basic.ai/blog-post/image-recognition)

* **CNN Algorithm used in Google Photos | Reference Google Research Blog & ImageNet Paper at University of Toronto**

    <br><img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/Capture_BasicAI_Google%20uses%20CNNs%20for%20Image%20Recognition_02_pre.JPG" width = "500">
    <br><img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/Capture_BasicAI_Google%20uses%20CNNs%20for%20Image%20Recognition_02.JPG" width = "500">
    
    [Which algorithm is used in Google Photos app for classification/labeling? - Quora](https://www.quora.com/Which-algorithm-is-used-in-Google-Photos-app-for-classification-labeling#:~:text=Google%20photos%20makes%20use%20of,number%20of%20classes%20and%20training.) <br>
    [How is Machine Learning applied to Google Photos? - Quora](https://www.quora.com/How-is-Machine-Learning-applied-to-Google-Photos)
  
* **Google Photos Makes Use of CNNs** <br>

    Google photos makes use of an convolutional neural network architecture similar to that used by Geoffrey Hinton's team in the ImageNet Large Scale Visual Recognition competition. The difference is only in terms of number of classes and training.
    <br> <br>
    In contrast to the 1000 visual classes made use in the competition,Google made use of 2000 visual classes based on the popular labels on google+ photos. They used labels which seemed to have visual effect on humans i.e photos which humans could recognize visually.
    <br> <br>
    They make use of FreebaseEntities which is the basis for knowledge graph in google search. These entities are used to identify elements in language independent way.In English when we encounter the word “jaguar”, it is hard to determine if it represents the animal or the car manufacturer. Entities assign a unique ID to each, removing that ambiguity.
    Google used more images to train than that used in the ImageNet competition. They also refined the classes from 2000 to 1100 to improve precision.
    Using this approach Google photos achieved double the precision that is normally achieved using other methods.
    <br> <br>
    Link to the architecture used by Geoffrey Hinton's team in ImageNet competition: Page on toronto.edu
    <br> <br>
    Source: Research Blog <br>
    <br><img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/Capture_BasicAI_Google%20uses%20CNNs%20for%20Image%20Recognition_03_IMAGENET.JPG" width = "500">
    <br><img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/Capture_BasicAI_Google%20uses%20CNNs%20for%20Image%20Recognition_04_IMAGENET.JPG" width = "500">
    <br><img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/Capture_BasicAI_Google%20uses%20CNNs%20for%20Image%20Recognition_05_IMAGENET-PAPER.JPG" width = "500"> <br>
    [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/) <br>
    [imagenet.pdf](https://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) <br>
      
</details>


<details> <summary> <mark>  Literature Review 04.3_EXTRAs: An Excellent Paper on understanding the mathematics of a CNN | National Key Lab of Novel Software Technology, China | 2017 </mark> </summary>

  <br><img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/introToCNN_paper_00.JPG" width = "500"> 
  <br><img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/introToCNN_paper_01.JPG" width = "500"> <br>
  * [CNN.pdf](https://jasoncantarella.com/downloads/CNN.pdf)

</details>


<details> <summary> <mark>  Literature Review 05: HISTORY MODULE: AlexNet and ImageNet: The Birth of Deep Learning | 2006 </mark> </summary>

* **ImageNet (2009) and AlexNet (2012) | Visio.AI**
  <br><img src = "https://viso.ai/wp-content/uploads/2024/04/ImageNet.jpg" width = "500"> <br>
  [Img Source: t-SNE visualization of CNN codes](https://cs.stanford.edu/people/karpathy/cnnembed/)<br>
    [alexnet - Google Search](https://www.google.com/search?q=alexnet&rlz=1C1GCEB_enGB1045GB1045&oq=alexnet&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRg9MgYIAhBFGDzSAQgyODQzajBqN6gCALACAA&sourceid=chrome&ie=UTF-8) <br>
    [AlexNet: A Revolutionary Deep Learning Architecture - viso.ai](https://viso.ai/deep-learning/alexnet/#:~:text=AlexNet%20is%20an%20Image%20Classification,from%2026.2%25%20to%2015.3%25.) <br>

* **ImageNet 2009 Explained | DeepAI**
  <br><img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/imagenet_info1_.JPG" width = "500">
  <br><img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/imagenet_info2.JPG" width = "500"> <br>
  [ImageNet Definition | DeepAI](https://deepai.org/machine-learning-glossary-and-terms/imagenet#:~:text=ImageNet%20was%20created%20by%20Dr,way%20that%20machines%20could%20understand.) <br>
  [Convolutional Neural Network Definition | DeepAI](https://deepai.org/machine-learning-glossary-and-terms/convolutional-neural-network) <br>
  
  <br> <img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/ALEXNET_01.JPG" width = "500">
  <br> <img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/ALEXNET_00.JPG" width = "500">
  <br> <img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/alexnet_02.JPG" width = "500">
  <br> <img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/alexnet_03.JPG" width = "500">
  <br>
  [alexnet - Google Search](https://www.google.com/search?q=alexnet&rlz=1C1GCEB_enGB1045GB1045&oq=alexnet&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRg9MgYIAhBFGDzSAQgyODQzajBqN6gCALACAA&sourceid=chrome&ie=UTF-8) <br>
  <br> <img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/alexnet_05_article_deepAI.JPG" width = "500">
  <br> <img src= "https://viso.ai/wp-content/uploads/2024/04/alexNet.jpg" width = "500">
  <mark>[AlexNet: A Revolutionary Deep Learning Architecture - viso.ai](https://viso.ai/deep-learning/alexnet/#:~:text=AlexNet%20is%20an%20Image%20Classification,from%2026.2%25%20to%2015.3%25.) </mark> <br>

* **AlexNet 2012 Explained | DeepAI**
  <br><img src = "https://viso.ai/wp-content/uploads/2024/01/Convolutional-Neural-Network-for-Detection-of-Cargo-Ships.jpg" width = "500"> <br>
  <br><img src = "https://github.com/gasingh/furniture-style-classifier/blob/main/cnn%20and%20gnns.JPG" width = "500"> <br>
  <mark>[Convolutional Neural Networks (CNNs): A 2024 Deep Dive - viso.ai](https://viso.ai/deep-learning/convolutional-neural-networks/) </mark> <br>
  <br>
  <br><img src = "https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-22_at_6.35.45_PM.png" width = "500"> <br>
  [AlexNet Explained | Papers With Code](https://paperswithcode.com/method/alexnet#:~:text=AlexNet%20is%20a%20classic%20convolutional,the%20model%20across%20two%20GPUs.) <br>
  
  <br><img src = "https://github.com/gasingh/furniture-style-classifier/blob/main/alexnet_04_writing%20from%20scratch%20in%20pytorch.JPG" width = "500"> <br>
  [Writing AlexNet from Scratch in PyTorch | DigitalOcean](https://www.digitalocean.com/community/tutorials/alexnet-pytorch)<br>


</details>


## Data sources: AI training datasets (digital)

The Idea is to be able to build an AI Model which can automatically learn to recognize image styles from data/ images available freely on the the internet from the opensource domain, freely available image datasets to train an AI Model.

I found a few datasources online:

<details> <summary> Literature Review 05: AI Training Datasets Online </summary>


</details>


## Data sources: Stylization references (analog)

A few interesting webpages which talk about and illustrate various futniture styles.

<details> <summary> Literature Review 06: Stylization References </summary>

* 1. Classifying Wardrobes
    <br><img src="https://i.pinimg.com/736x/4b/b6/42/4bb6427371d007215fd60268b71972a0.jpg" height="300"> <br>
    [source](https://in.pinterest.com/pin/28288303900156751/)
* 2. Chair Styles
     <br><img src="https://cdn.shopify.com/s/files/1/0845/8923/files/furniture-design-timeline_2048x2048.jpg?2354855286364133834" width = "500"> <br>
    * [How To Identify Antique Furniture | Buyer's Guide | Styylish](https://styylish.com/how-to-identify-antique-furniture/)
* 3. Famous Designers
     ![furniture-design-timeline_2048x2048.jpg (1814×1148)](https://cdn.shopify.com/s/files/1/0845/8923/files/furniture-design-timeline_2048x2048.jpg?2354855286364133834)
    * [FURNITURE DESIGN HISTORY | ebarza Modern Furniture in Abu Dhabi & Dubai](https://www.ebarza.com/pages/famous-designers)
    
* 4. An interesting book looking at visual aspects of identifying furniture from 1985.
   <img src= "https://m.media-amazon.com/images/I/91NIvFMx4BL._AC_UF894,1000_QL80_.jpg" width= "500"> <br>
   <!-- <img src= "https://m.media-amazon.com/images/I/81Xutlgn5VL._SL1500_.jpg" width= "500"> <br> -->
   * [American Antique Furniture Styles (A Roundtable Press book): Amazon.co.uk: Butler, Joseph T., Johnson, Kathleen Eagen: 9780816010080: Books](https://www.amazon.co.uk/American-Antique-Furniture-Styles-Roundtable/dp/0816010080)
   * [Field Guide to American Antique Furniture: A Unique Visual System for Identifying the Style of Virtually Any Piece of American Antique Furniture: Amazon.co.uk: Butler, Joseph T., Skibinski, Ray: Books](https://www.amazon.co.uk/Field-Guide-American-Antique-Furniture/dp/B0013TMMZI)
 
</details>




## AI & Ethics

The ethical concerns around use of online images for training AI Models. Since the original inspiration for my project is a book, I also investigated the various ethical and legal implications which arise, and should be understood, if one were to simply digitize a book for building an AI application. Although this is not what we are doing for this project, I nevertheless delved deep into this aspect, as it is something very contemporary and definitely related to a project which involves training an AI model on publically available images online. 

In my research online, I found that it might be fine to takeup such a pursuit for purely educational, research and non-commercial purposes: under **fair use**. Given that the book is already accessible the public domain on the Internet Archive.

<details> <summary> Literature Review 07: AI and Ethics around use of images for training AI models </summary>
  
  * SET 1
    * [Identifying American furniture : a pictorial guide to styles and terms, Colonial to contemporary : Naeve, Milo M : Free Download, Borrow, and Streaming : Internet Archive](https://archive.org/details/identifyingameri00milo/page/96/mode/2up)
    * [Identifying American furniture : a pictorial guide to styles and terms, Colonial to contemporary | WorldCat.org](https://search.worldcat.org/title/39883858)
    * [Milo Naeve - Wikipedia](https://en.wikipedia.org/wiki/Milo_Naeve)
    * [Identifying American furniture by Milo M. Naeve | Open Library](https://openlibrary.org/books/OL17574209M/Identifying_American_furniture)
    * [Identifying American Furniture – A Pictorial Guide to Styles & Terms Colonial to Contemporary 3e Rev: A Pictorial Guide to Styles and Terms Colonial ... State and Local History Books (Paperback)): Amazon.co.uk: Naeve, Milo M: 9780393318449: Books](https://www.amazon.co.uk/Identifying-American-Furniture-Contemporary-Association/dp/0393318443)
  
  * SET 2
    * [CNN Training with 90s Book: ChatGPT](https://chatgpt.com/c/670447d1-d62c-8000-bfaf-e6ac90f6f115)
    * [check copyright of a book - Google Search](https://www.google.com/search?q=check+copyroght+of+a+book&rlz=1C1GCEB_enGB1045GB1045&oq=check+copyroght+of+a+book&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBCDQyNDFqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
    * [copyright for images to use in ai research project - Google Search](https://www.google.com/search?q=copyright+for+images+to+use+in+ai+research+project&sca_esv=37c01b8175ee359e&rlz=1C1GCEB_enGB1045GB1045&sxsrf=ADLYWIJpL1rfxHTsKqExXO3CBt_V5VvaRQ:1728334589781&ei=_UoEZ6CxL_6yi-gP3t_JuQE&start=10&sa=N&sstk=AagrsugsJOpupZqFq6k8HVPixSEVkueT5tjakOKxullKq12rGfdM6LtnYob8GnWy6sgc7sol9L17Kvpb0Qt0vKtwcfxQHz84VWjYPA&ved=2ahUKEwjgo5uKlP2IAxV-2QIHHd5vMhcQ8NMDegQIBxAW&biw=1383&bih=747&dpr=2.5)
    * [Training Generative AI Models on Copyrighted Works Is Fair Use — Association of Research Libraries](https://www.arl.org/blog/training-generative-ai-models-on-copyrighted-works-is-fair-use/)
    * [Copyright law and AI: How to make sure your images don’t infringe copyright laws - KeyShot](https://www.keyshot.com/blog/copyright-and-artificial-intelligence/)
    * [using content from the internet archive under fair use for personal research project - Google Search](https://www.google.com/search?q=using+content+from+the+internet+archive+under+fair+use+for+personal+research+project&rlz=1C1GCEB_enGB1045GB1045&oq=using+content+from+the+internet+archive+under+fair+use+for+personal+research+project&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBCTE2MzE2ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
    * [Fair Use in Action at the Internet Archive | Internet Archive Blogs](https://blog.archive.org/2024/03/01/fair-use-in-action-at-the-internet-archive/)
    * [Rights – Internet Archive Help Center](https://help.archive.org/help/rights/)
    * [Code of Best Practices in Fair Use for Scholarly Research in Communication - Center for Media and Social Impact](https://cmsimpact.org/code/code-best-practices-fair-use-scholarly-research-communication/)

 <img src= "https://github.com/gasingh/furniture-style-classifier/blob/main/Capture_241013_fairUse_GoogleSearch.JPG" width="500"><br>
 ![IMG](https://github.com/gasingh/furniture-style-classifier/blob/main/Capture_241013_fairUse_GoogleSearch_highlightGoogle.JPG)
 
   * [Internet Archive Loses Copyright Lawsuit: What to Know | TIME](https://time.com/6266147/internet-archive-copyright-infringement-books-lawsuit/)
   * [Copyright related to research - LibAnswers](https://libanswers.kcl.ac.uk/faq/226572#:~:text=%E2%80%9CFair%20dealing%E2%80%9D%20is%20a%20legal,a%20work%20may%20be%20used.)
       * ![img](https://github.com/gasingh/furniture-style-classifier/blob/main/Capture_241013_fairUse_GoogleSearch_guide.JPG)
       * [Exceptions to copyright](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/375954/Research.pdf)


</details>

However, my learning pursuit would be better served if I were to learn the fundamentals of Image recognition and then learning how to apply those learnings and knowledge and work towards building my own custom tailored AI system. Hence the research on finding appropriate opensource image training sets online. Please refer Sections: _**Data sources: AI training datasets (digital)**_.

## What next?

Can this project grow and become something even more? 
I love browsing design and furniture shops, and an AI like that could be a cool tool to have on a mobile phone. Ideally a mobile phone app!
