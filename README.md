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

<details> <summary> Literature Review Section 1 </summary>  

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

<details> <summary> Literature Review Section 2 </summary>  

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


## Data sources: AI training datasets (digital)

The Idea is to be able to build an AI Model which can automatically learn to recognize image styles from data/ images available freely on the the internet from the opensource domain, freely available image datasets to train an AI Model.

I found a few datasources online:

<details> <summary> Literature Review 03: AI Training Datasets Online </summary>


</details>


## Data sources: Stylization references (analog)

A few interesting webpages which talk about and illustrate various futniture styles.

<details> <summary> Literature Review 04: Stylization References </summary>

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

<details> <summary> Literature Review 05: AI and Ethics around use of images for training AI models </summary>
  
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
