# Intro 

Vogliamo fare neural style transfer (NST). In particolare:

- Multi-style transfer (combinazione e transizione tra più stili in una singola immagine)
- Real-time style transfer su video 
- conditional style transfer usando modelli di segmentazione (per applicare lo stile solo a regioni specifiche)

-------------------------------------



# Multi-Style Transfer (Multiple Styles in One Image)

Neural style transfer typically applies a single artistic style to a content image, but multi-style transfer aims to blend or switch between several styles within one output. Key PyTorch resources include:

- Multi-Style Generative Network (MSG-Net) – Hang Zhang et al. (2017) introduced MSG-Net, a feed-forward network that can produce multiple styles from one model in real time. Their open-source PyTorch code allows training on a dataset like MS-COCO with a folder of style images, producing a single model that can render 21 different styles (or more) by conditioning the network. The official GitHub provides pre-trained models and training scripts for custom styles.
    https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer

- Conditional Instance Normalization (CIN) – Dumoulin et al. (2017) proposed learning an embedding for artistic styles by modulating normalization layers. In practice, a single style-transfer network can be trained on multiple styles by using a distinct set of instance norm parameters for each style image ￼. This technique, implemented in PyTorch by repositories like pytorch-multiple-style-transfer, lets you interpolate between styles by blending these parameters. The result is a continuous transition between styles – the network “generalizes across a diversity of artistic styles” and even permits arbitrary combinations of learned styles.
    https://ar5iv.labs.arxiv.org/html/1610.07629
    https://github.com/kewellcjj/pytorch-multiple-style-transfer


- Style Loss Blending – An alternative approach is to combine multiple style reference images at the loss level. For example, one PyTorch script demonstrates taking several style images and summing their style loss contributions to generate an output that mixes those styles. By adjusting the relative weights of each style loss (style_weight) for each style image, one can control the influence of each style ￼. This is essentially the extended Gatys optimization method where you minimize content loss plus multiple style losses. It’s slower than feed-forward methods, but very flexible – you can create a pastiche that has, say, 70% of Van Gogh’s style and 30% of Monet’s style by weighting their losses accordingly.
https://github.com/DorsaRoh/multi-style-transfer


---------------

# Real-Time Style Transfer on Video

Applying style transfer to video (frame sequence) in real time introduces the challenge of maintaining temporal consistency (avoiding flickering) while being efficient. 
Key resources and models (PyTorch implementations) include:

- Huang et al. (CVPR 2017) – Feedforward Video Style Transfer – This work is one of the first to achieve real-time neural style transfer on videos by training a feed-forward CNN with a temporal loss. They enforce that consecutive frames output by the network are both well-stylized and consistent over time. The training uses a two-frame synergistic training with optical flow to penalize temporal differences. The published code (referenced in a PyTorch reimplementation) demonstrates how a hybrid loss (content + style + temporal consistency terms) yields smooth, flicker-free stylized video at real-time speeds, far faster than naive frame-by-frame styling which can cause jitter.
https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Real-Time_Neural_Style_CVPR_2017_paper.pdf


- ReCoNet (CVPR 2019) – Realtime Coherent Video Style Transfer Network is another approach focusing on temporal coherence. A PyTorch implementation of ReCoNet is available. It uses an encoder-decoder architecture similar to fast style transfer for images, augmented with a multi-level temporal loss that uses optical flow to align the style between adjacent frames ￼. During training, ReCoNet leverages pre-computed optical flow (from datasets like MPI-Sintel) to warp the previous frame’s features and enforce stability ￼. The result is a network that runs at 200+ FPS on modern GPUs ￼, achieving style transfer on video without noticeable flicker. The repository includes training code and pre-trained models for several artistic styles (e.g. Candy, Mosaic), demonstrating high temporal consistency.
https://github.com/safwankdb/ReCoNet-PyTorch


- LearnOpenCV – Live Style Transfer Tutorial – For a more tutorial-oriented resource, the LearnOpenCV guide (2020) shows how to train Johnson’s fast style transfer network in PyTorch and then apply it to a webcam/Zoom feed in real time ￼. While not focused on advanced temporal loss, it provides practical tips on optimizing the model for video streaming (e.g. using smaller resolution, batching on GPU) and can be a starting point for real-time applications. It uses the COCO dataset for content training and gives code to stylize video frame-by-frame at interactive rates.
https://learnopencv.com/real-time-style-transfer-in-a-zoom-meeting/

(For completeness, note that research on video style transfer is active. Recent PyTorch projects explore architectures for better temporal stability, such as ReReVST (Relaxation & Regularization for Video Style, 2020) and CCPL (Contrastive Coherence Preserving Loss, 2022), which further improve temporal consistency. These are more experimental but their code is open-source.)
https://paperswithcode.com/paper/ccpl-contrastive-coherence-preserving-loss
https://paperswithcode.com/paper/consistent-video-style-transfer-via


----------------------

# Conditional Style Transfer with Segmentation

Standard style transfer uniformly stylizes the whole image, but conditional style transfer lets us apply style selectively to certain regions or objects (e.g. stylize the background but not the person, or give each object its own style). This is achieved by integrating semantic segmentation or masking models with the style transfer process:

- Class-Based Styling (CBS) – Kurzman, Vazquez, and Laradji (ICCV Workshops 2019) developed a real-time localized style transfer pipeline that uses semantic segmentation to target specific object classes. Their PyTorch code CBStyling combines a lightweight segmentation network (DABNet on Cityscapes) with a fast style transfer model (Johnson et al.). At runtime, the segmentation provides a mask (e.g. all “car” pixels), and the style network stylizes the entire frame. By masking, only the chosen class gets the style applied, and then the stylized region is merged back with the original image. This produces outputs like a driving scene where cars and road are painted in Van Gogh style while everything else remains photorealistic. The method runs in real-time (their demo stylized Cityscapes videos at ~16 FPS). The repository includes pre-trained models and a sample script to stylize an image with a given style on a specified class.
    - https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVFAD/Kurzman_Class-Based_Styling_Real-Time_Localized_Style_Transfer_with_Semantic_Segmentation_ICCVW_2019_paper.pdf
    - https://github.com/IssamLaradji/CBStyling?tab=readme-ov-file



- Mask R-CNN or Detectron2 for Object-Level Style – An alternative approach is to use instance segmentation to get masks for individual objects, then stylize each object separately. A project by R. Rifat et al. (2020) demonstrates using Facebook’s Detectron2 (Mask R-CNN) to identify multiple objects and then applying a different style to each object in one image ￼ ￼. For example, one can choose to render the horse in a “fire” texture and the person riding it in an “ocean waves” style ￼. The pipeline applies a fast style transfer model to each masked region and composites them, addressing mask overlaps and seams. Their code (in Colab) shows how to customize style per object, giving fine-grained control that standard style transfer lacks. This idea can be extended to any segmentation mask – for instance, stylizing the background while leaving the human subject untouched, by using a people-segmentation model (like DeepLabv3 on COCO) to get a background mask.
https://github.com/RashedRifat/Multiple-Object-Style-Transfer


- SAM + Style Transfer (Interactive) – The newest segmentation models like Segment Anything (SAM) enable zero-shot masks for any user-specified region. Integrating this with style transfer opens up interactive tools. SamStyler (Psychogyios et al. 2023) and similar projects use SAM to let the user click on an object, then apply a chosen style to just that region ￼ ￼. An example implementation is Segify, a PyTorch + Streamlit app that uses SAM for segmentation and an AdaIN-based style transfer network for stylization ￼ ￼. This combination allows real-time preview: the user can select multiple regions and assign different styles to each before generating the final image. Such approaches leverage powerful segmentation to achieve precise localization—for instance, stylizing only the sky with a starry night texture, or only the foreground subject—while using arbitrary style transfer (AdaIN) under the hood for flexibility.
https://github.com/g-nitin/stylized-segmentation

