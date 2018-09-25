# Experimental Results
​    This DR-GAN implementation aims for genarating frontal-face-images from profile. The network backbone is the same with Single-Image DR-GAN. And in inference phase, I add some data augmentation in order to test how robust the model is.
## 1. Experimental Settings
**Database**  CFP dataset is a small dataset for evaluating face recognition. Due to not find other labeled dataset before, I regard it as training set and divide it into train-set and dev-set by ratio 9:1.
    CFP dataset is a real and in-the-wild setting dataset, which consists of 500 subjects each with 10 frontal and 4 profile images. Therefore, I have identity number $N^d = 450$, pose code $N^p = 2$. And I set noise dimension $N^z = 50$ in generator network.

**Implementation Details** Following network structure mentioned in paper,  I align all the face images to a unity size $100 \times 100$ . I randomly crop and sample $96 \times 96$ regions from the aligned images for data augmentation. Three channels of RGB are normalized by Gauss distribution with mean of 0.5 and standrad deviation of 0.5 respectively. Because of lack of data and simplicity, I only use 2 pose code representing frontal and profile images. My network architecture is the same as that in paper. The train batch size is set to be 4 for better generalized ablity. All the weights are initialized form a zero-centered Gauss distribution with a standard deviation of 0.02. And I use Adam optimizer with a initial learning rate of 0.0002 , momentum 0.5 and beta 0.999 for both generator and discriminator.

​    I train this model in 1800 epochs with 8 GTX 1080Ti. In initial 1000 epochs, due to *D* has strong supervisions because of the identity labels, I optimize *D* one step while 5 steps for *G*. Then, in later epochs, *D* is much close to the optimal solution, I update *G* more frequently the before with15 steps.

## 2. Test Result

+ **Multi-View Test**
  I use multi-view images of same person from 15 degree to 90 degree with a interval of 15 degree.
  (15, 30, 45, 60, 75, 90 degree)

  + better case (15-60 degree)
    In these views, different views show little influence on generated images,  there are only some outline differences. The left eye and right eye are basicly symmetrical with no defomation.
  + worse case(75, 90degree)
    In these views, generated images have some severely deformation. Location, size and shape  have been varying degrees of differences. Sometimes there are some artifacts like overlapping eyes.

  Therefore, this network architecture is more proper in views ranging from -60 to 60.

  concrete result images under the directory *multi-view*

+ **Data augmentation Test**

  I use 6 kinds of data augmentation in order to test robustness. 

  + add Gauss noise
    *operation* : add Gauss noise to three channels respectively(default mean = 0, std = 16)
    *analysis* : 

    + Heavy noise will distort images even generate other person's face (*noise_1.jpg*)
    + Slight noise show little influence on generated images (*noise_2.jpg*)
    + Sometimes slight noise can make images more robust and reduce local deformation (*noise_3.jpg, noise_4.jpg*)

  + gamma transform
    *operation* : apply gamma transform to the whole image(default parameter = 0.9)
    *analysis* : 

    + Slight gamma(parameter = 0.9) almost has no impact on images.
    + Heavy gamma will make the origin image destoryed so I don't take it into account.

  + shadow
    *operation* :convert a square region in image to 0 pixel value to imitate shelter and occlusion(default size = 20)
    *analysis* : 

    + Sometimes shadow on key point(mouth, eyes) can cause some distortion and black spot. (*shadow_1.jpg, shadow_2.jpg, shadow_3.jpg*)

  + random color(color dithering)
    *operation* : apply contrast, saturation, brightness and sharpness transform to the whole image with a random factor

    *analysis* : 

    + Because of using random factor, it is hard to quantitative analysis. But it would be appearant that excessive transformation can cause severe artifacts(*color_1.jpg, color_2.jpg, color_3.jpg*)

  + blur
    *operation* : apply Gauss Filter to blur the image
    *analysis* : 

    + Blur show no influence on generated images(in proper range). 

  concrete result images under the directory *data-augmentation*

+ **Hard Case**

  + Glasses
    Glasses frames often produces black shdows and strange wrinkle (*glasses_1.jpg, glasses_2.jpg*)
  + Profile Bangs
    Bangs in proile can cause overlapping shadow and other artifacts (*bangs_1.jpg*) 
  + Black-and-White Photograph
    Black-and-white photograph is not hard case(*black_white.jpg*) 

  concrete result images under the directory *hard-case*

+ **Run Time Test**

  In a GTX 1080Ti, Pytorch0.3.1 enviromment, *G* executes a forward cost 0.065s averagely.

+ **Limit**

  + Due to limited training set and train strategy problems, different face identity generates some similar results. Enrich the dataset and add some identity preserving loss like perceptual loss will be helpful.
  + Single-Image DR-GAN works not well in large degrees(60+). Use Multi-Image to complement information or draw lessons from symmetrical loss in TP-GAN
  + Skin color is hard to learn. The result is almost white when input image is black.
  + Train strategy need to be improved a lot, it's still difficult to converge.

  