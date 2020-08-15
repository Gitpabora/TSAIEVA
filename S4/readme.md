{\rtf1\ansi\ansicpg1252\cocoartf1504\cocoasubrtf830
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;\red34\green45\blue53;}
{\*\expandedcolortbl;;\cssrgb\c17647\c23137\c27059;}
\paperw11900\paperh16840\margl1440\margr1440\vieww28300\viewh17220\viewkind0
\deftab720
\pard\tx566\pardeftab720\partightenfactor0

\f0\fs32 \cf2 \expnd0\expndtw0\kerning0
\
    \
 **Assignment 4 Architectural Basics**\
   -\
    \
   \
\
 **How many layers in the network** \
   - \
The Number of layers in the network depends on the level of features we want to extract. As each layer is extracting one feature , simple feature at the begining and complex feature deep down or later layers. \
Theoritically  layers are needed  to be added in the network till the receptive field is equal to the image or the image size becomes 1x1.However practically depending on the kernel size we decide to use  the number of layer will vary . An Example is that in case of MNIST data with English digitsof sixe 28x28, the maximal feature can be well extracted at the image size 7X7.This gives a good option to have only one convolutional layer with kernel size 7X7 before the final output layer.On the other hand there would have been possibility to add two  layers of convolutions with 3x3 kernel to reach the image size of 1x1 , if we would not have decided to use the 7X7 kernel. \
     \
     \
\
 **MaxPooling** \
   - \
Max pooling is a process to pick the dominant information content from a part of imag , at the cost of loosing small part of information. Max Pooling helps down sampling reducing the image size on which it is applied.\
Below is an example of max pooling on  4x4 matrix using 2x2 max pool \
        \
|2 | 6 | 7 | 8 |\
|--|--|--|--|\
|5 |3 | 1 |2  |      \
| 3| 4| 6 |7  |                                     \
|1 |2| 3 |4|                                          \
  \
On max pooling  using 2x2 on the above maxix will result the below where the size of the image matrix is reduced by half because of 2x2 max pool. \
  \
  | 6 | 8 |\
  |--|--|  \
  | 4 | 7|\
 \
 2x2 max pool is commonly used as it is minimal and the the information lost is minimal.\
 \
 **The distance of MaxPooling from Prediction Layer** \
   - \
   The max pooling is not preferred near the prediction layer as importalt information /features can be lost if max pool is done before the output prediction. Hence Max pooling is preferred only before 3 to 4 layers before the outout layer.\
\
  **1x1 Convolutions** \
   - \
   1x1 kernel is used in convolutions which is useful in change ( increase of decrease the number of kernels in the network layer , as per our need. convolvingc with 1x1 through out the image channels is equivalent to muliplying by a nunber.\
   1x1 convolutions are computationally insignificant.\
   1x1 convolution is equivalent to multiplying the channel by one number.\
   With 1x1 , merging of available extractor is possible since in that case that it can be considered that the features that make the composite feasure are found together.\
   It is like weighted sum for the channels, so it can be aslo used as a filtering tool for segregating an object from the back ground.\
 \
  **3x3 Convolutions**\
   - \
   The  kernel used in this convolution is 3 x 3. This size of kernel is predeferred because of its minimality in  terms of lower computation along with the asymetricity being the  off number and most importantly the underlying harware is accelerated in common hardware architectures  making  the code work faster usng . \
   \
  **Receptive Field** \
   - \
   Receptive field is the pixel matrix in the original image that one cell in the kernel / in any other layer sees during convoltios.\
   At the output layer the comple information about the image is well captured when the receptive field is equal to the image . in case of 3x3 convoltions with defailt stride of 1  at every layer from the input layer the receptive field chronologically  groms in a sequence 3x3 --> 5 x5 --> 7 X 7  etc. On Applying t=Max pool the receptive field gets doubled\
   \
 **SoftMax** \
   -  \
   SoftMax function is used at the output layer in case  a Network output has multiple calass results.\
   Softmax function is also known as normalized exponential function.\
   An example for definition softmax for values in X =[x1, x2, x3]  is \
   [exp(x1)/SE ,exp(x2)/SE , exp(x3)/SE ] where SE =exp(x1) + exp(x2) + exp(x3)\
   \
   Softmax of [ 4 , 2, 5 ]  is  [ .04 , .26 , .71 ]  implies that it extends the range for a better feel, and it is like probability  that the sum of the values from softmax is  one.\
   Softmax is very useful in case of multi classification however in case of critical scenarios softmax is not suggested as it changes the original significant data.\
      \
    \
     \
 **Learning Rate** \
   - \
   The learning rate is a parameter to gradient descent algorithm, whne the gradient steps are high learning rate is high. An optimization algorithom which is used in neural network, where the main ofjective is to optimize some loss function. Learning rate in the  the network  effects the training or learning or in the seeking method  for the optimized parameters in the network.\
 This hyperparameter needs tuning to make it work with a network as if the Learning rate is  too small then the time taken by the network in tarining will be very long and when the learning rate is too large , the network  may fail to learn properly.\
   \
   \
 **Kernels and how do we decide the number of kernels?** \
   - \
  A Kernel is a mechanism to extract a feature.The number of kernels to be  used  depends on the number of channels in the input image.\
 e.g, if an image has a size 200 X 200X  32  and needs to be convolved with  3x3  kernels , then 32 such kernels has to be applied. also mathematically it is mandatory.\
if it is asked how many type of kernels e.g  say 1x1 , 3x3 , 7x7 to be used  then it will be totally scenario based and depending on the hardware suitability , Expressivity and interdependence  and intra dependence of  different feature classes decides the use of kernel\
   \
 **Batch Normalization** \
   - \
   Batch normalization is a mechanism of scaling the intensity values so that they remain in a specific range , e.g  ( -1 to 1 ) or. ( 0  to. 1)\
   By  using Batch normalization we can get a bettert resut as the optimazation alogorith can work more efficiently ,  Firstly ,since gradient descent can reach the optimum easily , esulting traing coverses faster.\
Secondly it can allow higher learning rate , while going deeper in the network. Thirdly. it also helps the network  by providing minimal regularization effect. One reason it helps is how is by helping some of the activation function that needs the input values to be in a particular range . and other reason is it adds some minimal noise in the layers that helps like dropout to remove over fitting.\
   \
   \
 **Image Normalization** \
   - \
  Image normalization or contrast stretching is normalizing the intensity values in an images so that they fall in particular range.Usually for the  gray scale images where the intensities are  in the range 0 to 255 , this method of changing or stretching the intensities  helps as the image will have a good contrast as the intensites will be stretched well in the whole range.\
   \
 **Position of MaxPooling** \
   - \
   The position of max pooling has to be such that  important  information is not lost .  It should not be at near input layer as at this we do not want to loose any information . It should be near to output layer as loosing information at the stage may lead to  incorrectness.At the position when the basic features edges and gradients are visible , Max pooling layer is most beneficial as capturing the dominent information will keep the prominent features or information intact.\
   \
 **Concept of Transition Layers** \
   - \
   Whenever we require to add layers and change the number of channels Transition from one  channel size to another is required.  Transition layer can be constructed using 1x1 kernel.  e.g. for an input  resolution 32 x 32 and channel size 10 if we wnat to change the channel size to channel size 4 then 10 numbers of 1x1 kernel can do the trick.\
In known layered architectures  e.g. in the changing sequence of Channel sizes  32  -> 64 ->128 ->256 -> 512 -> 32->64 -->128 -->256 --> 512 -->32-->64 ... so on, the transition block comes into picture while  changing from e.g 512 --->32\
and the rest is convolution block. So the network is structed as A number of convolution  layers  followed by Trnasition layer  then again transition block of layers.\
    \
   \
 **Position of Transition Layers** \
   - \
   Position of transition layer can be followed by a number of convolution layers , where there  is a need to change the channel size , as mentioned above.\
\
\
   \
 **Number of Epochs and when to increase them**\
   -\
   One Epoch means the interval when the same image is seen by the network 2nd time.\
   A dataset of N examples  if devided into batches of B_size , it will take (N / B_Size )iterations to complete one Epoch\
  \
  When we  see that the training has not gone well ., and the training accuracy is lower than expected , Epoch can be increased for retraining.\
 \
 \
   \
**DropOut** \
  -\
  Drop out means removing some kernels, so as not to allow to remember some information at the layers.\
  Drop out is a very good solution to overfitting as the network will unlearn part of information.\
  Drop out can increase the gap between the Training accuracy and validation accuracy.\
  \
  \
**When do we introduce DropOut, or when do we know we have some overfitting** \
  - \
  When ther is a sign in overfitting , the Training accuracy is already high. The validation accuracy is not  changing  and not going higher with  more epochs, then we should apply drop out so that it will increase the gap between the two.\
   \
\
\pard\tx566\pardeftab720\partightenfactor0
\cf2  **The distance of Batch Normalization from Prediction**\
   - \
   Batch normalization can be used before and after every convolution layers in the network.  Only just before the prediction layer, ie after the final convolution before the output layer it is not used.  At the output layer the actual information need to be intact  for prediction to happen correctly with all  negative or positive values  . in case of multi-class prediction the is a softmax that acts like probability.\
   \
\pard\tx566\pardeftab720\partightenfactor0
\cf2 \
**When do we stop convolutions and go ahead with a larger kernel or some other alternative ** \
  -  \
 Looking at the image, if the features are well extracted already and not use of convolving the small size output images again then we have to decide on a larger kernel based on the scenario. For example  in case of MNIST at the image size 7x7,the features are already extracted , we sould not not be convolving using 3x3  when the images size is very very small and it is time to apply a page kernel of size 7x7 .\
\
**How do we know our network is not going well, comparatively, very early** \
  -\
  We have to be alert to know the network condition early enough , below are few points \
  a)when the time consumed is very high. \
  b)when the Training accuracy is very low in first few epoch and   The training accuracy is not improving at  all after  epochs, The validation accuracy is comparatively very very low.\
  A very importtant point is to write down the layered structure before any implementation.\
   \
**Batch Size, and effects of batch size** \
  -\
  A dataset can be divided into a batches. if a dataset has N examples and it is devided into batches of B_size , it will take (N / B_Size )iterations to complete one Epoch.\
  Depending on computing infrastructure choosing the batch size is a good option as it will result in faster executions.\
  \
  \
  \
**When to add validation checks** \
  -\
  While  using Model.fit , to apply  the network architeture on the  input data validation checkes can be added. Keras  provides validation functionality ,  e.g in case of Mnist \
   model.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_test, Y_test) )\
 Adding the Validation check will log the Validation accuracy at every step.\
 Keras also provide a functionality to call user defined Callback functions  e.g while chaning the optimizer learning rate  e.g \
  model.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_test, Y_test), callbacks=[LearningRateScheduler(scheduler, verbose=1)])\
 here callback has to list the functions that has to be applied.\
  \
  \
  \
**LR schedule and concept behind it** \
  -\
  LR (Learning rate) schedules tries to change  the learning rate while training.\
  It can be changed by decrementing the rate  according to a known schedule.\
  Few other LR schedules are time-based decay, step decay, exponential decay.\
  As and when gradient descent becomes smaller the learning rate has to be small\
   \
**Adam vs SGD** \
  -\
  Adam stands for 'ADAptive Momentum'. It is combination of  \'92Momentum Gradient Descent\'92 and \'92RM- SProp\'92 optimization  algorithms\
  \
 SGD is a kind of gradient descent algorithm that calculates the gradient on a small portion of data/sample or subset taken randmoly from the data.  its performance is same as gradient descent when learning rate is low\
 \
\
}