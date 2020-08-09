# Violence-Detection-in video
Real time violence in a video is a big problem, as it needs lot of computation power to process a video. In this project I did the comparison between the popular name in this field i.e., Optical Flow, and with my new way. The results were almost similar. I applied CNN-LSTM to train on hockey fight and violence in movies data set. Model was giving ~98% accuracy on training data set and ~90% on the test data set in just 50 epochs.
### Methodology Used
Extracted 40 continueous frames and applied optical flow to these to catch the temporal activaties. But It was taking too much time. In order to reduced the computation, I took 2 frames let them be t and t+1, and subtracted t+1 with t but the output was too noisy. After analysis of the output i found i that it was -10 to 10 values that are resulting into that noise, That could be because of shadows or camera. I converted this range to 0 and almost all the noise was removed. Results can be found in sample folder.  
### Sample folder
Contains 2 original videos and their results after applying transformation.
