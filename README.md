# Hand-Gesture-Recognition
Usig Hand Gesture Recognition  American sign language can be interpreted into normal English language and this acts as a bridge between people who use ASL(mute people) and normal people 


Sign language is one of the oldest and most natural form of language for communication, but since most people do not know sign language and interpreters are very difficult to come by we have come up with a real time method using neural networks for fingerspelling based american sign language. In our method, the hand is first passed through a filter and after the filter is applied the hand is passed through a classifier which predicts the class of the hand gestures. Our method provides 95.7 % accuracy for the 26 letters of the alphabet.

Data Pre processing and Algorithm :
Used Open computer vision(OpenCV) library in order to produce our dataset.Firstly we captured around 800 images of each of the symbol in ASL 		for training purposes and around 200 images per symbol for testing purpose. First we capture each frame shown by the webcam of our machine. In the 	each frame we define a region of interest (ROI) which is denoted by a blue 	bounded square as shown in output section.


Algorithm Layer 1:
1.Apply gaussian blur filter and threshold to the frame taken with opencv to get the processed image after feature extraction.
2.This processed image is passed to the CNN model for prediction and if a letter is detected for more than 50 frames then the letter is printed and taken into consideration for forming the word.
3.Space between the words are considered using the blank symbol.

Algorithm Layer 2:
1.We detect various sets of symbols which show similar results on getting detected.
2.We then classify between those sets using classifiers made for those sets only.
