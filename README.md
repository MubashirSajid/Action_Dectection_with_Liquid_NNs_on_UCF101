# Main.py
This file has an implementation where the screen opens and closes per sequence. The sequqence length selection is much more simple as compared to main2.py. Just have to change the sequence length in the last section and all done.


# Main2.py
Actual real time implementation where the window for cam is open always. Text is displayed in the window of what the model has predicted. To change the sequence, you need to hard code variables in the code at the end section. 


# Main3.py
Each passing frame is added to the window to be used in prediction. This can be seen as total real time implementation. This implmentation uses a double ended queue.


# Main4.py
Adding attention map to main3. But the attention map is for resnet


# Main5.py
Trying to add attention map from Liquid NN

# Main6.py
Same as Main4 but without resnet 18. Instead own conv layers used



#### Right now there is no way to evaluate how well the model is performing