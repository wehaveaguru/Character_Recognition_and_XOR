# Character Recognition and XOR MLP

My first two stepping stones to get the essence of neural networks.

## XOR MLP
To build the XOR MLP, I first studied what MLP is. Then, I went on to study feed forwarding, and backpropagation. I learnt the calculus of backpropagation and I have implemented it in the MLP.
I have used sigmoid function as the activation function. 

MLP structure: 4 inputs, 2 features each
2->2->2->1

Results:
<img width="937" height="326" alt="outputs" src="https://github.com/user-attachments/assets/afcc5f76-6352-47a9-a723-bbe3c15afeea" />

## Character Recognition MLP

I used the EMNIST digits dataset. Since the size of the dataset was too large, I only took a portion of it, around 10,000 samples. 
This MLP identifies the handwritten digit

I have used one_hot_y. For the activation function I have used softmax and RELU

MLP STRUCTRE: 10,000 samples, 784 features each

784->128->64->10

Results:


<img width="752" height="272" alt="cr_nn_output" src="https://github.com/user-attachments/assets/9297e44c-10e2-47b2-a658-e2d43f130b5b" />
