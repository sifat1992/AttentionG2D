### Attention Enhanced Learnable 2D Gaussian Filters for Computationally Efficient Abdominal Organ Classification

## Key Features
- Gaussian 2D layer 
- Supervised learning 
- VGG16, ResNet50, DenseNet121, AlexNet, Custom CNN 
- Ultrasound imagery
- Squeeze and Excitation (SE) and Efficient Channel Attention (ECA)

## Abstract
Medical image processing plays a crucial role in disease diagnosis and clinical decision-making, demanding highly accurate and efficient classification models. However, state-of-the- art deep learning models used for medical image classification typically have large parameter counts, making them computationally intensive and unsuitable for real-time applications. To
address this challenge, this paper introduces a novel, computationally efficient neural network architecture incorporating a learnable 2D Gaussian filter for the real-time classification of abdominal organs. Unlike traditional convolutional layers, the proposed 2D Gaussian filter requires only four learnable parameters, two means and two standard deviations corresponding to
each spatial axis, significantly reducing computational complexity. Comparative evaluations were conducted against prominent baseline models, including VGG-16, ResNet-50, DenseNet-121, AlexNet, and a custom CNN. The proposed model achieved an accuracy of 87.81%, substantially outperforming the baseline models while maintaining the lowest parameter count. Further-
more, the performance of the proposed architecture was further enhanced by integrating attention mechanisms such as Squeeze-and-Excitation (SE), and Efficient Channel Attention (ECA-Net). Among these, the SE mechanism with 64 filters provided the best performance, achieving an accuracy of 91.22%, along with marked improvements in Precision, Recall, and F1 scores,
with only a minimal increase in parameters. This study thus demonstrates the potential of the proposed model for efficient and accurate real-time medical image processing.


## Dataset
- Custom curated ultrasound dataset  
- Not publicly available due to privacy and sensitivity concerns  
- Contact for access if needed for research collaboration


## Motivation
For real-time clinical use, medical image classification requires both efficiency and accuracy. However, the majority of cutting-edge deep learning models have a high number of parameters and are computationally demanding, which restricts their practical application. Because of this, there is a need for high-performing, lightweight architectures that strike a balance between speed and accuracy, allowing for scalable and useful healthcare solutions.


### Results
ROC curve 1 and PR curve 2 show the impacts of SE block in the architecture. ROC curve 2 and PR1 show the baseline model's persormance.
| ROC Curve 1 | ROC Curve 2 |
|-------------|-------------|
| [ROC1](Results/stv-2.png) | [ROC2](Results/sc.png) |

| PR Curve 1  | PR Curve 2  |
|-------------|-------------|
| [PR1](Results/gsetv.png) | [PR2](Results/gsec.png) |



# Project Structure
```
├── Results                     
│   ├── git1.png
    ├── roc1.pdf
    ├── roc2.pdf
    ├── recall1.pdf
    ├── recall2.pdf
├── codes
|   ├── pg_decoders.py                      
    ├── pg_encoders.py                
    ├── evaluate.py         
    ├── latent_dis.py             
    ├── latent_model.py         
    ├── optimizer.py                
    ├── train.py   
    ├── utils.py              
    ├── Attn_models.py               
    ├── layers.py               
    ├── pg_networks.py
    ├── eval_example.yaml
    ├── train_example.yaml               
    └── .gitignore                   
    ├── README.md
├── requirements.txt                      
  ```


```bash
# Clone this repo
git clone https://github.com/sifat1992/SEPIAD.git
cd SEPIAD

# Install dependencies
pip install -r requirements.txt
```

## References
1. Nina Tuluptceva, Bart Bakker, Irina Fedulova, Anton Konushin
   “PERCEPTUAL IMAGE ANOMALY DETECTION.” [arXiv:1909.05904](https://arxiv.org/pdf/1909.05904) 


## Author
- **Sifat Z. Karim** — Graduate Student, Mississippi State University  
  📧 [sifatzinakarim1992@gmail.com](mailto:sifatzinakarim1992@gmail.com)  
  🧑‍💻 GitHub: [@sifat1992](https://github.com/sifat1992)

## Contact

For questions, suggestions, or collaboration opportunities, feel free to reach out!  
I’m happy to receive feedback and open to connecting with fellow researchers.

## License
This project is licensed under the [MIT License](LICENSE).


---



