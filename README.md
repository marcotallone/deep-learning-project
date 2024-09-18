<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url]
[![Gmail][gmail-shield]][gmail-url] -->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/marcotallone/deep-learning-project">
    <img src="images/brain.png" alt="Logo" width="120" height="120">
  </a>

<h2 align="center">Multi-Model Approach for Brain Tumor Classification and Segmentation</h2>
<h4 align="center">Deep Learning Course Exam Project</h4>
<h4 align="center">SDIC Master Degree, University of Trieste (UniTS)</h4>
<h4 align="center">2024-2025</h4>

  <p align="center">
    A deep learning project for brain tumor classification and segmentation on MRI images using CNN, U-Net, and VIT models.
    <br />
    <br />
    <table>
      <tr>
        <td><a href=""><strong>View Demo</strong></a></td>
        <td><a href="https://github.com/marcotallone/deep-learning-project/issues"><strong>Report bug</strong></a></td>
        <td><a href="https://github.com/marcotallone/deep-learning-project/issues"><strong>Request Feature</strong></a></td>
    </table>
</div>

<!-- TABLE OF CONTENTS -->
<div style="width: 360px; text-align: center; border: 2px solid currentColor; padding: 10px 10px 10px 10px; border-radius: 10px; margin: auto;">
  <h4>📑 Table of Contents</h4>
  <ul style="list-style-type: none; padding: 0;">
    <li><a href="#students-info">Students Info</a></li>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage Examples</a></li>
    <li><a href="#datasets-description">Datasets Description</a></li>
    <li><a href="#models-description">Models Description</a></li>
    <li><a href="#performance-assessment">Performance Assessment</a></li>
    <li><a href="#conclusions">Conclusions</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#references">References</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ul>
</div>
</br>

<!-- STUDENTS INFO-->
## Students Info

| Name | Surname | Student ID | UniTS mail | Google mail | Master |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Stefano | Lusardi | SM3600001 | <stefano.lusardi@studenti.units.it> | <stefanosadde@gmail.com> | **SDIC** |
| Marco | Tallone | SM3600002 | <marco.tallone@studenti.units.it> | <marcotallone85@gmail.com> | **SDIC** |
| Piero | Zappi | SM3600004 | <piero.zappi@studenti.units.it> | <piero.z.2001@gmail.com> | **SDIC** |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ABOUT THE PROJECT -->
## About The Project

Deep learning models are widely used in medical imaging for their ability to learn complex patterns and features from images. In particular, convolutional neural networks (CNNs) have for long time been the most common models used for image **classification**, i.e. the task of assigning a label to an image and **segmentation**, i.e. the task of identifying and delineating the boundaries of objects in an image. In recent times, however, transformers models have been introduced in the field of computer vision and have shown to be very effective in the former tasks.\
This project aims to develop multiple deep learning models for brain tumor classification and segmentation on magnetic resonance imaging (MRI) images using CNN, U-Net, and VIT models in order to compare their performance as well as their strengths and weaknesses.\
Given an input MRI image, the **classification** task in this context consists in predicting the type of tumor present in the scan if any. Anomg the possible tumor classes found in the adopted dataset, the implemented networks were trained to classify between *glioma*, *meningioma*, *pituitary*, and *no tumor* classes.\
On the other hand, the **segmentation** task consists in identifying different tumor regions in input MRI images. In this case, in fact, the dataset consisted of images labelled with multiple masks highlighting *necrotic and non-enhancing tumour core (NCR/NET)*, *edema (ED)*, and *enhancing tumour (ET)* regions respectively.\
For the development of the models, the following two datasets have been used:

- For the **classification** task, the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) has been used.
- For the **segmentation** task, the [BraTS 2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) has been used.

With the data at our disposal, we have developed the following models:

- For the **classification** task:
  - CNN model
  - AlexNet model
  - VGG-16 model
  - A VIT model

- For the **segmentation** task:
  - U-Net model 1
  - U-Net model 2
  - U-Net model 3

Further details about the datasets and the implemented models are given below, after installation instructions, dependencies requirements and usage examples.
<!-- TODO: Add datasets detailed description and models description and performance -->

### Project Structure

The project is structured as follows:

```bash
.
├── ...
```
  
### Built With

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!--TODO LiST -->
## TODO

List of things to be done for the project:

- [x] Create a CNN model for classification
- [x] Create a U-Net model for segmentation
- [ ] Create a VIT model for classification
- [ ] Write README file with all the information
- [ ] Add residual/skip connections to the CNN models
- [ ] Fix U-Net input images size and kernels parameters for faster training...
- [ ] Visualize CNN layers/kernels to see if there is something interesting they observe
- [ ] Compare CNN models (AlexNet, VGG, ...) one with the other (in particular prediction confidence before extracting softmax)
- [ ] Find suitable metric for segmentation models predictions
- [ ] Extend U-Nets model for segmentation to see if they can predict life expectancy of the patient from tumor prediction
- [ ] Create a presentation for the project
- [ ] REMOVE TODO and USEFUL LINKS AT THE END
- [ ] See if the time a patient still has to live is predictable
- [ ] Review static typing in models definitions
- [ ] ... **(add more things to do)**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USEFUL LINKS -->
## Useful Links

- [Classification Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

- [Segmentation Dataset](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)

- [CNN YouTube Videos](https://www.youtube.com/watch?v=ArPaAX_PhIs&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)

- [Brain Tumor Segmentation U-Net Notebook](https://www.kaggle.com/code/auxeno/brain-tumour-segmentation-cv)

- [Trasformers in Pytorch](https://www.kaggle.com/code/auxeno/transformers-from-scratch-dl)

- [VIT - Transformers for images in Pytorch (video)](https://www.youtube.com/watch?v=ovB0ddFtzzA) -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Requirements

### Installation

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage Examples

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DATASETS DESCRIPTION -->
## Datasets Description

### Brain Tumor MRI Dataset (Classification)

### BraTS 2020 Dataset (Segmentation)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MODELS DESCRIPTION -->
## Models Description

### Classification Models

### Segmentation Models

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- PERFORMANCE ASSESSMENT -->
## Performance Assessment

### Classification Models

### Segmentation Models

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONCLUSIONS -->
## Conclusions

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- REFERENCES -->
## References

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

- [Repository for the Deep Learning Course Labs/Practica (UniTS, Spring 2024)](https://github.com/emaballarin/deeplearning-units)
- [Best-README-Template](https://github.com/othneildrew/Best-README-Template?tab=readme-ov-file)
- [Freepik](https://www.flaticon.com/free-icons/machine-learning")

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[forks-shield]: https://img.shields.io/github/forks/marcotallone/deep-learning-project.svg?style=for-the-badge
[forks-url]: https://github.com/marcotallone/deep-learning-project/network/members
[stars-shield]: https://img.shields.io/github/stars/marcotallone/deep-learning-project.svg?style=for-the-badge
[stars-url]: https://github.com/marcotallone/deep-learning-project/stargazers
[issues-shield]: https://img.shields.io/github/issues/marcotallone/deep-learning-project.svg?style=for-the-badge
[issues-url]: https://github.com/marcotallone/deep-learning-project/issues
[license-shield]: https://img.shields.io/github/license/marcotallone/deep-learning-project.svg?style=for-the-badge
[license-url]: https://github.com/marcotallone/deep-learning-project/blob/master/LICENSE.txt
<!-- [linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/marco-tallone-40312425b -->
<!-- [gmail-shield]: https://img.shields.io/badge/-Gmail-black.svg?style=for-the-badge&logo=gmail&colorB=555
[gmail-url]: mailto:marcotallone85@gmail.com -->
