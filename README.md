# Training Generative-Adversarial Networks with Adversarial Attacks
Slava Pirogov's coursework Moscow HSE AMI 2022

## Abstract
First Generative Adversarial Networks (GANs) appeared only a few years ago, but they quickly start to show the best results in the image generation, and also quite good for creating sound. At the 
same time, GANs are flexible tool, which can generate a variety of entities. There are many ways to make the GAN work better, and often training is slow or perform badly, making it worth to use 
absolutely everything that could help. Training with Adversarial Attacks is an effective, albeit time-consuming, method which this work describes. All code is written in Python, models are created 
using PyTorch version 1.10.

## Аннотация
Первые Генеративно Состязательные Сети появились всего несколько лет назад, однако они быстро стали показывать лучшие результаты по генерации картинок, а также весьма неплохие для создания звука. 
При этом данные модели являются гибким инструментом, с помощью которых можно генерировать самые различные сущности. Существует множество способов заставить Генеративно Состязательную Сеть работать 
лучше, при этом зачастую обучение проходит медленно, либо показывает плачевные результаты, из-за чего стоит использовать абсолютно все, что может помочь. Обучение при помощи Состязательных Атак как 
раз и является полезным, хоть и трудоемким способом, которое и будет рассмотрено в этой работе. Весь код написан на Python, модели созданы с помощью PyTorch версии 1.10.

## Formulation of the problem
The purpose of this work is to explore different ways of building GANs and compare them with GANs that have been trained using Adversarial Attacks. The main idea is to stabilize learning with 
attacks on Discriminator, that will work like regularization

## Main results
• Developed a theory of GAN Adversarial Training
Adversarial Attacks were investigated on the following GANs:
• WGAN: Huge improvement of key metrics, stabilizing Generator loss.
• WGAN-GP: Similar results as in the classic version, destabilizing Discrim- inator loss.
• DCGAN: Similar results as in the classic version.
• SNGAN: Small improvement in key metrics, stabilizing Generator loss.

Almost 100 experiments were conducted with different implementations and hyperparameters, spent more than 30 days of computing resources, 4 different GAN architectures were explored with FGSM 
attacks.

## Acknowledgments
This research was supported in part through computational resources of [HPC facilities at HSE (University Kostenetskiy, Chulkevich, and Kozyrev 
(2021))](https://iopscience.iop.org/article/10.1088/1742-6596/1740/1/012050)

# Navigation



