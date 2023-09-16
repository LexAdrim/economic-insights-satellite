# Gaining Economic Insights Through Deep Learning on Satellite Imagery

This repository contains some of the code used in the thesis submitted for my MSc in Statistical Science at Oxford. This thesis explores the potential of high-resolution satellite imagery for predicting urban wealth levels, as indicated by income per capita. Urban infrastructure such as, airports or roads, are recognised drivers of economic growth and are easily identifiable in overhead imagery. Our approach involves designing a two-component pipeline for information extraction from aerial images. The first component is a segmentation model that extracts road networks from overhead images, while the second is an object detection model that locates selected infrastructure. These two components work in tandem, with their outputs fed into a neural network architecture designed for wealth prediction.


<div align="center"> 

![introfig](https://github.com/LexAdrim/economic-insights-satellite/blob/main/figures/introfig.png)
  
</div>

The full thesis is available in the repository. 

## Example of a satellite image with segmented road network and detected infrastructure 

![examplefig](https://github.com/LexAdrim/economic-insights-satellite/blob/main/figures/example.jpg)


