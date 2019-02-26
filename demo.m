clc;
close all;
img=imread('test_img.jpg');
tic;
pred_score=TCLT_score(img,5);
toc;
