function [ img_feature ] = TCLT_feature( img )
%TCLT_FEATURE Summary of this function goes here
%   Detailed explanation goes here
img_feature=cell(6,1);

c = size(img,3);
map_data = load('map_data_16_ri.mat'); 
mapping = map_data.mapping; 
range_wavelet=-200:0.5:200;
range_dct1=-1.5:0.1:3.5;
range_dct2=0:0.5:255;
range_dct3=-255:1:255;
if c==1    
    %lbp
    lbp_feature = lbp(img,2,16,mapping,'nh');
    %wavelet
    [wavelet_entropy,wavelet_kld]=wavelet_feature(img,4,range_wavelet);
    %dct
    [dct_entropy,dct_skewness,dct_diff_entropy]=dct_feature(img,range_dct1,range_dct2,range_dct3);
else
    img_gray = rgb2gray(img);
    img_ycbcr = rgb2ycbcr(img);
    %lbp
    lbp_feature = lbp(img_gray,2,16,mapping,'nh');
    %wavelet
    [temp_entropy_y,temp_kld_y]=wavelet_feature(img_ycbcr(:,:,1),4,range_wavelet);
    [temp_entropy_cb,temp_kld_cb]=wavelet_feature(img_ycbcr(:,:,2),4,range_wavelet);
    [temp_entropy_cr,temp_kld_cr]=wavelet_feature(img_ycbcr(:,:,3),4,range_wavelet);
    wavelet_entropy=[temp_entropy_y;temp_entropy_cb;temp_entropy_cr];
    wavelet_kld=[temp_kld_y;temp_kld_cb;temp_kld_cr]; 
    %dct
    [temp_dct_entropy_y,temp_dct_skewness_y,temp_dct_diff_entropy_y]=dct_feature(img_ycbcr(:,:,1),range_dct1,range_dct2,range_dct3);
    [temp_dct_entropy_cb,temp_dct_skewness_cb,temp_dct_diff_entropy_cb]=dct_feature(img_ycbcr(:,:,2),range_dct1,range_dct2,range_dct3);
    [temp_dct_entropy_cr,temp_dct_skewness_cr,temp_dct_diff_entropy_cr]=dct_feature(img_ycbcr(:,:,3),range_dct1,range_dct2,range_dct3);
    dct_entropy=[temp_dct_entropy_y;temp_dct_entropy_cb;temp_dct_entropy_cr];
    dct_skewness=[temp_dct_skewness_y,temp_dct_skewness_cb,temp_dct_skewness_cr];
    dct_diff_entropy=[temp_dct_diff_entropy_y;temp_dct_diff_entropy_cb;temp_dct_diff_entropy_cr];
end

img_feature{1}=lbp_feature;
img_feature{2}=wavelet_entropy;
img_feature{3}=wavelet_kld;
img_feature{4}=dct_entropy;
img_feature{5}=dct_skewness;
img_feature{6}=dct_diff_entropy;

end

