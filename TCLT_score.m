function [pred_score,img_feature] = TCLT_score( img,k )
%TCLT_PREDICTION Summary of this function goes here
%   Detailed explanation goes here
%   img: the input color image
%   ref_feature: the features of all annotated images -> 6 x 1 cell data 
%   1: LBP, 2: wavelet entropy, 3: wavelet intersubband KLD, 4: DCT entropy, 
%   5: DCT skewness, 6: DCT difference entropy
%   ref_dmos: the DMOS of all annotated images -> 5 x 1 cell data
%   1: JP2K, 2: JPEG, 3: WN, 4: Blur, 5: FF
%   model: the classifier model 
%   ps: the mapminmax normalization parameter
%   k: the k nearest neighbor number
ref_data=load('annotated_data_all.mat');
classifier=load('classifier_data.mat');

ref_feature=ref_data.ref_feature;
ref_dmos=ref_data.ref_dmos;
model=classifier.model;
ps=classifier.ps;

img_feature=TCLT_feature(img);
disp('Feature extraction is done');
%% introduce the distortion type identification
temp = mapminmax('apply',img_feature{1}',ps);
test_lbp = temp';
% generate distortion type probability
[~,~,prob]=svmpredict(1,test_lbp,model,'-b 1 -q');
% feature normalization
cand1=img_feature{1}';%LBP
cand2=img_feature{2};%wavelet entropy
eng=sum(cand2);
if eng~=0
    cand2=cand2/eng;
end
cand3=img_feature{3};%wavelet KLD
eng=sqrt(sum(cand3.^2));
if eng~=0
    cand3=cand3/eng;
end
cand4=img_feature{4};%dct entropy
eng=sum(abs(cand4));
if eng~=0
    cand4=cand4/eng;
end
cand5=img_feature{5}';%dct skewness
eng=sum(abs(cand5));
if eng~=0
    cand5=cand5/eng;
end
cand6=img_feature{6};%dct diff entropy
eng=sum(abs(cand6));
if eng~=0
    cand6=cand6/sum(abs(cand6));
end
% predict the quality score
if size(img,3)>1
    ref_feature=ref_feature{2};
else
    ref_feature=ref_feature{1};
end
ref_lbp=ref_feature{1};
ref_wavelet_entropy=ref_feature{2};
ref_wavelet_diff_entropy=ref_feature{3};
ref_dct_entropy=ref_feature{4};
ref_dct_skewness=ref_feature{5};
ref_dct_diff_entropy=ref_feature{6};
trans_dmos_type=zeros(5,1);
for i=1:5
    ref1=cell2mat(ref_lbp{i})';%LBP
    ref2=cell2mat(ref_wavelet_entropy{i}');%wavelet entropy
    ref2=ref2./repmat(sum(ref2),size(ref2,1),1);
    ref3=cell2mat(ref_wavelet_diff_entropy{i}');%wavelet KLD
    ref3=ref3./sqrt(repmat(sum(ref3.^2),size(ref3,1),1));
    ref4=cell2mat(ref_dct_entropy{i}');%dct entropy
    ref4=ref4./repmat(sum(abs(ref4)),size(ref4,1),1);
    ref5=cell2mat(ref_dct_skewness{i})';%dct skewness
    ref5=ref5./repmat(sum(abs(ref5)),size(ref5,1),1);
    ref6=cell2mat(ref_dct_diff_entropy{i}');%dct difference entropy
    ref6=ref6./repmat(sum(abs(ref6)),size(ref6,1),1);
    dist1=feature_dist(cand1,ref1);
    dist2=feature_dist(cand2,ref2);
    dist3=feature_dist(cand3,ref3);
    dist4=feature_dist(cand4,ref4);
    dist5=feature_dist(cand5,ref5);
    dist6=feature_dist(cand6,ref6);
    switch i
        case 1
            dist=[dist1;dist2;dist3;dist5];
        case 2
            dist=[dist1;dist2;dist3;dist5;dist6];
        case 3
            dist=[dist1;dist2;dist3;dist4];
        case 4
            dist=[dist1;dist2;dist3;dist6];
        case 5
            dist=[dist1;dist2;dist3;dist5];
    end
    trans_dmos_type(i)=feature_combine_knn(dist,ref_dmos{i}',k);
end
pred_score=dot(trans_dmos_type,prob);
disp(['Predicted perceptual quality is ',num2str(pred_score)]);
%}

end

