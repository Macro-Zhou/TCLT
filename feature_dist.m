function [ dist ] = feature_dist( input_feature,ref_feature )
%FEATURE_DIST Summary of this function goes here
%   Detailed explanation goes here
feature_num=size(ref_feature,2);
temp_input=repmat(input_feature,1,feature_num);
dist=chi_dist(temp_input,ref_feature);
if sum(dist==0)~=0
    if sum(dist==0)==feature_num
        dist=ones(feature_num,1);
    else
        idx=(dist~=0);
        dist=dist+min(dist(idx))*0.01;
    end
end

end

