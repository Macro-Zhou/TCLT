function [ dmos_pred ] = feature_combine_knn( dist,dmos,k )
%FEATURE_COMBINE Summary of this function goes here
%   Detailed explanation goes here
if size(dist,1)>1
    dist=prod(dist);
end
dist_reg=1./dist;
[sort_dist,idx]=sort(dist_reg,'descend');
range=1:k;
w=sort_dist(range)./sum(sort_dist(range));
dmos_pred=sum(dmos(idx(range)).*w);

end

