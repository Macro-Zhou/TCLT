function [ dist ] = kld( curr_dis,neighbor_dis )
%KLD Summary of this function goes here
%   Detailed explanation goes here
idx_curr=(curr_dis==0);
idx_neighbor=(neighbor_dis==0);
idx=idx_curr|idx_neighbor;
curr_dis(idx)=1;
neighbor_dis(idx)=1;
dist=sum(curr_dis.*log2(curr_dis./neighbor_dis));

end

