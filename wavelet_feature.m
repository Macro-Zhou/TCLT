function [ out_entropy,out_kld ] = wavelet_feature( dis_img,level,range )
%WAVELET_FEATURE Summary of this function goes here
%   Detailed explanation goes here
[coeff,s]=wavedec2(dis_img,level,'bior3.7');
H=cell(level,1);
V=cell(level,1);
D=cell(level,1);
he=zeros(level,1);
ve=zeros(level,1);
de=zeros(level,1);
len=length(range);
h_dis=zeros(len,level);
v_dis=zeros(len,level);
d_dis=zeros(len,level);
h_kld=zeros(level-1,1);
v_kld=zeros(level-1,1);
d_kld=zeros(level-1,1);
for i=1:level
    [H{i},V{i},D{i}]=detcoef2('all',coeff,s,i);
    [he(i),h_dis(:,i)]=coeff_entropy(H{i}(:),range);
    [ve(i),v_dis(:,i)]=coeff_entropy(V{i}(:),range);
    [de(i),d_dis(:,i)]=coeff_entropy(D{i}(:),range);
    if i>1
        h_kld(i-1)=kld(h_dis(:,i-1),h_dis(:,i));
        v_kld(i-1)=kld(v_dis(:,i-1),v_dis(:,i));
        d_kld(i-1)=kld(d_dis(:,i-1),d_dis(:,i));
    end
end
out_entropy=[he;ve;de];
out_kld=[h_kld;v_kld;d_kld];
end

