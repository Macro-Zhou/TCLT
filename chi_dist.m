function [ dist ] = chi_dist( input1,input2 )
%CHI_DIST Summary of this function goes here
%   Detailed explanation goes here
temp=input1+input2;
idx=temp==0;
input1(idx)=1;
input2(idx)=1;
dist=sum(((input1-input2).^2)./abs(input1+input2))/2;

end

