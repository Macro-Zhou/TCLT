function [ output,prob_dis ] = coeff_entropy( input,range )
%COEFF_ENTROPY Summary of this function goes here
%   Detailed explanation goes here
temp=hist(input,range);temp=temp/sum(temp);
prob_dis=temp;
temp(temp==0)=1;
output=sum(-temp.*log2(temp));

end

