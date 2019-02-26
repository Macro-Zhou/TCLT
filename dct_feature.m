function [ out_entropy,out_skewness,out_diff_entropy ] = dct_feature( dis_img,range1,range2,range3 )
%DCT_ENTROPY Summary of this function goes here
%   Detailed explanation goes here
[h,w,~]=size(dis_img);
h_block=floor(h/8);
w_block=floor(w/8);
out_energy=cell(1,h_block*w_block);
out_energy_diff=cell(1,h_block*w_block);
curr_skewness=zeros(1,h_block*w_block);
len=[2;3;4;5;6;7;8;7;6;5;4;3;2;1];
cnt=1;
for i=1:h_block
    for j=1:w_block
        curr_block=dis_img((i-1)*8+1:i*8,(j-1)*8+1:j*8);
        temp_coeff=dct2(curr_block-mean(curr_block(:)));
        temp_coeff=temp_coeff.^2;
        temp_energy=zeros(14,1);
        for k=1:14
            for m=1:8
                for n=1:8
                    if m+n==k+2
                        temp_energy(k)=temp_energy(k)+temp_coeff(m,n);
                    end
                end
            end
        end
        temp_energy=sqrt(temp_energy./len);
        temp_energy_diff=temp_energy(1:13)-temp_energy(2:14);
        out_energy{cnt}=temp_energy;
        out_energy_diff{cnt}=temp_energy_diff;
        if var(temp_energy)==0
            curr_skewness(cnt)=0;
        else
            curr_skewness(cnt)=skewness(temp_energy);
        end       
        cnt=cnt+1;
    end
end
out_skewness=hist(curr_skewness,range1);
out_skewness=out_skewness/sum(out_skewness);
out_energy=cell2mat(out_energy);
out_energy_diff=cell2mat(out_energy_diff);
out_entropy=zeros(14,1);
out_diff_entropy=zeros(13,1);
for i=1:14
    [out_entropy(i),~]=coeff_entropy(out_energy(i,:),range2);
    if i<14
        [out_diff_entropy(i),~]=coeff_entropy(out_energy_diff(i,:),range3);
    end    
end

end

