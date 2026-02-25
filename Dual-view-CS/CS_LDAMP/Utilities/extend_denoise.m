function rec_im = extend_denoise(denoi, noise_im, sigma_hat)

noise_im_extend = self_extend_transform(noise_im, 0);
rec_im_extend = cell(size(noise_im_extend));
[h, w]= size(noise_im);
for i =1:8
    rec_im_extend{i} = reshape(double(denoi(noise_im_extend{i},sigma_hat)),[h, w]);
end

rec_im = self_extend_transform(rec_im_extend, 1);

rec_im = rec_im(:);

end