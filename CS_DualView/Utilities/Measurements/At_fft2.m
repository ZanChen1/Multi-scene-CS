function b = At_fft2(x,mask,h,w,lambda,signvec)


x_fft2 = zeros(h,w);
x_fft2(mask) = x;
b = ifft2(ifftshift(x_fft2));
b = bsxfun(@times,conj(signvec),b);
b = b(:);
b = b.*lambda;


end