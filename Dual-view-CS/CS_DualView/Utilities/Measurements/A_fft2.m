function b = A_fft2(x,mask,h,w,lambda,signvec)

x=reshape(x,[h w]);
x = bsxfun(@times,signvec,x);
x_fft2 = fftshift(fft2(x));
b = x_fft2(mask);
b = b.*lambda;

end