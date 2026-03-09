function output=Video_SigEstmate_SigCNN(noisy)


for i=1:1
    temp_noisy = squeeze(noisy(:,:,i));
    output(i)=SigEstmate_SigCNN(temp_noisy);
    
end

end