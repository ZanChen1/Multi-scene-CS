function  out=impulse_response(A, At, w, h)



out=zeros(w,h);

for i=1:w
    for j=1:h
        temp = zeros(w,h);
        temp(i,j)=1;
        temp=reshape((At(A(temp(:)))),[w,h]);
        out(i,j)=temp(i,j);
        
    end
    
end
        
    
end