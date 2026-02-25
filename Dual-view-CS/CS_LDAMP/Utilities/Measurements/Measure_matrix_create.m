function [A, At, measure] = Measure_matrix_create(measure)
randn('state',0);
rand('state',0);
switch measure.model
    
    case  'Bernoulli'
%         p = 0.5;
%         Phi = double((rand(measure.block_size^2, measure.block_size^2)<p));
%         Phi = Phi*2-1;
%         %Phi = randn(measure.block_size^2,measure.block_size^2);
%         for i =1:size(Phi,2)
%             Phi(i,:) = Phi(i,:)/norm(Phi(i,:));
%         end
%         k = ceil(length(measure.OMEGA)/(measure.image_height*measure.image_width/measure.block_size^2));
% %         Phi_mt = zeros(k,measure.block_size^2);
% %         for j = 1:measure.block_size^2
% %             Phi_mt(1:k,j) = Phi(1:k,j)./(sum(abs(Phi(1:k,j)).^2));
% %             %(sum(abs(Phi(1:k,j)).^2))
% %         end
%         Phi_mt = Phi./(k/measure.block_size^2);
%         Phi_mp = pinv(Phi(1:k,:));


        k = ceil(length(measure.OMEGA)/(measure.image_height*measure.image_width/(measure.block_width*measure.block_height)));
        p = 0.5;
        Phi_B = double((rand(k, measure.block_width*measure.block_height)<p));
        Phi_B = Phi_B*2-1;
        Phi_B = Phi_B./(sqrt(measure.block_width*measure.block_height));
        
        Phi=Phi_B;
        Phi_mt = Phi_B*(measure.block_width*measure.block_height/k);
        Phi_mp = pinv(Phi);
        
    case 'Hadarmad'
        N = measure.block_size^2;
        Phi=hadamard(N);
        Phi=Phi./measure.block_size;
        Phi_mt = Phi';
        Phi_mt = Phi./(k/measure.block_size^2);
        Phi_mp = Phi';
        
    case 'Cartesian'  %%% need to rewrite
        mask = cell2mat(struct2cell(load('Cartesian_0p2.mat')));
        measure.OMEGA = find(mask>0);
        measure.signvec = exp(1i*2*pi*rand(measure.image_width,measure.image_height));
        %measure.signvec = 1;
        n = measure.image_width*measure.image_height;
        m = length(measure.OMEGA );
        lambda = (1/sqrt(n))*sqrt(n/m);
        A=@(z)A_fft2(z,measure.OMEGA,measure.image_width,measure.image_height,lambda,measure.signvec);
        lambda = sqrt(n)*sqrt(n/m);
        At=@(z)At_fft2(z,measure.OMEGA,measure.image_width,measure.image_height,lambda,measure.signvec);
        
    case 'Diffraction' %%% need to rewrite
        
        %load('D_0p1.mat');
        %signvec = mask.perm;
        %SubsampM = mask.picks;
        
        height = measure.image_width;
        width = measure.image_height;
        n = measure.image_width*measure.image_height;
        m = length(measure.OMEGA );
        
        signvec = exp(1i*2*pi*rand(n,1));
        inds=[1;randsample(n-1,m-1)+1];
        I=speye(n);
        SubsampM=I(inds,:);
        
        %A=@(x) SubsampM*reshape(fft2(reshape(bsxfun(@times,signvec,x(:)),[height,width])),[n,1])*(1/sqrt(n))*sqrt(n/m);
        %At=@(x) bsxfun(@times,conj(signvec),reshape(ifft2(reshape(SubsampM'*x(:),[height,width])),[n,1]))*sqrt(n)*sqrt(n/m);
        A=@(x) SubsampM*reshape(fft2(reshape(bsxfun(@times,signvec,x(:)),[height,width])),[n,1]);
        At=@(x) bsxfun(@times,conj(signvec),reshape(ifft2(reshape(SubsampM'*x(:),[height,width])),[n,1]))*n/m;
        
        
        
        
        
    otherwise
        error('Unrecognized measurement model')
        
end


if strcmp(measure.model,'Hadarmad')||strcmp(measure.model,'Bernoulli')
    measure.Phi = Phi;
    measure.Phi_mt = Phi_mt;
    measure.Phi_mp = Phi_mp';
    A=@(z)A_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,measure.Phi);
    At=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,measure.Phi_mt);
else
    
    %     signvec = exp(1i*2*pi*rand(n,1));
    %     inds=[1;randsample(n-1,m-1)+1];
    %     I=speye(n);
    %     SubsampM=I(inds,:);
    %     A=@(x) SubsampM*reshape(fft2(reshape(bsxfun(@times,signvec,x(:)),[measure.image_width,measure.image_height])),[n,1])*(1/sqrt(n))*sqrt(n/m);
    %     At=@(x) bsxfun(@times,conj(signvec),reshape(ifft2(reshape(SubsampM'*x(:),[measure.image_width,measure.image_height])),[n,1]))*sqrt(n)*sqrt(n/m);
end




end