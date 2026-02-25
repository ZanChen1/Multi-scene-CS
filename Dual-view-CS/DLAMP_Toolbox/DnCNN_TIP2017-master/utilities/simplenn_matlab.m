function res = simplenn_matlab(net, input)

%% If you did not install the matconvnet package, you can use this for testing.

n = numel(net.layers);

res = struct('x', cell(1,n+1));
res(1).x = input;

for ilayer = 1 : n
    l = net.layers{ilayer};
    switch l.type
        case 'conv'
            for noutmaps = 1 : size(l.weights{1},4)
                z = zeros(size(res(ilayer).x,1),size(res(ilayer).x,2),'single');
                for ninmaps = 1 : size(res(ilayer).x,3)
                    z = z + convn(res(ilayer).x(:,:,ninmaps), l.weights{1}(:,:,ninmaps,noutmaps),'same');
                end
                res(ilayer+1).x(:,:,noutmaps) = z + l.weights{2}(noutmaps);
            end
        case 'relu'
            res(ilayer+1).x = max(res(ilayer).x,0);
        case 'bnorm'
            %             res(ilayer+1).x=res(ilayer).x;
            
            %             res(ilayer+1).x = vl_nnbnorm(res(ilayer).x, l.weights{1}, l.weights{2}, ...
            %                 'moments', l.weights{3}, ...
            %                 'epsilon', l.epsilon, ...
            %                 bnormCudnn{:}) ;
          
            mu=net.layers{1,ilayer}.weights{1,3}(:,1);
            sigma=net.layers{1,ilayer}.weights{1,3}(:,2);
            G=net.layers{1,ilayer}.weights{1,1};
            B=net.layers{1,ilayer}.weights{1,2};
            for ii=1:64
                res(ilayer+1).x(:,:,ii)=G(ii,1)*(res(ilayer).x(:,:,ii)-mu(ii,1))/(sigma(ii,1)+net.layers{1,ilayer}.epsilon);
            end
    end
    res(ilayer).x = [];
end

end
