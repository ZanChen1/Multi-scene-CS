function Phi = NLR_GenerateProjection(block_size, model)

N = block_size;
if strcmp(model,'Orthogonal')
    filename = ['Orthogonal_' num2str(block_size) '.mat'];  
elseif strcmp(model,'Dct_matrix')
    filename = ['Dct_matrix_' num2str(block_size) '.mat'];  
elseif strcmp(model,'Hadamard_matrix')
    filename = ['Hadamard_matrix' num2str(block_size) '.mat']; 
end


%M=N;
%M = round(subrate * N);

if ((nargin == 3) && exist(filename, 'file'))
    load(filename);
else
    if strcmp(model,'Orthogonal')
        Phi = orth(randn(N, N))';
    elseif strcmp(model,'Dct_matrix')
        Phi = dctmtx(N);
    elseif strcmp(model,'Hadamard_matrix')
        Phi=hadamard(N);
        Phi=Phi./sqrt(N);
    end
end

if ((nargin == 4) && (~exist(filename, 'file')))
  save(filename, 'Phi');
end