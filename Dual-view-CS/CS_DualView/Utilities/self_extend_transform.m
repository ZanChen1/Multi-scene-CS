function output = self_extend_transform(input, inverse)

if inverse == 0
    output{1} = input;
    output{2} = imrotate(input, 90);
    output{3} = imrotate(input, 90*2);
    output{4} = imrotate(input, 90*3);
    output{5} = input';
    output{6} = imrotate(input', 90);
    output{7} = imrotate(input', 90*2);
    output{8} = imrotate(input', 90*3);
    % for i =1:8
    %    figure(i),imshow(output{i},[]);
    % end
    
else
    output = input{1};
    output = output+imrotate(input{2}, -90);
    output = output+imrotate(input{3}, -90*2);
    output = output+imrotate(input{4}, -90*3);
    output = output+input{5}';
    temp = imrotate(input{6}, -90);
    output = output+temp';
    temp = imrotate(input{7}, -90*2);
    output = output+temp';
    temp = imrotate(input{8}, -90*3);
    output = output+temp';
    output = output./8;

end

end