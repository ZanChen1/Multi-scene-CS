function tar_range = region_search(tar, range)
    for i=1:size(range,1)
       if tar>=range(i,1)&&tar<=range(i,2)
          break;
       end
    end
    tar_range = i;
end