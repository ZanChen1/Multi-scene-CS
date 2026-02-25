function [b_h, b_w, h, w] = blocksize_urban100_set(h, w)
%%
switch w
    case 564
        w = 560;
    case 684
        w = 680;
    case 692
        w = 688;
    case 700
        w = 696;
    case 740
        w = 736;  
    case 796
        w = 792;
    case 1004
        w = 1000;
end

switch h
    case 564
        h = 560;
    case 572
        h = 568;
    case 644
        h = 640;           
    case 660
        h = 656;
    case 676
        h = 672;
    case 684
        h = 680;
    case 692
        h = 688;
    case 700
        h = 696; 
    case 724
        h = 720;
    case 732
        h = 728;
    case 764
        h = 760;
    case 772
        h = 768;
    case 796
        h = 792;
    case 804
        h = 800;
    case 980
        h = 976;
    case 1036
        h = 1032;
end

%%
switch w
    case 560
        b_w = 70;
    case 680
        b_w = 68;
    case 688
        b_w = 86;
    case 696
        b_w = 87;
    case 736
        b_w = 92;
    case 768
        b_w = 64;
    case 776
        b_w = 97;
    case 792
        b_w = 99;
    case 1000
        b_w = 250;
    case 1024
        b_w = 32;
    case 1200
        b_w = 8;
    case 1280
        b_w = 80;
    otherwise
        b_w = 64;
end

switch h
    case 560
        b_h = 70;
    case 568
        b_h = 71;
    case 576
        b_h = 64;
    case 616
        b_h = 56;
    case 656
        b_h = 41;
    case 664
        b_h = 83;
    case 672
        b_h = 96;
    case 680
        b_h = 68;
    case 688
        b_h = 86;
    case 696
        b_h = 87;
    case 720
        b_h = 90;
    case 728
        b_h = 91;
    case 736
        b_h = 92;
    case 760
        b_h = 95;
    case 768
        b_h = 64;
    case 776
        b_h = 97;
    case 792
        b_h = 99;
    case 800
        b_h = 80;
    case 816
        b_h = 102;
    case 824
        b_h = 103;
    case 832
        b_h = 52;
    case 864
        b_h = 54;
    case 960
        b_h = 80;
    case 976
        b_h = 61;
    case 1024
        b_h = 64;
    case 1200
        b_h = 75;
    case 1032
        b_h = 86;
    otherwise
        b_h = 64;
end

if h == 796 && w == 1200
    b_h = 8;
    b_w = 199;
end


end