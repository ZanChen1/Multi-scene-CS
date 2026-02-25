function b = f_handleA(x, A, At, mode)


switch mode
    case 1
        b = A(x)';
        
    case 2
        b = At(x);
        
    otherwise
        error('Unknown mode passed to f_handleA in ftv_cs.m');
end

end