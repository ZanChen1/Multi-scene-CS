function [U, out] = TVNLR_preprocess(U,opts)

opts.TVnorm = 1;
opts.nonneg = true;
global D Dt
[D,Dt] = defDDt;
% problem dimension
p = opts.row;
q = opts.col;
% mark important constants
mu = opts.mu;

% initialize U, beta
muf = mu;

if mu > muf
    mu = muf;
end


for ii = 1
    %% TV norm
    [Ux,Uy] = D(U);
    alpha = 1;
    % if back tracking is succeceful, then recompute
    if alpha ~= 0
        Uxbar = Ux;
        Uybar = Uy;
        if opts.TVnorm == 1
            % ONE-DIMENSIONAL SHRINKAGE STEP
            Wx = max(abs(Uxbar) - 1.6, 0).*sign(Uxbar);
            Wy = max(abs(Uybar) - 1.6, 0).*sign(Uybar);
        end
        % update parameters related to Wx, Wy
        Vx = Ux - Wx;
        Vy = Uy - Wy;
        g2 = Dt(Vx,Vy);
    end
    %% NLM norm
    NLR_flag = 1;
    if NLR_flag
        Options.kernelratio= 3;
        Options.windowratio= 6;
        Options.verbose=0;
        Options.filterstrength=0.03;
        J=NLMF(U/255,Options);
        X = double(uint8(J*255));
    end
    g3 = (U(:)-X(:));
    %% compute gradient
    theta = 0.8;
    d = (1-theta)*g2 + theta*g3;
    
    %% Set Steepest Descent step length
    tau = 0.21;
    
    %% ONE-STEP GRADIENT DESCENT %
    taud = tau*d;
    U = U(:) - taud;
    U = reshape(U,p,q);
    %U = X;
end

end






